// FacetGlass - OpenFX plugin.
// SPDX-License-Identifier: MIT
//
// Ray-traced solid glass in front of the lens. The shading maths lives in
// src/core/fg3d_core.h; shapes, the BVH and the UI -> parameter conversion live in
// src/host. This file describes the plugin, reads parameters, caches the scene and
// runs the render on CUDA (when the host offers it) or on the CPU.

#include <algorithm>
#include <cfloat>
#include <climits>
#include <cmath>
#include <cstring>
#include <memory>
#include <mutex>

#include "ofxsImageEffect.h"
#include "ofxsMultiThread.h"
#include "ofxsProcessing.h"

#include "fg3d_cpu.h"
#ifdef FG_HAVE_CUDA
#include "fg3d_cuda.h"
#endif

#define kPluginName "FacetGlass"
#define kPluginGrouping "FacetGlass"
#define kPluginDescription                                                                                   \
    "Ray-traced glass held in front of the lens: prisms, diopters, gems and crystals as solid 3D objects, " \
    "with internal reflections, bevelled edges, dispersion, lens depth of field and virtual lights.\n"     \
    "Free and open source (MIT)."
#define kPluginIdentifier "com.facetglass.FacetGlass"
#define kPluginVersionMajor 1
#define kPluginVersionMinor 1

#define kClipRefract "RefractSource"

// Parameter script names: stored in saved projects, never rename.
#define P_SHAPE "shape"
#define P_FACETS "facets"
#define P_BOTH "bothSides"
#define P_PRISM "prismAngle"
#define P_CLEAR "clearCentre"
#define P_POWER "diopterPower"
#define P_THICK "thickness"
#define P_DETAIL "detail"
#define P_SEED "seed"
#define P_SMOOTH "smoothAngle"
#define P_OBJ "objFile"
#define P_CENTER "center"
#define P_DEPTH "depth"
#define P_TILT "tilt"
#define P_TURN "turn"
#define P_ROLL "roll"
#define P_SIZE "size"
#define P_IOR "ior"
#define P_DISP "chromaticAberration"
#define P_REFL "reflection"
#define P_ROUGH "roughness"
#define P_BEVEL "bevelWidth"
#define P_BEVANG "bevelAngle"
#define P_BOUNCES "bounceLimit"
#define P_FSTOP "fStop"
#define P_FOCUS "focus"
#define P_GFOCUS "glassFocus"
#define P_WORLD "outsideFrame"
#define P_WBLUR "worldBlur"
#define P_FEATHER "feather"
#define P_WOBBLE "wobble"
#define P_WSPEED "wobbleSpeed"
#define P_BOOST "highlightBoost"
#define P_THRESH "highlightThreshold"
#define P_MIX "mix"
#define P_CSPACE "inputColorSpace"
#define P_QUALITY "quality"
#define P_MINS "minSamples"
#define P_MAXS "maxSamples"
#define P_NOISE "noiseThreshold"
#define kNumLights 2
static const char* kLightPos[kNumLights] = { "light1Position", "light2Position" };
static const char* kLightDepth[kNumLights] = { "light1Depth", "light2Depth" };
static const char* kLightSize[kNumLights] = { "light1Size", "light2Size" };
static const char* kLightKelvin[kNumLights] = { "light1Kelvin", "light2Kelvin" };
static const char* kLightInt[kNumLights] = { "light1Intensity", "light2Intensity" };

enum { kQualityDraft = 0, kQualityGood, kQualityFinal, kQualityCustom };

using namespace OFX;

namespace {

int choiceAt(ChoiceParam* c, double t) { int v = 0; c->getValueAtTime(t, v); return v; }
OfxPointD pointAt(Double2DParam* d, double t) { OfxPointD v; d->getValueAtTime(t, v.x, v.y); return v; }

OfxRectI toPixelEnclosing(const OfxRectD& r, const OfxPointD& rs, double par)
{
    OfxRectI p;
    p.x1 = (int)std::floor(r.x1 * rs.x / par);
    p.y1 = (int)std::floor(r.y1 * rs.y);
    p.x2 = (int)std::ceil(r.x2 * rs.x / par);
    p.y2 = (int)std::ceil(r.y2 * rs.y);
    return p;
}

// Which Shape controls matter for each shape (for enabling them in the UI).
struct ShapeUse { bool facets, both, prism, clear, power, thick, detail, seed, smooth, obj; };
ShapeUse shapeUse(int s)
{
    using namespace fg3d;
    ShapeUse u = {};
    switch (s) {
    case kFacetPlate: u.facets = u.both = u.prism = u.thick = true; break;
    case kWedgePrism: case kBox: case kHexPrism: u.thick = true; break;
    case kRadialPrism: case kHalfRadial: u.facets = u.prism = u.clear = u.thick = true; break;
    case kKaleidoscope: u.facets = u.thick = true; break;
    case kHaloRing: u.prism = u.clear = u.thick = true; break;
    case kRippleRings: case kHalfRipple: case kHoneycomb: u.facets = u.prism = u.thick = true; break;
    case kFresnelLens: u.facets = u.power = u.thick = true; break;
    case kRoundDiopter: case kSplitDiopter: case kDoubleConvex: case kCylinderLens: u.power = u.thick = true; break;
    case kCrystalCluster: case kShards: u.facets = u.seed = true; break;
    case kGeoSphere: u.detail = true; break;
    case kWavyGlass: case kFlutedGlass: case kBubbleGlass: case kWaterDrops:
        u.facets = u.prism = u.thick = u.seed = true; break;
    case kIceCube: case kMeltedBlob: u.seed = true; break;
    case kTwistedBar: u.prism = true; break;
    case kBentSlab: u.prism = u.thick = true; break;
    case kCustomOBJ: u.smooth = u.obj = true; break;
    default: break;
    }
    return u;
}

}  // namespace

////////////////////////////////////////////////////////////////////////////////
// CPU processor: runs fg3d_shade over the render window on the host's threads.

class CpuProcessor : public ImageProcessor
{
public:
    CpuProcessor(ImageEffect& e, const FG3DParams& p, const float* src, const float* gls,
                 const fg3d::SceneBuffers& s)
        : ImageProcessor(e), _p(p), _src(src), _gls(gls), _s(s) {}

    void multiThreadProcessImages(OfxRectI win) override
    {
        for (int y = win.y1; y < win.y2; ++y) {
            if (_effect.abort()) break;
            fg3d::shadeCPU(_p, _src, _gls, _s, win.x1, win.x2, y, y + 1,
                           [&](int yy) { return static_cast<float*>(_dstImg->getPixelAddress(win.x1, yy)); });
        }
    }

private:
    FG3DParams _p;
    const float* _src;
    const float* _gls;
    const fg3d::SceneBuffers& _s;
};

////////////////////////////////////////////////////////////////////////////////

class FacetGlassPlugin : public ImageEffect
{
public:
    explicit FacetGlassPlugin(OfxImageEffectHandle handle);

    void render(const RenderArguments& args) override;
    bool isIdentity(const IsIdentityArguments& args, Clip*& identityClip, double& identityTime) override;
    bool getRegionOfDefinition(const RegionOfDefinitionArguments& args, OfxRectD& rod) override;
    void getRegionsOfInterest(const RegionsOfInterestArguments& args, RegionOfInterestSetter& rois) override;
    void changedParam(const InstanceChangedArgs& args, const std::string& paramName) override;

private:
    void readUI(double t, fg3d::UIParams& ui);
    std::shared_ptr<const fg3d::SceneBuffers> scene(const fg3d::UIParams& ui);
    void updateVisibility();
    bool refractConnected() const { return _refract && _refract->isConnected(); }

    Clip *_dst, *_src, *_refract;

    ChoiceParam *_shape, *_world, *_cspace, *_quality;
    IntParam *_facets, *_detail, *_seed, *_bounces, *_minS, *_maxS;
    StringParam* _obj;
    BooleanParam* _both;
    DoubleParam *_prism, *_clear, *_power, *_thick, *_depth, *_tilt, *_turn, *_roll, *_size, *_ior, *_disp, *_refl,
        *_rough, *_bevel, *_bevAng, *_fstop, *_focus, *_gfocus, *_wblur, *_feather, *_wobble, *_wspeed, *_boost,
        *_thresh, *_mix, *_noise, *_smooth;
    Double2DParam* _center;
    Double2DParam* _lpos[kNumLights];
    DoubleParam *_ldepth[kNumLights], *_lsize[kNumLights], *_lkelvin[kNumLights], *_lint[kNumLights];

    std::mutex _sceneMutex;
    fg3d::ShapeParams _sceneKey;
    float _sceneBevel = -1.0f;
    std::shared_ptr<const fg3d::SceneBuffers> _scene;
};

FacetGlassPlugin::FacetGlassPlugin(OfxImageEffectHandle handle) : ImageEffect(handle), _refract(nullptr)
{
    _dst = fetchClip(kOfxImageEffectOutputClipName);
    _src = fetchClip(kOfxImageEffectSimpleSourceClipName);
    if (getContext() == eContextGeneral) _refract = fetchClip(kClipRefract);

    _shape = fetchChoiceParam(P_SHAPE);
    _world = fetchChoiceParam(P_WORLD);
    _cspace = fetchChoiceParam(P_CSPACE);
    _quality = fetchChoiceParam(P_QUALITY);
    _facets = fetchIntParam(P_FACETS);
    _detail = fetchIntParam(P_DETAIL);
    _seed = fetchIntParam(P_SEED);
    _bounces = fetchIntParam(P_BOUNCES);
    _obj = fetchStringParam(P_OBJ);
    _smooth = fetchDoubleParam(P_SMOOTH);
    _minS = fetchIntParam(P_MINS);
    _maxS = fetchIntParam(P_MAXS);
    _both = fetchBooleanParam(P_BOTH);
    _prism = fetchDoubleParam(P_PRISM);
    _clear = fetchDoubleParam(P_CLEAR);
    _power = fetchDoubleParam(P_POWER);
    _thick = fetchDoubleParam(P_THICK);
    _depth = fetchDoubleParam(P_DEPTH);
    _tilt = fetchDoubleParam(P_TILT);
    _turn = fetchDoubleParam(P_TURN);
    _roll = fetchDoubleParam(P_ROLL);
    _size = fetchDoubleParam(P_SIZE);
    _ior = fetchDoubleParam(P_IOR);
    _disp = fetchDoubleParam(P_DISP);
    _refl = fetchDoubleParam(P_REFL);
    _rough = fetchDoubleParam(P_ROUGH);
    _bevel = fetchDoubleParam(P_BEVEL);
    _bevAng = fetchDoubleParam(P_BEVANG);
    _fstop = fetchDoubleParam(P_FSTOP);
    _focus = fetchDoubleParam(P_FOCUS);
    _gfocus = fetchDoubleParam(P_GFOCUS);
    _wblur = fetchDoubleParam(P_WBLUR);
    _feather = fetchDoubleParam(P_FEATHER);
    _wobble = fetchDoubleParam(P_WOBBLE);
    _wspeed = fetchDoubleParam(P_WSPEED);
    _boost = fetchDoubleParam(P_BOOST);
    _thresh = fetchDoubleParam(P_THRESH);
    _mix = fetchDoubleParam(P_MIX);
    _noise = fetchDoubleParam(P_NOISE);
    _center = fetchDouble2DParam(P_CENTER);
    for (int i = 0; i < kNumLights; ++i) {
        _lpos[i] = fetchDouble2DParam(kLightPos[i]);
        _ldepth[i] = fetchDoubleParam(kLightDepth[i]);
        _lsize[i] = fetchDoubleParam(kLightSize[i]);
        _lkelvin[i] = fetchDoubleParam(kLightKelvin[i]);
        _lint[i] = fetchDoubleParam(kLightInt[i]);
    }
    updateVisibility();
}

void FacetGlassPlugin::updateVisibility()
{
    const bool custom = choiceAt(_quality, 0.0) == kQualityCustom;
    _minS->setIsSecret(!custom);
    _maxS->setIsSecret(!custom);
    _noise->setIsSecret(!custom);
    const ShapeUse u = shapeUse(choiceAt(_shape, 0.0));
    _facets->setEnabled(u.facets);
    _both->setEnabled(u.both);
    _prism->setEnabled(u.prism);
    _clear->setEnabled(u.clear);
    _power->setEnabled(u.power);
    _thick->setEnabled(u.thick);
    _detail->setEnabled(u.detail);
    _smooth->setEnabled(u.smooth);
    _obj->setEnabled(u.obj);
}

void FacetGlassPlugin::changedParam(const InstanceChangedArgs&, const std::string& name)
{
    if (name == P_QUALITY || name == P_SHAPE) updateVisibility();
    if (name == P_OBJ) {
        std::lock_guard<std::mutex> lock(_sceneMutex);
        _scene.reset();  // the file may have changed on disk
    }
}

void FacetGlassPlugin::readUI(double t, fg3d::UIParams& ui)
{
    ui.shape.shape = choiceAt(_shape, t);
    ui.shape.facets = _facets->getValueAtTime(t);
    ui.shape.bothSides = _both->getValueAtTime(t);
    ui.shape.prismAngleDeg = (float)_prism->getValueAtTime(t);
    ui.shape.clearCentre = (float)_clear->getValueAtTime(t);
    ui.shape.power = (float)_power->getValueAtTime(t);
    ui.shape.thickness = (float)_thick->getValueAtTime(t);
    ui.shape.detail = _detail->getValueAtTime(t);
    ui.seed = ui.shape.seed = _seed->getValueAtTime(t);
    ui.shape.smoothAngleDeg = (float)_smooth->getValueAtTime(t);
    if (ui.shape.shape == fg3d::kCustomOBJ) _obj->getValue(ui.shape.objPath);
    const OfxPointD c = pointAt(_center, t);
    ui.centerX = c.x; ui.centerY = c.y;
    ui.depth = _depth->getValueAtTime(t);
    ui.rotX = _tilt->getValueAtTime(t);
    ui.rotY = _turn->getValueAtTime(t);
    ui.rotZ = _roll->getValueAtTime(t);
    ui.size = _size->getValueAtTime(t);
    ui.ior = _ior->getValueAtTime(t);
    ui.dispersion = _disp->getValueAtTime(t);
    ui.reflection = _refl->getValueAtTime(t);
    ui.roughness = _rough->getValueAtTime(t);
    ui.bevelWidth = _bevel->getValueAtTime(t);
    ui.bevelAngle = _bevAng->getValueAtTime(t);
    ui.maxBounces = _bounces->getValueAtTime(t);
    ui.fstop = _fstop->getValueAtTime(t);
    ui.focus = _focus->getValueAtTime(t);
    ui.glassFocus = _gfocus->getValueAtTime(t);
    ui.worldMode = choiceAt(_world, t);
    ui.worldBlur = _wblur->getValueAtTime(t);
    ui.feather = _feather->getValueAtTime(t);
    ui.wobble = _wobble->getValueAtTime(t);
    ui.wobbleSpeed = _wspeed->getValueAtTime(t);
    ui.boost = _boost->getValueAtTime(t);
    ui.threshold = _thresh->getValueAtTime(t);
    ui.mix = _mix->getValueAtTime(t);
    ui.colorSpace = choiceAt(_cspace, t);
    switch (choiceAt(_quality, t)) {
    case kQualityDraft: ui.minSamples = 4; ui.maxSamples = 8; ui.noiseThreshold = 0.05; break;
    case kQualityGood: ui.minSamples = 12; ui.maxSamples = 48; ui.noiseThreshold = 0.03; break;
    case kQualityFinal: ui.minSamples = 48; ui.maxSamples = 384; ui.noiseThreshold = 0.008; break;
    default:
        ui.minSamples = _minS->getValueAtTime(t);
        ui.maxSamples = _maxS->getValueAtTime(t);
        ui.noiseThreshold = _noise->getValueAtTime(t);
    }
    ui.numLights = 0;
    for (int i = 0; i < kNumLights; ++i) {
        fg3d::UIParams::Light& L = ui.lights[ui.numLights];
        const OfxPointD lp = pointAt(_lpos[i], t);
        L.x = lp.x; L.y = lp.y;
        L.depth = _ldepth[i]->getValueAtTime(t);
        L.size = _lsize[i]->getValueAtTime(t);
        L.kelvin = _lkelvin[i]->getValueAtTime(t);
        L.intensity = _lint[i]->getValueAtTime(t);
        if (L.intensity > 0) ++ui.numLights;
    }
    ui.shape.worldScale = (float)fg3d::objectScale(ui);
    ui.shape.depth = (float)ui.depth;
    // Only lenses depend on scale and depth: keep other shapes' cache keys stable.
    const int sh = ui.shape.shape;
    if (sh != fg3d::kSplitDiopter && sh != fg3d::kRoundDiopter && sh != fg3d::kDoubleConvex &&
        sh != fg3d::kCylinderLens && sh != fg3d::kFresnelLens) {
        ui.shape.worldScale = 0.0f;
        ui.shape.depth = 0.0f;
    }
}

std::shared_ptr<const fg3d::SceneBuffers> FacetGlassPlugin::scene(const fg3d::UIParams& ui)
{
    std::lock_guard<std::mutex> lock(_sceneMutex);
    if (!_scene || !(_sceneKey == ui.shape) || _sceneBevel != (float)ui.bevelAngle) {
        auto s = std::make_shared<fg3d::SceneBuffers>();
        std::string err;
        fg3d::buildScene(fg3d::makeShape(ui.shape, &err), (float)ui.bevelAngle, *s);
        if (!err.empty()) setPersistentMessage(Message::eMessageWarning, "", "FacetGlass: " + err);
        else clearPersistentMessage();
        _scene = s;
        _sceneKey = ui.shape;
        _sceneBevel = (float)ui.bevelAngle;
    }
    return _scene;
}

bool FacetGlassPlugin::isIdentity(const IsIdentityArguments& args, Clip*& identityClip, double& identityTime)
{
    if (_mix->getValueAtTime(args.time) == 0.0) {
        identityClip = _src;
        identityTime = args.time;
        return true;
    }
    return false;
}

bool FacetGlassPlugin::getRegionOfDefinition(const RegionOfDefinitionArguments& args, OfxRectD& rod)
{
    rod = _src->getRegionOfDefinition(args.time);
    return true;
}

// Rays can land anywhere in the frame (and the world blur reads all of it).
void FacetGlassPlugin::getRegionsOfInterest(const RegionsOfInterestArguments& args, RegionOfInterestSetter& rois)
{
    rois.setRegionOfInterest(*_src, _src->getRegionOfDefinition(args.time));
    if (refractConnected()) rois.setRegionOfInterest(*_refract, _refract->getRegionOfDefinition(args.time));
}

void FacetGlassPlugin::render(const RenderArguments& args)
{
    if (_dst->getPixelDepth() != eBitDepthFloat || _dst->getPixelComponents() != ePixelComponentRGBA)
        throwSuiteStatusException(kOfxStatErrUnsupported);
    const double t = args.time;
    std::unique_ptr<Image> dst(_dst->fetchImage(t));
    std::unique_ptr<Image> src(_src->fetchImage(t));
    if (!dst || !src) throwSuiteStatusException(kOfxStatFailed);
    if (src->getPixelDepth() != eBitDepthFloat || src->getPixelComponents() != ePixelComponentRGBA)
        throwSuiteStatusException(kOfxStatErrImageFormat);
    std::unique_ptr<Image> gls;
    if (refractConnected()) {
        gls.reset(_refract->fetchImage(t));
        if (gls && (gls->getPixelDepth() != eBitDepthFloat || gls->getPixelComponents() != ePixelComponentRGBA))
            throwSuiteStatusException(kOfxStatErrImageFormat);
    }

    const OfxRectI frame = toPixelEnclosing(_src->getRegionOfDefinition(t), args.renderScale, _src->getPixelAspectRatio());
    const int fw = frame.x2 - frame.x1, fh = frame.y2 - frame.y1;
    if (fw < 1 || fh < 1) throwSuiteStatusException(kOfxStatFailed);
    OfxRectI gframe = frame;
    if (gls) gframe = toPixelEnclosing(_refract->getRegionOfDefinition(t), args.renderScale, _refract->getPixelAspectRatio());

    fg3d::UIParams ui;
    readUI(t, ui);
    std::shared_ptr<const fg3d::SceneBuffers> sc = scene(ui);
    FG3DParams p;
    fg3d::makeParams(ui, (float)frame.x1, (float)frame.y1, (float)fw, (float)fh, t, *sc, p);

    const OfxRectI sb = src->getBounds();
    const int srcStride = src->getRowBytes() / 4;

#ifdef FG_HAVE_CUDA
    if (args.isEnabledCudaRender) {
        fg3d::CudaFrame f;
        f.stream = args.pCudaStream;
        f.src.data = static_cast<const float*>(src->getPixelData());
        f.src.x1 = sb.x1; f.src.y1 = sb.y1; f.src.w = sb.x2 - sb.x1; f.src.h = sb.y2 - sb.y1; f.src.stride = srcStride;
        if (gls) {
            const OfxRectI gb = gls->getBounds();
            f.gls.data = static_cast<const float*>(gls->getPixelData());
            f.gls.x1 = gb.x1; f.gls.y1 = gb.y1; f.gls.w = gb.x2 - gb.x1; f.gls.h = gb.y2 - gb.y1;
            f.gls.stride = gls->getRowBytes() / 4;
            f.glsFrameX = gframe.x1; f.glsFrameY = gframe.y1;
            f.glsFrameW = gframe.x2 - gframe.x1; f.glsFrameH = gframe.y2 - gframe.y1;
        }
        f.frameX = frame.x1; f.frameY = frame.y1; f.frameW = fw; f.frameH = fh;
        const OfxRectI db = dst->getBounds();
        f.dst = static_cast<float*>(dst->getPixelData());
        f.dstX1 = db.x1; f.dstY1 = db.y1; f.dstStride = dst->getRowBytes() / 4;
        f.winX1 = args.renderWindow.x1; f.winY1 = args.renderWindow.y1;
        f.winX2 = args.renderWindow.x2; f.winY2 = args.renderWindow.y2;
        if (const char* err = fg3d::renderCUDA(f, p, *sc)) {
            setPersistentMessage(Message::eMessageError, "", std::string("FacetGlass CUDA: ") + err);
            throwSuiteStatusException(kOfxStatFailed);
        }
        return;
    }
#endif
    if (args.isEnabledCudaRender || args.isEnabledOpenCLRender || args.isEnabledMetalRender)
        throwSuiteStatusException(kOfxStatErrUnsupported);

    // CPU: linear pyramids for the source (and refract source), then shade.
    std::vector<float> srcPyr, glsPyr;
    int levels = 0, glevels = 0;
    fg3d::buildPyramidCPU(static_cast<const float*>(src->getPixelAddress(sb.x1, sb.y1)), sb.x1, sb.y1, sb.x2 - sb.x1,
                          sb.y2 - sb.y1, srcStride, frame.x1, frame.y1, fw, fh, ui.colorSpace, srcPyr, levels);
    p.srcW = fw; p.srcH = fh; p.srcLevels = levels;
    p.glsW = fw; p.glsH = fh; p.glsLevels = levels; p.useSource = 0;
    if (gls) {
        const OfxRectI gb = gls->getBounds();
        const int gw = gframe.x2 - gframe.x1, gh = gframe.y2 - gframe.y1;
        fg3d::buildPyramidCPU(static_cast<const float*>(gls->getPixelAddress(gb.x1, gb.y1)), gb.x1, gb.y1,
                              gb.x2 - gb.x1, gb.y2 - gb.y1, gls->getRowBytes() / 4, gframe.x1, gframe.y1, gw, gh,
                              ui.colorSpace, glsPyr, glevels);
        p.glsW = gw; p.glsH = gh; p.glsLevels = glevels; p.useSource = 1;
    }
    CpuProcessor proc(*this, p, srcPyr.data(), gls ? glsPyr.data() : srcPyr.data(), *sc);
    proc.setDstImg(dst.get());
    proc.setRenderWindow(args.renderWindow);
    proc.process();
}

////////////////////////////////////////////////////////////////////////////////
// Factory

class FacetGlassFactory : public PluginFactoryHelper<FacetGlassFactory>
{
public:
    FacetGlassFactory() : PluginFactoryHelper<FacetGlassFactory>(kPluginIdentifier, kPluginVersionMajor, kPluginVersionMinor) {}
    void describe(ImageEffectDescriptor& desc) override;
    void describeInContext(ImageEffectDescriptor& desc, ContextEnum context) override;
    ImageEffect* createInstance(OfxImageEffectHandle handle, ContextEnum) override { return new FacetGlassPlugin(handle); }
};

void FacetGlassFactory::describe(ImageEffectDescriptor& desc)
{
    desc.setLabels(kPluginName, kPluginName, kPluginName);
    desc.setPluginGrouping(kPluginGrouping);
    desc.setPluginDescription(kPluginDescription);
    desc.addSupportedContext(eContextFilter);
    desc.addSupportedContext(eContextGeneral);
    desc.addSupportedBitDepth(eBitDepthFloat);
    desc.setSingleInstance(false);
    desc.setHostFrameThreading(false);
    desc.setSupportsMultiResolution(true);
    desc.setSupportsTiles(true);
    desc.setTemporalClipAccess(false);
    desc.setRenderTwiceAlways(false);
    desc.setSupportsMultipleClipPARs(false);
    desc.setRenderThreadSafety(eRenderFullySafe);
#ifdef FG_HAVE_CUDA
    desc.setSupportsCudaRender(true);
    desc.setSupportsCudaStream(true);
#endif
}

namespace {

PageParamDescriptor* g_page = nullptr;

GroupParamDescriptor* group(ImageEffectDescriptor& d, const char* name, const char* label, bool open = true)
{
    GroupParamDescriptor* g = d.defineGroupParam(name);
    g->setLabels(label, label, label);
    g->setOpen(open);
    if (g_page) g_page->addChild(*g);
    return g;
}

template <class T>
T* finish(T* p, const char* label, const char* hint, GroupParamDescriptor* g)
{
    p->setLabels(label, label, label);
    p->setHint(hint);
    p->setParent(*g);
    if (g_page) g_page->addChild(*p);
    return p;
}

DoubleParamDescriptor* dbl(ImageEffectDescriptor& d, const char* name, const char* label, const char* hint,
                           GroupParamDescriptor* g, double def, double dmin, double dmax, double mn = -DBL_MAX,
                           double mx = DBL_MAX)
{
    DoubleParamDescriptor* p = d.defineDoubleParam(name);
    p->setDefault(def);
    p->setRange(mn, mx);
    p->setDisplayRange(dmin, dmax);
    p->setIncrement((dmax - dmin) / 100.0);
    p->setDigits(3);
    return finish(p, label, hint, g);
}

IntParamDescriptor* intp(ImageEffectDescriptor& d, const char* name, const char* label, const char* hint,
                         GroupParamDescriptor* g, int def, int dmin, int dmax, int mn, int mx)
{
    IntParamDescriptor* p = d.defineIntParam(name);
    p->setDefault(def);
    p->setRange(mn, mx);
    p->setDisplayRange(dmin, dmax);
    return finish(p, label, hint, g);
}

Double2DParamDescriptor* point(ImageEffectDescriptor& d, const char* name, const char* label, const char* hint,
                               GroupParamDescriptor* g, double x, double y)
{
    Double2DParamDescriptor* p = d.defineDouble2DParam(name);
    p->setDefault(x, y);
    p->setDisplayRange(0.0, 0.0, 1.0, 1.0);
    p->setIncrement(0.001);
    p->setDigits(3);
    p->setDimensionLabels("x", "y");
    return finish(p, label, hint, g);
}

ChoiceParamDescriptor* choice(ImageEffectDescriptor& d, const char* name, const char* label, const char* hint,
                              GroupParamDescriptor* g, std::initializer_list<const char*> opts, int def)
{
    ChoiceParamDescriptor* p = d.defineChoiceParam(name);
    for (const char* o : opts) p->appendOption(o);
    p->setDefault(def);
    return finish(p, label, hint, g);
}

}  // namespace

void FacetGlassFactory::describeInContext(ImageEffectDescriptor& desc, ContextEnum context)
{
    ClipDescriptor* src = desc.defineClip(kOfxImageEffectSimpleSourceClipName);
    src->addSupportedComponent(ePixelComponentRGBA);
    src->setTemporalClipAccess(false);
    src->setSupportsTiles(true);
    src->setIsMask(false);
    if (context == eContextGeneral) {
        ClipDescriptor* g = desc.defineClip(kClipRefract);
        g->setLabels("Refract Source", "Refract", "Refract Source");
        g->addSupportedComponent(ePixelComponentRGBA);
        g->setTemporalClipAccess(false);
        g->setSupportsTiles(true);
        g->setIsMask(false);
        g->setOptional(true);
    }
    ClipDescriptor* dst = desc.defineClip(kOfxImageEffectOutputClipName);
    dst->addSupportedComponent(ePixelComponentRGBA);
    dst->setSupportsTiles(true);

    g_page = desc.definePageParam("Controls");

    // ---------------------------------------------------------------- Shape
    GroupParamDescriptor* gs = group(desc, "shapeGroup", "Shape");
    {
        ChoiceParamDescriptor* c = desc.defineChoiceParam(P_SHAPE);
        for (int i = 0; i < fg3d::kShapeCount; ++i) c->appendOption(fg3d::shapeName(i));
        c->setDefault(fg3d::kGem);
        finish(c, "Shape", "The glass object held in front of the lens.", gs);
    }
    intp(desc, P_FACETS, "Facets",
         "Count for the shape: plate steps, prism sides, ripples, honeycomb cells, Fresnel zones, crystals, "
         "shards, flutes, bubbles or drops.", gs, 3, 1, 12, 1, 64);
    finish(desc.defineBooleanParam(P_BOTH), "Both Sides", "Faceted Plate: facets on both sides of the centre.", gs);
    dbl(desc, P_PRISM, "Prism Angle",
        "Steepness for the shape (degrees): facet angle, ripple and lenslet strength, waviness, bend or twist.", gs,
        6.0, 0.0, 30.0, 0.0, 70.0);
    dbl(desc, P_CLEAR, "Clear Centre", "Radial Prism: size of the flat centre.", gs, 0.4, 0.0, 0.95, 0.0, 0.95);
    dbl(desc, P_POWER, "Diopter Power",
        "Diopters: magnification through the lens centre (+10 = 2x). Negative values shrink.", gs, 4.0, -10.0, 10.0,
        -30.0, 30.0);
    dbl(desc, P_THICK, "Thickness", "Thickness of plates, prisms and lenses.", gs, 0.3, 0.05, 1.0, 0.02, 3.0);
    intp(desc, P_DETAIL, "Detail", "Geodesic Sphere: subdivisions (more = rounder).", gs, 1, 0, 4, 0, 4);
    intp(desc, P_SEED, "Seed", "Variation of random shapes and of the sampling noise.", gs, 1, 0, 100, 0, 100000);
    {
        StringParamDescriptor* o = desc.defineStringParam(P_OBJ);
        o->setStringType(eStringTypeFilePath);
        o->setFilePathExists(true);
        o->setDefault("");
        o->setAnimates(false);
        finish(o, "OBJ File", "Custom OBJ: a .obj model (e.g. exported from Blender). It is centred and scaled.", gs);
    }
    dbl(desc, P_SMOOTH, "Smooth Angle", "Custom OBJ: edges flatter than this angle (degrees) are shaded smooth.", gs,
        30.0, 0.0, 180.0, 0.0, 180.0);

    // ---------------------------------------------------------------- Placement
    GroupParamDescriptor* gp = group(desc, "placementGroup", "Placement");
    point(desc, P_CENTER, "Center", "Position of the glass in the frame (0..1, y up).", gp, 0.5, 0.5);
    dbl(desc, P_DEPTH, "Depth", "Distance of the glass from the lens; the footage is at 1.", gp, 0.15, 0.02, 0.6, 0.01, 0.95);
    for (const char* n : { P_TILT, P_TURN, P_ROLL }) {
        const char* label = n == std::string(P_TILT) ? "Tilt" : n == std::string(P_TURN) ? "Turn" : "Roll";
        const char* hint = n == std::string(P_TILT) ? "Rotate the glass up/down (degrees)."
                           : n == std::string(P_TURN) ? "Rotate the glass left/right (degrees)."
                                                      : "Rotate the glass around the lens axis (degrees).";
        DoubleParamDescriptor* a = dbl(desc, n, label, hint, gp, 0.0, -180.0, 180.0);
        a->setDoubleType(eDoubleTypeAngle);
    }
    dbl(desc, P_SIZE, "Size", "Size of the glass as a fraction of the frame height.", gp, 0.7, 0.1, 3.0, 0.005, 20.0);

    // ---------------------------------------------------------------- Glass
    GroupParamDescriptor* gg = group(desc, "glassGroup", "Glass");
    dbl(desc, P_IOR, "Strength (IOR)", "Index of refraction: how strongly the glass bends light.", gg, 1.55, 1.0,
        2.5, 1.0, 4.0);
    dbl(desc, P_DISP, "Chromatic Aberration", "Spectral dispersion: rainbow edges.", gg, 0.4, 0.0, 1.5, 0.0, 5.0);
    dbl(desc, P_REFL, "Reflections", "Strength of reflections off the glass faces (1 = physical).", gg, 1.0, 0.0,
        3.0, 0.0, 20.0);
    dbl(desc, P_ROUGH, "Roughness", "Blurs reflections and spreads highlights.", gg, 0.0, 0.0, 0.5, 0.0, 2.0);
    dbl(desc, P_BEVEL, "Bevel Width", "Rounded edges that catch the light (fraction of the object size).", gg, 0.02,
        0.0, 0.15, 0.0, 1.0);
    dbl(desc, P_BEVANG, "Bevel Angle Limit", "Only edges sharper than this angle (degrees) get a bevel.", gg, 20.0,
        0.0, 90.0, 0.0, 180.0);
    intp(desc, P_BOUNCES, "Bounce Limit", "Maximum light bounces inside the glass.", gg, 12, 1, 64, 1, 256);

    // ---------------------------------------------------------------- Optics
    GroupParamDescriptor* go = group(desc, "opticsGroup", "Optics");
    dbl(desc, P_FSTOP, "F-Stop", "Lens aperture: lower blurs the out-of-focus glass more.", go, 4.0, 0.95, 22.0, 0.5, 64.0);
    dbl(desc, P_FOCUS, "Focus", "Focus distance: 0 = on the glass, 1 = on the footage.", go, 1.0, 0.0, 1.0, 0.0, 1.0);
    dbl(desc, P_GFOCUS, "Glass Focus",
        "Focus for what is seen through the glass (0 = glass, 1 = footage), e.g. a split diopter.", go, 1.0, 0.0, 1.0,
        0.0, 1.0);

    // ---------------------------------------------------------------- World
    GroupParamDescriptor* gw = group(desc, "worldGroup", "Outside Frame", false);
    choice(desc, P_WORLD, "Outside Frame", "What light from beyond the frame edge shows.", gw,
           { "Spherical", "Mirror", "Repeat", "Clamp", "Black", "Transparent" }, FG_WORLD_SPHERICAL);
    dbl(desc, P_WBLUR, "World Blur", "Blur of the off-screen surroundings.", gw, 4.0, 0.0, 8.0, 0.0, 12.0);
    dbl(desc, P_FEATHER, "Feather", "Softens the frame edge seen in the glass.", gw, 0.05, 0.0, 0.5, 0.0, 5.0);

    // ---------------------------------------------------------------- Lights
    GroupParamDescriptor* gl = group(desc, "lightsGroup", "Lights", false);
    for (int i = 0; i < kNumLights; ++i) {
        const std::string n = "Light " + std::to_string(i + 1) + " ";
        point(desc, kLightPos[i], (n + "Position").c_str(), "Where the light is (0..1, y up).", gl,
              i == 0 ? 0.25 : 0.75, 0.8);
        dbl(desc, kLightDepth[i], (n + "Depth").c_str(), "Distance of the light from the lens.", gl, 0.3, 0.02, 1.0,
            0.01, 2.0);
        dbl(desc, kLightSize[i], (n + "Size").c_str(), "Size of the light.", gl, 0.1, 0.01, 0.5, 0.001, 5.0);
        dbl(desc, kLightKelvin[i], (n + "Kelvin").c_str(), "Colour temperature.", gl, i == 0 ? 5600 : 3200, 1500,
            12000, 1000, 40000);
        dbl(desc, kLightInt[i], (n + "Intensity").c_str(), "Brightness; 0 = off.", gl, 0.0, 0.0, 10.0, 0.0, 1000.0);
    }

    // ---------------------------------------------------------------- Motion
    GroupParamDescriptor* gm = group(desc, "motionGroup", "Motion", false);
    dbl(desc, P_WOBBLE, "Hand-held", "Amount of hand-held drift, tilt and roll over time.", gm, 0.0, 0.0, 1.0, 0.0, 5.0);
    dbl(desc, P_WSPEED, "Hand-held Speed", "Speed of the hand-held motion.", gm, 1.0, 0.0, 5.0, 0.0, 50.0);

    // ---------------------------------------------------------------- Look
    GroupParamDescriptor* gk = group(desc, "lookGroup", "Look");
    dbl(desc, P_BOOST, "Highlight Boost", "Brightens highlights seen through the glass (sharper bokeh).", gk, 0.5,
        0.0, 4.0, 0.0, 50.0);
    dbl(desc, P_THRESH, "Highlight Threshold", "Linear brightness above which highlights are boosted.", gk, 1.0, 0.0,
        4.0, 0.0, 100.0);
    dbl(desc, P_MIX, "Mix", "Blend between the original (0) and the glass (1).", gk, 1.0, 0.0, 1.0);

    // ---------------------------------------------------------------- Technical
    GroupParamDescriptor* gt = group(desc, "technicalGroup", "Technical", false);
    choice(desc, P_CSPACE, "Input Colour Space", "Encoding of the incoming footage; the glass is traced in linear light.",
           gt, { "Linear", "Rec.709 (Gamma 2.4)", "sRGB", "Sony S-Log3", "ARRI LogC3", "ARRI LogC4", "Panasonic V-Log",
                 "DaVinci Intermediate" }, FG_CS_REC709);
    ChoiceParamDescriptor* q = choice(desc, P_QUALITY, "Quality", "Rays per pixel (adaptive).", gt,
                                      { "Draft", "Good", "Final", "Custom" }, kQualityGood);
    q->setAnimates(false);
    IntParamDescriptor* mn = intp(desc, P_MINS, "Min Samples", "Custom quality: rays per pixel at least.", gt, 16, 1, 128, 1, 4096);
    IntParamDescriptor* mx = intp(desc, P_MAXS, "Max Samples", "Custom quality: rays per pixel at most.", gt, 64, 1, 1024, 1, 16384);
    DoubleParamDescriptor* nz = dbl(desc, P_NOISE, "Noise Threshold",
                                    "Custom quality: stop adding rays when a pixel's noise falls below this.", gt, 0.02,
                                    0.0, 0.1, 0.0, 1.0);
    for (ParamDescriptor* d : { (ParamDescriptor*)mn, (ParamDescriptor*)mx, (ParamDescriptor*)nz }) d->setIsSecret(true);
    mn->setAnimates(false);
    mx->setAnimates(false);
    nz->setAnimates(false);

    g_page = nullptr;
}

void OFX::Plugin::getPluginIDs(PluginFactoryArray& ids)
{
    static FacetGlassFactory factory;
    ids.push_back(&factory);
}
