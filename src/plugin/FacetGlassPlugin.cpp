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
#include <cstdarg>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <memory>
#include <mutex>

#ifdef _WIN32
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>
#endif
#ifdef __APPLE__
#include <OpenGL/gl.h>
#else
#include <GL/gl.h>
#endif

#include "ofxsImageEffect.h"
#include "ofxsInteract.h"
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
#define P_INTERACT "interacting"
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
    // Persistent messages need the host's message suite V2, which not every host
    // (e.g. DaVinci Resolve) provides; the Support library throws when it is missing.
    // A message must never break a render, so failures are ignored.
    void showMessage(const std::string& text)
    {
        try {
            if (text.empty()) clearPersistentMessage();
            else setPersistentMessage(Message::eMessageError, "", text);
        } catch (...) {
        }
    }
    bool refractConnected() const { return _refract && _refract->isConnected(); }

    Clip *_dst, *_src, *_refract;

    ChoiceParam *_shape, *_world, *_cspace, *_quality;
    IntParam *_facets, *_detail, *_seed, *_bounces, *_minS, *_maxS;
    StringParam* _obj;
    BooleanParam *_both, *_interacting;
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
    _interacting = fetchBooleanParam(P_INTERACT);
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
    bool dragging = false;
    _interacting->getValue(dragging);
    switch (dragging ? (int)kQualityDraft : choiceAt(_quality, t)) {
    case kQualityDraft: ui.minSamples = 4; ui.maxSamples = 8; ui.noiseThreshold = 0.05; break;
    case kQualityGood: ui.minSamples = 16; ui.maxSamples = 64; ui.noiseThreshold = 0.025; break;
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
        showMessage(err.empty() ? std::string() : "FacetGlass: " + err);
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
            showMessage(std::string("FacetGlass CUDA: ") + err);
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
// Viewer overlay (gizmo). Colour-coded handles:
//   white centre dot: move          red / green arrows: move along x / y only
//   blue ring: roll                 red / green ellipses: tilt / turn
//   inside the ring: free spin      white square: size
//   purple arc (left): f-stop       orange arc (right): strength (IOR)
//   yellow arc (top): focus         cyan bar (below): depth
//   suns (tinted by Kelvin): lights
// While a handle is dragged a hidden param switches the render to Draft.

namespace ov {

const double kPi = 3.14159265358979323846;

struct Col { float r, g, b; };
const Col kWhite = { 0.95f, 0.95f, 0.95f }, kRed = { 1.0f, 0.36f, 0.36f }, kGreen = { 0.42f, 0.92f, 0.45f },
          kBlue = { 0.38f, 0.62f, 1.0f }, kPurple = { 0.76f, 0.5f, 1.0f }, kOrange = { 1.0f, 0.62f, 0.22f },
          kYellow = { 1.0f, 0.88f, 0.3f }, kCyan = { 0.25f, 0.88f, 0.95f };

void circle(double x, double y, double r, int n = 64)
{
    glBegin(GL_LINE_LOOP);
    for (int i = 0; i < n; ++i) glVertex2d(x + r * std::cos(2 * kPi * i / n), y + r * std::sin(2 * kPi * i / n));
    glEnd();
}
void disc(double x, double y, double r, int n = 20)
{
    glBegin(GL_TRIANGLE_FAN);
    glVertex2d(x, y);
    for (int i = 0; i <= n; ++i) glVertex2d(x + r * std::cos(2 * kPi * i / n), y + r * std::sin(2 * kPi * i / n));
    glEnd();
}
void ellipse(double x, double y, double a, double b, int n = 64)
{
    glBegin(GL_LINE_LOOP);
    for (int i = 0; i < n; ++i) glVertex2d(x + a * std::cos(2 * kPi * i / n), y + b * std::sin(2 * kPi * i / n));
    glEnd();
}
void arc(double x, double y, double r, double a0, double a1, int n = 40)
{
    glBegin(GL_LINE_STRIP);
    for (int i = 0; i <= n; ++i) {
        const double a = (a0 + (a1 - a0) * i / n) * kPi / 180.0;
        glVertex2d(x + r * std::cos(a), y + r * std::sin(a));
    }
    glEnd();
}

// Tiny stroke font (seven segments + a few symbols) for value readouts.
void text(double x, double y, double h, const char* s)
{
    // segments: 0 top, 1 top-right, 2 bottom-right, 3 bottom, 4 bottom-left, 5 top-left, 6 middle
    static const unsigned char digit[10] = { 0x3f, 0x06, 0x5b, 0x4f, 0x66, 0x6d, 0x7d, 0x07, 0x7f, 0x6f };
    const double w = h * 0.55, adv = h * 0.8;
    glBegin(GL_LINES);
    for (const char* c = s; *c; ++c, x += adv) {
        auto seg = [&](int k) {
            const double xs[6][4] = { { 0, 1, 1, 1 }, { 1, 1, 1, 0.5 }, { 1, 0.5, 1, 0 }, { 0, 0, 1, 0 },
                                      { 0, 0.5, 0, 0 }, { 0, 1, 0, 0.5 } };
            if (k == 6) { glVertex2d(x, y + 0.5 * h); glVertex2d(x + w, y + 0.5 * h); return; }
            glVertex2d(x + xs[k][0] * w, y + xs[k][1] * h);
            glVertex2d(x + xs[k][2] * w, y + xs[k][3] * h);
        };
        if (*c >= '0' && *c <= '9') {
            for (int k = 0; k < 7; ++k)
                if (digit[*c - '0'] & (1 << k)) seg(k);
        } else if (*c == '.') {
            glVertex2d(x + 0.1 * w, y); glVertex2d(x + 0.25 * w, y);
            x -= adv * 0.6;
        } else if (*c == '/') {
            glVertex2d(x, y); glVertex2d(x + w, y + h);
        } else if (*c == 'f') {
            glVertex2d(x + 0.2 * w, y); glVertex2d(x + 0.2 * w, y + h);
            glVertex2d(x + 0.2 * w, y + h); glVertex2d(x + w, y + h);
            glVertex2d(x + 0.2 * w, y + 0.55 * h); glVertex2d(x + 0.8 * w, y + 0.55 * h);
        } else if (*c == '-') {
            seg(6);
        } else if (*c == '%') {
            glVertex2d(x, y); glVertex2d(x + w, y + h);
            glVertex2d(x, y + h); glVertex2d(x + 0.2 * w, y + h);
            glVertex2d(x + 0.8 * w, y); glVertex2d(x + w, y);
        }
    }
    glEnd();
}

}  // namespace ov

class GlassOverlay : public OverlayInteract
{
public:
    GlassOverlay(OfxInteractHandle h, ImageEffect* e) : OverlayInteract(h)
    {
        _center = e->fetchDouble2DParam(P_CENTER);
        _size = e->fetchDoubleParam(P_SIZE);
        _tilt = e->fetchDoubleParam(P_TILT);
        _turn = e->fetchDoubleParam(P_TURN);
        _roll = e->fetchDoubleParam(P_ROLL);
        _depth = e->fetchDoubleParam(P_DEPTH);
        _fstop = e->fetchDoubleParam(P_FSTOP);
        _ior = e->fetchDoubleParam(P_IOR);
        _focus = e->fetchDoubleParam(P_FOCUS);
        _interacting = e->fetchBooleanParam(P_INTERACT);
        _src = e->fetchClip(kOfxImageEffectSimpleSourceClipName);
        for (int i = 0; i < kNumLights; ++i) {
            _lpos[i] = e->fetchDouble2DParam(kLightPos[i]);
            _lint[i] = e->fetchDoubleParam(kLightInt[i]);
            _lkel[i] = e->fetchDoubleParam(kLightKelvin[i]);
        }
        for (Param* p : std::initializer_list<Param*>{ _center, _size, _tilt, _turn, _roll, _depth, _fstop, _ior, _focus })
            addParamToSlaveTo(p);
    }

    bool draw(const DrawArgs& args) override
    {
        try { return drawImpl(args); } catch (...) { return false; }
    }
    bool penMotion(const PenArgs& args) override
    {
        try { return motionImpl(args); } catch (...) { return false; }
    }
    bool penDown(const PenArgs& args) override
    {
        try { return downImpl(args); } catch (...) { return false; }
    }
    bool penUp(const PenArgs& args) override
    {
        try { return upImpl(args); } catch (...) { return false; }
    }

private:
    enum Part { kNone, kMove, kMoveX, kMoveY, kRoll, kTilt, kTurn, kSpin, kSize, kFStop, kIOR, kFocus, kDepth, kLight0 };

    // Value ranges shown on the handles.
    static constexpr double kFMin = 0.95, kFMax = 22.0, kIorMin = 1.0, kIorMax = 2.5, kDMin = 0.02, kDMax = 0.6;
    static constexpr double kFA0 = 140, kFA1 = 220;   // f-stop arc (left), degrees
    static constexpr double kIA0 = -40, kIA1 = 40;    // IOR arc (right)
    static constexpr double kCA0 = 130, kCA1 = 50;    // focus arc (top): glass -> footage

    struct Geo { double x1, y1, w, h, cx, cy, r, ps; };

    Geo geo(double t, double ps) const
    {
        Geo g;
        OfxRectD rod = { 0, 0, 0, 0 };
        try { rod = _src->getRegionOfDefinition(t); } catch (...) {}
        if (rod.x2 - rod.x1 < 2 || rod.y2 - rod.y1 < 2) {
            try {
                const OfxPointD off = _effect->getProjectOffset(), sz = _effect->getProjectSize();
                rod = { off.x, off.y, off.x + sz.x, off.y + sz.y };
            } catch (...) { rod = { 0, 0, 1920, 1080 }; }
        }
        g.x1 = rod.x1; g.y1 = rod.y1;
        g.w = std::max(rod.x2 - rod.x1, 1.0); g.h = std::max(rod.y2 - rod.y1, 1.0);
        const OfxPointD c = pointAt(_center, t);
        g.cx = g.x1 + c.x * g.w; g.cy = g.y1 + c.y * g.h;
        g.ps = ps > 0 ? ps : 1.0;
        g.r = std::max(std::max(_size->getValueAtTime(t), 0.005) * 0.5 * g.h, 45.0 * g.ps);
        return g;
    }

    static double clamp01(double v) { return v < 0 ? 0 : (v > 1 ? 1 : v); }
    double arcR(const Geo& g) const { return g.r + 16 * g.ps; }
    double fT(double t) const { return clamp01(std::log(_fstop->getValueAtTime(t) / kFMin) / std::log(kFMax / kFMin)); }
    double iT(double t) const { return clamp01((_ior->getValueAtTime(t) - kIorMin) / (kIorMax - kIorMin)); }
    double cT(double t) const { return clamp01(_focus->getValueAtTime(t)); }
    double dT(double t) const { return clamp01((_depth->getValueAtTime(t) - kDMin) / (kDMax - kDMin)); }
    double barY(const Geo& g) const { return g.cy - g.r - 34 * g.ps; }

    static OfxPointD onArc(const Geo& g, double r, double deg)
    {
        return { g.cx + r * std::cos(deg * ov::kPi / 180), g.cy + r * std::sin(deg * ov::kPi / 180) };
    }
    // Angle of the pen around the centre, mapped to t along an arc (a0 -> a1), or -1 if off the arc's span.
    static double arcParam(const Geo& g, double px, double py, double a0, double a1)
    {
        double a = std::atan2(py - g.cy, px - g.cx) * 180 / ov::kPi;
        const double mid = 0.5 * (a0 + a1);
        while (a - mid > 180) a -= 360;
        while (a - mid < -180) a += 360;
        return (a - a0) / (a1 - a0);
    }

    Part hit(const PenArgs& a, const Geo& g) const
    {
        const double tol = 7.0 * g.ps, px = a.penPosition.x, py = a.penPosition.y, t = a.time;
        auto isNear = [&](OfxPointD p, double k = 1.0) { return std::hypot(px - p.x, py - p.y) < tol * k; };
        for (int i = 0; i < kNumLights; ++i) {
            if (_lint[i]->getValueAtTime(t) <= 0) continue;
            const OfxPointD l = pointAt(_lpos[i], t);
            if (isNear({ g.x1 + l.x * g.w, g.y1 + l.y * g.h }, 1.8)) return (Part)(kLight0 + i);
        }
        const double R = arcR(g);
        // Knobs first (small targets win over long curves).
        if (isNear(onArc(g, R, kFA0 + (kFA1 - kFA0) * fT(t)), 1.6)) return kFStop;
        if (isNear(onArc(g, R, kIA0 + (kIA1 - kIA0) * iT(t)), 1.6)) return kIOR;
        if (isNear(onArc(g, R, kCA0 + (kCA1 - kCA0) * cT(t)), 1.6)) return kFocus;
        if (isNear({ g.cx - g.r + 2 * g.r * dT(t), barY(g) }, 1.6)) return kDepth;
        if (isNear(onArc(g, g.r, -45), 1.4)) return kSize;
        if (isNear({ g.cx, g.cy }, 1.6)) return kMove;
        const double al = 0.42 * g.r;
        if (py > g.cy - tol && py < g.cy + tol && px > g.cx + 3 * tol && px < g.cx + al + tol) return kMoveX;
        if (px > g.cx - tol && px < g.cx + tol && py > g.cy + 3 * tol && py < g.cy + al + tol) return kMoveY;
        // Arc and bar bodies.
        const double d = std::hypot(px - g.cx, py - g.cy);
        if (std::fabs(d - R) < tol) {
            const double tf = arcParam(g, px, py, kFA0, kFA1), ti = arcParam(g, px, py, kIA0, kIA1),
                         tc = arcParam(g, px, py, kCA0, kCA1);
            if (tf >= 0 && tf <= 1) return kFStop;
            if (ti >= 0 && ti <= 1) return kIOR;
            if (tc >= 0 && tc <= 1) return kFocus;
        }
        if (std::fabs(py - barY(g)) < tol && px > g.cx - g.r - tol && px < g.cx + g.r + tol) return kDepth;
        if (std::fabs(d - g.r) < tol) return kRoll;
        // Tilt (vertical) and turn (horizontal) ellipses.
        auto onEllipse = [&](double ea, double eb) {
            const double q = std::sqrt(((px - g.cx) / ea) * ((px - g.cx) / ea) + ((py - g.cy) / eb) * ((py - g.cy) / eb));
            return std::fabs(q - 1.0) * std::min(ea, eb) < tol;
        };
        if (onEllipse(0.32 * g.r, 0.92 * g.r)) return kTilt;
        if (onEllipse(0.92 * g.r, 0.32 * g.r)) return kTurn;
        if (d < g.r) return kSpin;
        return kNone;
    }

    void colour(const ov::Col& c, Part part, bool shadow) const
    {
        if (shadow) { glColor4f(0, 0, 0, 0.55f); return; }
        const bool hot = _drag == part || (_drag == kNone && _hover == part);
        if (hot) glColor4f(std::min(1.0f, c.r + 0.25f), std::min(1.0f, c.g + 0.25f), std::min(1.0f, c.b + 0.25f), 1.0f);
        else glColor4f(c.r, c.g, c.b, 0.85f);
        glLineWidth(hot ? 3.0f : 1.6f);
    }
    bool hot(Part p) const { return _drag == p || (_drag == kNone && _hover == p); }

    void shapes(const Geo& g, double t, bool sh) const
    {
        const double ps = g.ps, R = arcR(g);
        if (sh) glLineWidth(3.0f);

        // Rotation: tilt / turn ellipses, roll ring.
        colour(ov::kRed, kTilt, sh);
        ov::ellipse(g.cx, g.cy, 0.32 * g.r, 0.92 * g.r);
        colour(ov::kGreen, kTurn, sh);
        ov::ellipse(g.cx, g.cy, 0.92 * g.r, 0.32 * g.r);
        colour(ov::kBlue, kRoll, sh);
        ov::circle(g.cx, g.cy, g.r, 96);
        const double ra = _roll->getValueAtTime(t);
        const OfxPointD r0 = onArc(g, g.r, ra), r1 = onArc(g, g.r - 10 * ps, ra);
        glBegin(GL_LINES); glVertex2d(r0.x, r0.y); glVertex2d(r1.x, r1.y); glEnd();

        // Orientation triad: the glass's own axes projected to the screen.
        if (!sh) {
            const double rx = _tilt->getValueAtTime(t) * ov::kPi / 180, ry = _turn->getValueAtTime(t) * ov::kPi / 180,
                         rz = ra * ov::kPi / 180;
            const double cx = std::cos(rx), sx = std::sin(rx), cy = std::cos(ry), sy = std::sin(ry), cz = std::cos(rz),
                         sz = std::sin(rz);
            const double M[9] = { cz * cy, cz * sy * sx - sz * cx, cz * sy * cx + sz * sx,
                                  sz * cy, sz * sy * sx + cz * cx, sz * sy * cx - cz * sx,
                                  -sy, cy * sx, cy * cx };
            const double L = 0.22 * g.r;
            const ov::Col cols[3] = { ov::kRed, ov::kGreen, ov::kBlue };
            glLineWidth(2.0f);
            for (int k = 0; k < 3; ++k) {
                glColor4f(cols[k].r, cols[k].g, cols[k].b, 0.6f);
                // Screen y is up and z points away from the camera (-z faces us).
                glBegin(GL_LINES);
                glVertex2d(g.cx, g.cy);
                glVertex2d(g.cx + L * M[k], g.cy + L * M[3 + k]);
                glEnd();
            }
        }

        // Move: centre dot and axis arrows.
        const double al = 0.42 * g.r, hs = 6 * ps;
        colour(ov::kRed, kMoveX, sh);
        glBegin(GL_LINES);
        glVertex2d(g.cx + 9 * ps, g.cy); glVertex2d(g.cx + al, g.cy);
        glVertex2d(g.cx + al, g.cy); glVertex2d(g.cx + al - hs, g.cy + hs * 0.6);
        glVertex2d(g.cx + al, g.cy); glVertex2d(g.cx + al - hs, g.cy - hs * 0.6);
        glEnd();
        colour(ov::kGreen, kMoveY, sh);
        glBegin(GL_LINES);
        glVertex2d(g.cx, g.cy + 9 * ps); glVertex2d(g.cx, g.cy + al);
        glVertex2d(g.cx, g.cy + al); glVertex2d(g.cx - hs * 0.6, g.cy + al - hs);
        glVertex2d(g.cx, g.cy + al); glVertex2d(g.cx + hs * 0.6, g.cy + al - hs);
        glEnd();
        colour(ov::kWhite, kMove, sh);
        ov::circle(g.cx, g.cy, 6 * ps, 20);
        if (!sh) ov::disc(g.cx, g.cy, 2.5 * ps, 12);

        // Size square on the ring.
        colour(ov::kWhite, kSize, sh);
        const OfxPointD sq = onArc(g, g.r, -45);
        const double s = 5 * ps;
        glBegin(GL_LINE_LOOP);
        glVertex2d(sq.x - s, sq.y - s); glVertex2d(sq.x + s, sq.y - s); glVertex2d(sq.x + s, sq.y + s); glVertex2d(sq.x - s, sq.y + s);
        glEnd();

        // Value arcs with knobs.
        auto valueArc = [&](const ov::Col& c, Part p, double a0, double a1, double tv) {
            colour(c, p, sh);
            ov::arc(g.cx, g.cy, R, a0, a1);
            const OfxPointD k = onArc(g, R, a0 + (a1 - a0) * tv);
            if (sh) ov::circle(k.x, k.y, 6 * ps, 16); else ov::disc(k.x, k.y, (hot(p) ? 6.5 : 5) * ps);
        };
        valueArc(ov::kPurple, kFStop, kFA0, kFA1, fT(t));
        valueArc(ov::kOrange, kIOR, kIA0, kIA1, iT(t));
        valueArc(ov::kYellow, kFocus, kCA0, kCA1, cT(t));

        // Depth bar with knob.
        colour(ov::kCyan, kDepth, sh);
        const double by = barY(g);
        glBegin(GL_LINES);
        glVertex2d(g.cx - g.r, by); glVertex2d(g.cx + g.r, by);
        glVertex2d(g.cx - g.r, by - 4 * ps); glVertex2d(g.cx - g.r, by + 4 * ps);
        glVertex2d(g.cx + g.r, by - 4 * ps); glVertex2d(g.cx + g.r, by + 4 * ps);
        glEnd();
        const double kx = g.cx - g.r + 2 * g.r * dT(t);
        if (sh) ov::circle(kx, by, 6 * ps, 16); else ov::disc(kx, by, (hot(kDepth) ? 6.5 : 5) * ps);

        // Lights, tinted by colour temperature.
        for (int i = 0; i < kNumLights; ++i) {
            if (_lint[i]->getValueAtTime(t) <= 0) continue;
            const OfxPointD l = pointAt(_lpos[i], t);
            const double lx = g.x1 + l.x * g.w, ly = g.y1 + l.y * g.h;
            float rgb[3];
            fg3d::kelvinToRGB(_lkel[i]->getValueAtTime(t), rgb);
            colour({ rgb[0], rgb[1], rgb[2] }, (Part)(kLight0 + i), sh);
            ov::circle(lx, ly, 6 * ps, 16);
            glBegin(GL_LINES);
            for (int k = 0; k < 8; ++k) {
                const double a = ov::kPi / 4 * k;
                glVertex2d(lx + 9 * ps * std::cos(a), ly + 9 * ps * std::sin(a));
                glVertex2d(lx + 14 * ps * std::cos(a), ly + 14 * ps * std::sin(a));
            }
            glEnd();
        }

        // Readout for the active handle.
        if (!sh && (_drag != kNone || _hover != kNone)) {
            const Part p = _drag != kNone ? _drag : _hover;
            char buf[32] = { 0 };
            ov::Col c = ov::kWhite;
            if (p == kFStop) { std::snprintf(buf, sizeof buf, "f/%.1f", _fstop->getValueAtTime(t)); c = ov::kPurple; }
            else if (p == kIOR) { std::snprintf(buf, sizeof buf, "%.2f", _ior->getValueAtTime(t)); c = ov::kOrange; }
            else if (p == kFocus) { std::snprintf(buf, sizeof buf, "%.0f%%", 100 * _focus->getValueAtTime(t)); c = ov::kYellow; }
            else if (p == kDepth) { std::snprintf(buf, sizeof buf, "%.3f", _depth->getValueAtTime(t)); c = ov::kCyan; }
            else if (p == kSize) { std::snprintf(buf, sizeof buf, "%.2f", _size->getValueAtTime(t)); }
            else if (p == kRoll) { std::snprintf(buf, sizeof buf, "%.0f", ra); c = ov::kBlue; }
            else if (p == kTilt || p == kSpin) { std::snprintf(buf, sizeof buf, "%.0f", _tilt->getValueAtTime(t)); c = ov::kRed; }
            else if (p == kTurn) { std::snprintf(buf, sizeof buf, "%.0f", _turn->getValueAtTime(t)); c = ov::kGreen; }
            if (buf[0]) {
                glLineWidth(1.8f);
                glColor4f(0, 0, 0, 0.7f);
                ov::text(_lastPen.x + 14 * ps + ps, _lastPen.y + 10 * ps - ps, 11 * ps, buf);
                glColor4f(c.r, c.g, c.b, 1.0f);
                ov::text(_lastPen.x + 14 * ps, _lastPen.y + 10 * ps, 11 * ps, buf);
            }
        }
    }

    bool drawImpl(const DrawArgs& args)
    {
        const Geo g = geo(args.time, args.pixelScale.x);
        glPushAttrib(GL_ALL_ATTRIB_BITS);
        glEnable(GL_BLEND);
        glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);
        glEnable(GL_LINE_SMOOTH);
        glMatrixMode(GL_MODELVIEW);
        glPushMatrix();
        glTranslated(g.ps, -g.ps, 0);  // shadow pass for contrast on bright footage
        shapes(g, args.time, true);
        glPopMatrix();
        shapes(g, args.time, false);
        glPopAttrib();
        return true;
    }

    bool motionImpl(const PenArgs& a)
    {
        const Geo g = geo(a.time, a.pixelScale.x);
        _lastPen = a.penPosition;
        if (_drag == kNone) {
            const Part h = hit(a, g);
            if (h != _hover || h != kNone) { _hover = h; _effect->redrawOverlays(); }
            return false;
        }
        const double px = a.penPosition.x, py = a.penPosition.y, t = a.time;
        const double dx = px - _start.x, dy = py - _start.y;
        switch (_drag) {
        case kMove: _center->setValueAtTime(t, _c0.x + dx / g.w, _c0.y + dy / g.h); break;
        case kMoveX: _center->setValueAtTime(t, _c0.x + dx / g.w, _c0.y); break;
        case kMoveY: _center->setValueAtTime(t, _c0.x, _c0.y + dy / g.h); break;
        case kRoll: {
            double v = (std::atan2(py - g.cy, px - g.cx) - _a0) * 180 / ov::kPi + _roll0;
            while (v > 180) v -= 360;
            while (v < -180) v += 360;
            _roll->setValueAtTime(t, v);
            break;
        }
        case kTilt: _tilt->setValueAtTime(t, _tilt0 + dy / g.r * 90); break;
        case kTurn: _turn->setValueAtTime(t, _turn0 - dx / g.r * 90); break;
        case kSpin:
            _turn->setValueAtTime(t, _turn0 - dx / g.r * 90);
            _tilt->setValueAtTime(t, _tilt0 + dy / g.r * 90);
            break;
        case kSize: _size->setValueAtTime(t, std::max(0.01, std::hypot(px - g.cx, py - g.cy) / (0.5 * g.h))); break;
        case kFStop: {
            const double u = clamp01(arcParam(g, px, py, kFA0, kFA1));
            _fstop->setValueAtTime(t, kFMin * std::pow(kFMax / kFMin, u));
            break;
        }
        case kIOR: _ior->setValueAtTime(t, kIorMin + (kIorMax - kIorMin) * clamp01(arcParam(g, px, py, kIA0, kIA1))); break;
        case kFocus: _focus->setValueAtTime(t, clamp01(arcParam(g, px, py, kCA0, kCA1))); break;
        case kDepth: _depth->setValueAtTime(t, kDMin + (kDMax - kDMin) * clamp01((px - (g.cx - g.r)) / (2 * g.r))); break;
        default:
            if (_drag >= kLight0) _lpos[_drag - kLight0]->setValueAtTime(t, (px - g.x1) / g.w, (py - g.y1) / g.h);
        }
        _effect->redrawOverlays();
        return true;
    }

    bool downImpl(const PenArgs& a)
    {
        const Geo g = geo(a.time, a.pixelScale.x);
        const Part h = hit(a, g);
        if (h == kNone) return false;
        _drag = h;
        _start = _lastPen = a.penPosition;
        _c0 = pointAt(_center, a.time);
        _tilt0 = _tilt->getValueAtTime(a.time);
        _turn0 = _turn->getValueAtTime(a.time);
        _roll0 = _roll->getValueAtTime(a.time);
        _a0 = std::atan2(a.penPosition.y - g.cy, a.penPosition.x - g.cx);
        _effect->beginEditBlock("FacetGlass handle");
        _interacting->setValue(true);
        _effect->redrawOverlays();
        return true;
    }

    bool upImpl(const PenArgs&)
    {
        if (_drag == kNone) return false;
        _drag = kNone;
        _interacting->setValue(false);  // back to full quality
        _effect->endEditBlock();
        _effect->redrawOverlays();
        return true;
    }

    Double2DParam* _center;
    DoubleParam *_size, *_tilt, *_turn, *_roll, *_depth, *_fstop, *_ior, *_focus;
    BooleanParam* _interacting;
    Clip* _src;
    Double2DParam* _lpos[kNumLights];
    DoubleParam *_lint[kNumLights], *_lkel[kNumLights];

    Part _drag = kNone, _hover = kNone;
    OfxPointD _start = { 0, 0 }, _c0 = { 0, 0 }, _lastPen = { 0, 0 };
    double _tilt0 = 0, _turn0 = 0, _roll0 = 0, _a0 = 0;
};

class GlassOverlayDescriptor : public DefaultEffectOverlayDescriptor<GlassOverlayDescriptor, GlassOverlay> {};

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
    desc.setOverlayInteractDescriptor(new GlassOverlayDescriptor);
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
    dbl(desc, P_DEPTH, "Depth", "Distance of the glass from the lens; the footage is at 1. Viewer: cyan bar.", gp, 0.15, 0.02, 0.6, 0.01, 0.95);
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
    dbl(desc, P_IOR, "Strength (IOR)", "Index of refraction: how strongly the glass bends light. Viewer: orange arc.", gg, 1.55, 1.0,
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
    dbl(desc, P_FSTOP, "F-Stop", "Lens aperture: lower blurs the out-of-focus glass more. Viewer: purple arc.", go, 4.0, 0.95, 22.0, 0.5, 64.0);
    dbl(desc, P_FOCUS, "Focus", "Focus distance: 0 = on the glass, 1 = on the footage. Viewer: yellow arc.", go, 1.0, 0.0, 1.0, 0.0, 1.0);
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

    {
        BooleanParamDescriptor* b = desc.defineBooleanParam(P_INTERACT);
        b->setDefault(false);
        b->setIsSecret(true);
        b->setAnimates(false);
        b->setLabels("Interacting", "Interacting", "Interacting");
        b->setHint("Set while a viewer handle is dragged: renders at Draft quality.");
    }

    g_page = nullptr;
}

void OFX::Plugin::getPluginIDs(PluginFactoryArray& ids)
{
    static FacetGlassFactory factory;
    ids.push_back(&factory);
}
