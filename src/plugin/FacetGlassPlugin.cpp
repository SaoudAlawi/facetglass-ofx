// FacetGlass - OpenFX plugin (CPU render).
// SPDX-License-Identifier: MIT
//
// Ray-traced glass in front of the lens. All shading math lives in
// src/core/fg_core.h; this file only describes the plugin, reads parameters,
// converts them exactly like the Fuse reference (reference/fuse/FacetGlass.fuse,
// Process()) and runs fg_shade over the render window.

#include <algorithm>
#include <cfloat>
#include <climits>
#include <cmath>
#include <cstring>
#include <memory>

#include "ofxsImageEffect.h"
#include "ofxsMultiThread.h"
#include "ofxsProcessing.h"

#include "fg_backend_cpu.h"

#define kPluginName "FacetGlass"
#define kPluginGrouping "FacetGlass"
#define kPluginDescription                                                                                   \
    "Ray-traced glass held in front of the lens: prisms, ghost repeats, split and round diopters and "      \
    "crystal facets, with lens-aperture depth of field, spectral dispersion and Fresnel reflections.\n"     \
    "Free and open source (MIT)."
#define kPluginIdentifier "com.facetglass.FacetGlass"
#define kPluginVersionMajor 1
#define kPluginVersionMinor 0

#define kSupportsTiles true
#define kSupportsMultiResolution true
#define kSupportsMultipleClipPARs false

#define kClipRefract "RefractSource"

// Parameter names (script names). These are stored in saved projects: never rename.
#define kParamShape "shape"
#define kParamCenter "center"
#define kParamAngle "angle"
#define kParamSize "size"
#define kParamFacets "facets"
#define kParamSymmetric "bothSides"
#define kParamInner "clearCentre"
#define kParamPrism "prismAngle"
#define kParamPower "diopterPower"
#define kParamIOR "ior"
#define kParamDispersion "chromaticAberration"
#define kParamSeed "seed"
#define kParamFStop "fStop"
#define kParamFocus "focus"
#define kParamFocusD "diopterFocus"
#define kParamGlassDist "glassDistance"
#define kParamEdge "outsideFrame"
#define kParamQuality "quality"
#define kParamSamples "samples"
#define kParamBoost "highlightBoost"
#define kParamThreshold "highlightThreshold"
#define kParamRefl "reflection"
#define kParamLight "lightPosition"
#define kParamLightSize "lightSize"
#define kParamLightInt "lightIntensity"
#define kParamWobble "wobble"
#define kParamWobbleSpeed "wobbleSpeed"
#define kParamMix "mix"

enum { kQualityDraft = 0, kQualityGood, kQualityFinal, kQualityCustom };
static const int kQualitySamples[3] = { 8, 24, 64 };

using namespace OFX;

////////////////////////////////////////////////////////////////////////////////
// Host-side conversions (same as the Fuse's Lua Process()).

namespace {

struct Wobble { double ox, oy, oa; };

// Hand-held wobble from time in frames, 24 fps base. Matches wobble() in the Fuse.
Wobble wobble(double t, int seed, double amount, double speed)
{
    Wobble w = { 0.0, 0.0, 0.0 };
    if (amount <= 0.0) return w;
    const double s = seed * 1.7319;
    const double tt = t * speed / 24.0;
    w.ox = amount * 0.03 * (std::sin(1.31 * tt + s) + 0.5 * std::sin(2.97 * tt + 2.1 * s));
    w.oy = amount * 0.03 * (std::sin(1.07 * tt + 3.3 * s) + 0.5 * std::sin(2.41 * tt + 0.7 * s));
    w.oa = amount * 6.0 * (std::sin(0.83 * tt + 5.1 * s) + 0.4 * std::sin(2.13 * tt + 1.3 * s));
    return w;
}

double focusInv(double f, double zg)
{
    f = std::min(std::max(f, 0.0), 1.0);
    return 1.0 + (1.0 / zg - 1.0) * (1.0 - f) * (1.0 - f);
}

double deg2rad(double d) { return d * 3.14159265358979323846 / 180.0; }

int choiceAt(ChoiceParam* c, double t) { int v = 0; c->getValueAtTime(t, v); return v; }
OfxPointD pointAt(Double2DParam* d, double t) { OfxPointD v; d->getValueAtTime(t, v.x, v.y); return v; }

// Canonical coordinates -> pixel coordinates at the given render scale (enclosing).
OfxRectI toPixelEnclosing(const OfxRectD& r, const OfxPointD& rs, double par)
{
    OfxRectI p;
    p.x1 = (int)std::floor(r.x1 * rs.x / par);
    p.y1 = (int)std::floor(r.y1 * rs.y);
    p.x2 = (int)std::ceil(r.x2 * rs.x / par);
    p.y2 = (int)std::ceil(r.y2 * rs.y);
    return p;
}

} // namespace

////////////////////////////////////////////////////////////////////////////////
// Processor: runs fg_shade on each pixel of the render window.

class FacetGlassProcessor : public ImageProcessor
{
public:
    explicit FacetGlassProcessor(ImageEffect& effect)
        : ImageProcessor(effect), _src(nullptr), _gls(nullptr)
    {
        std::memset(&_params, 0, sizeof _params);
    }

    void setup(const FGParams& p, const float* src, const float* gls)
    {
        _params = p;
        _src = src;
        _gls = gls;
    }

    void multiThreadProcessImages(OfxRectI win) override
    {
        const FGParams p = _params; // private copy, same as the GPU kernels will do
        for (int y = win.y1; y < win.y2; ++y) {
            if (_effect.abort()) break;
            float* d = static_cast<float*>(_dstImg->getPixelAddress(win.x1, y));
            for (int x = win.x1; x < win.x2; ++x, d += 4) {
                const fg4 c = fg_shade(&p, _src, _gls, x, y);
                d[0] = c.x; d[1] = c.y; d[2] = c.z; d[3] = c.w;
            }
        }
    }

private:
    FGParams _params;
    const float* _src;
    const float* _gls;
};

////////////////////////////////////////////////////////////////////////////////
// The effect instance.

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
    void updateSamplesVisibility();
    bool refractConnected() const { return _refractClip && _refractClip->isConnected(); }

    Clip* _dstClip;
    Clip* _srcClip;
    Clip* _refractClip; // only in the General context

    ChoiceParam* _shape;
    Double2DParam* _center;
    DoubleParam* _angle;
    DoubleParam* _size;
    IntParam* _facets;
    BooleanParam* _symmetric;
    DoubleParam* _inner;
    DoubleParam* _prism;
    DoubleParam* _power;
    DoubleParam* _ior;
    DoubleParam* _dispersion;
    IntParam* _seed;
    DoubleParam* _fstop;
    DoubleParam* _focus;
    DoubleParam* _focusD;
    DoubleParam* _glassDist;
    ChoiceParam* _edge;
    ChoiceParam* _quality;
    IntParam* _samples;
    DoubleParam* _boost;
    DoubleParam* _threshold;
    DoubleParam* _refl;
    Double2DParam* _light;
    DoubleParam* _lightSize;
    DoubleParam* _lightInt;
    DoubleParam* _wobble;
    DoubleParam* _wobbleSpeed;
    DoubleParam* _mix;
};

FacetGlassPlugin::FacetGlassPlugin(OfxImageEffectHandle handle)
    : ImageEffect(handle), _refractClip(nullptr)
{
    _dstClip = fetchClip(kOfxImageEffectOutputClipName);
    _srcClip = fetchClip(kOfxImageEffectSimpleSourceClipName);
    if (getContext() == eContextGeneral) _refractClip = fetchClip(kClipRefract);

    _shape = fetchChoiceParam(kParamShape);
    _center = fetchDouble2DParam(kParamCenter);
    _angle = fetchDoubleParam(kParamAngle);
    _size = fetchDoubleParam(kParamSize);
    _facets = fetchIntParam(kParamFacets);
    _symmetric = fetchBooleanParam(kParamSymmetric);
    _inner = fetchDoubleParam(kParamInner);
    _prism = fetchDoubleParam(kParamPrism);
    _power = fetchDoubleParam(kParamPower);
    _ior = fetchDoubleParam(kParamIOR);
    _dispersion = fetchDoubleParam(kParamDispersion);
    _seed = fetchIntParam(kParamSeed);
    _fstop = fetchDoubleParam(kParamFStop);
    _focus = fetchDoubleParam(kParamFocus);
    _focusD = fetchDoubleParam(kParamFocusD);
    _glassDist = fetchDoubleParam(kParamGlassDist);
    _edge = fetchChoiceParam(kParamEdge);
    _quality = fetchChoiceParam(kParamQuality);
    _samples = fetchIntParam(kParamSamples);
    _boost = fetchDoubleParam(kParamBoost);
    _threshold = fetchDoubleParam(kParamThreshold);
    _refl = fetchDoubleParam(kParamRefl);
    _light = fetchDouble2DParam(kParamLight);
    _lightSize = fetchDoubleParam(kParamLightSize);
    _lightInt = fetchDoubleParam(kParamLightInt);
    _wobble = fetchDoubleParam(kParamWobble);
    _wobbleSpeed = fetchDoubleParam(kParamWobbleSpeed);
    _mix = fetchDoubleParam(kParamMix);

    updateSamplesVisibility();
}

void FacetGlassPlugin::updateSamplesVisibility()
{
    const bool custom = choiceAt(_quality, 0.0) == kQualityCustom;
    _samples->setIsSecret(!custom);
    _samples->setEnabled(custom);
}

void FacetGlassPlugin::changedParam(const InstanceChangedArgs& /*args*/, const std::string& paramName)
{
    if (paramName == kParamQuality) updateSamplesVisibility();
}

bool FacetGlassPlugin::isIdentity(const IsIdentityArguments& args, Clip*& identityClip, double& identityTime)
{
    if (_mix->getValueAtTime(args.time) == 0.0) {
        identityClip = _srcClip;
        identityTime = args.time;
        return true;
    }
    return false;
}

// The output covers exactly the source frame (an optional Refract Source must not enlarge it).
bool FacetGlassPlugin::getRegionOfDefinition(const RegionOfDefinitionArguments& args, OfxRectD& rod)
{
    rod = _srcClip->getRegionOfDefinition(args.time);
    return true;
}

// Refracted rays can land anywhere in the frame, so every render needs the whole source.
void FacetGlassPlugin::getRegionsOfInterest(const RegionsOfInterestArguments& args, RegionOfInterestSetter& rois)
{
    rois.setRegionOfInterest(*_srcClip, _srcClip->getRegionOfDefinition(args.time));
    if (refractConnected())
        rois.setRegionOfInterest(*_refractClip, _refractClip->getRegionOfDefinition(args.time));
}

void FacetGlassPlugin::render(const RenderArguments& args)
{
    if (_dstClip->getPixelDepth() != eBitDepthFloat || _dstClip->getPixelComponents() != ePixelComponentRGBA)
        throwSuiteStatusException(kOfxStatErrUnsupported);

    const double t = args.time;
    std::unique_ptr<Image> dst(_dstClip->fetchImage(t));
    std::unique_ptr<Image> src(_srcClip->fetchImage(t));
    if (!dst || !src) throwSuiteStatusException(kOfxStatFailed);
    if (src->getPixelDepth() != eBitDepthFloat || src->getPixelComponents() != ePixelComponentRGBA)
        throwSuiteStatusException(kOfxStatErrImageFormat);

    std::unique_ptr<Image> gls;
    if (refractConnected()) {
        gls.reset(_refractClip->fetchImage(t));
        if (gls && (gls->getPixelDepth() != eBitDepthFloat || gls->getPixelComponents() != ePixelComponentRGBA))
            throwSuiteStatusException(kOfxStatErrImageFormat);
    }
    const Image* glsImg = gls ? gls.get() : src.get();

    FGParams p;
    std::memset(&p, 0, sizeof p);

    // Frame: source RoD in pixels at the current render scale.
    const OfxRectI frame = toPixelEnclosing(_srcClip->getRegionOfDefinition(t), args.renderScale,
                                            _srcClip->getPixelAspectRatio());
    p.frameX = (float)frame.x1;
    p.frameY = (float)frame.y1;
    p.frameW = (float)(frame.x2 - frame.x1);
    p.frameH = (float)(frame.y2 - frame.y1);
    if (p.frameW < 1.0f || p.frameH < 1.0f) throwSuiteStatusException(kOfxStatFailed);

    // Image layouts. rowBytes may be negative; the base is pixel (x1, y1).
    const OfxRectI sb = src->getBounds();
    const OfxRectI gb = glsImg->getBounds();
    if (sb.x2 <= sb.x1 || sb.y2 <= sb.y1 || gb.x2 <= gb.x1 || gb.y2 <= gb.y1)
        throwSuiteStatusException(kOfxStatFailed);
    p.srcX1 = sb.x1; p.srcY1 = sb.y1; p.srcW = sb.x2 - sb.x1; p.srcH = sb.y2 - sb.y1;
    p.srcStride = src->getRowBytes() / 4;
    p.glsX1 = gb.x1; p.glsY1 = gb.y1; p.glsW = gb.x2 - gb.x1; p.glsH = gb.y2 - gb.y1;
    p.glsStride = glsImg->getRowBytes() / 4;
    p.useSource = gls ? 1 : 0;
    const float* srcBase = static_cast<const float*>(src->getPixelAddress(sb.x1, sb.y1));
    const float* glsBase = static_cast<const float*>(glsImg->getPixelAddress(gb.x1, gb.y1));

    // Parameters, converted like the Fuse.
    const double zg = std::max(_glassDist->getValueAtTime(t), 0.005);
    const double fstop = std::max(_fstop->getValueAtTime(t), 0.5);
    const int quality = choiceAt(_quality, t);
    const int samples = (quality >= 0 && quality < 3) ? kQualitySamples[quality] : _samples->getValueAtTime(t);
    const int seed = _seed->getValueAtTime(t);
    const Wobble w = wobble(t, seed, _wobble->getValueAtTime(t), _wobbleSpeed->getValueAtTime(t));
    const OfxPointD center = pointAt(_center, t);
    const OfxPointD light = pointAt(_light, t);

    p.shape = choiceAt(_shape, t);
    p.facets = _facets->getValueAtTime(t);
    p.symmetric = _symmetric->getValueAtTime(t) ? 1 : 0;
    p.samples = std::max(samples, 1);
    p.edgeMode = choiceAt(_edge, t);
    p.seed = seed;
    p.centerX = (float)(center.x + w.ox);
    p.centerY = (float)(center.y + w.oy);
    p.angle = (float)deg2rad(_angle->getValueAtTime(t) + w.oa);
    p.size = (float)std::max(_size->getValueAtTime(t), 0.01);
    p.inner = (float)_inner->getValueAtTime(t);
    p.prism = (float)deg2rad(_prism->getValueAtTime(t));
    p.power = (float)_power->getValueAtTime(t);
    p.ior = (float)_ior->getValueAtTime(t);
    p.dispersion = (float)_dispersion->getValueAtTime(t);
    p.aperture = (float)(zg * 0.3 / fstop);
    p.focusInv = (float)focusInv(_focus->getValueAtTime(t), zg);
    p.focusInvD = (float)focusInv(_focusD->getValueAtTime(t), zg);
    p.glassDist = (float)zg;
    p.boost = (float)_boost->getValueAtTime(t);
    p.threshold = (float)_threshold->getValueAtTime(t);
    p.refl = (float)_refl->getValueAtTime(t);
    p.lightX = (float)light.x;
    p.lightY = (float)light.y;
    p.lightSize = (float)_lightSize->getValueAtTime(t);
    p.lightInt = (float)_lightInt->getValueAtTime(t);
    p.mix = (float)_mix->getValueAtTime(t);

    FacetGlassProcessor proc(*this);
    proc.setDstImg(dst.get());
    proc.setGPURenderArgs(args);
    proc.setRenderWindow(args.renderWindow);
    proc.setup(p, srcBase, glsBase);
    proc.process();
}

////////////////////////////////////////////////////////////////////////////////
// Factory.

class FacetGlassFactory : public PluginFactoryHelper<FacetGlassFactory>
{
public:
    FacetGlassFactory() : PluginFactoryHelper<FacetGlassFactory>(kPluginIdentifier, kPluginVersionMajor, kPluginVersionMinor) {}
    void describe(ImageEffectDescriptor& desc) override;
    void describeInContext(ImageEffectDescriptor& desc, ContextEnum context) override;
    ImageEffect* createInstance(OfxImageEffectHandle handle, ContextEnum context) override;
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
    desc.setSupportsMultiResolution(kSupportsMultiResolution);
    desc.setSupportsTiles(kSupportsTiles);
    desc.setTemporalClipAccess(false);
    desc.setRenderTwiceAlways(false);
    desc.setSupportsMultipleClipPARs(kSupportsMultipleClipPARs);
    desc.setRenderThreadSafety(eRenderFullySafe);
    // GPU backends (CUDA/OpenCL/Metal) are not implemented yet: CPU only for now.
}

namespace {

PageParamDescriptor* g_page = nullptr;

GroupParamDescriptor* defineGroup(ImageEffectDescriptor& desc, const char* name, const char* label)
{
    GroupParamDescriptor* g = desc.defineGroupParam(name);
    g->setLabels(label, label, label);
    g->setOpen(true);
    if (g_page) g_page->addChild(*g);
    return g;
}

template <class T>
T* finish(T* param, const char* label, const char* hint, GroupParamDescriptor* group)
{
    param->setLabels(label, label, label);
    param->setHint(hint);
    param->setParent(*group);
    param->setAnimates(true);
    if (g_page) g_page->addChild(*param);
    return param;
}

DoubleParamDescriptor* defineDouble(ImageEffectDescriptor& desc, const char* name, const char* label, const char* hint,
                                    GroupParamDescriptor* group, double def, double dispMin, double dispMax,
                                    double min = -DBL_MAX, double max = DBL_MAX)
{
    DoubleParamDescriptor* p = desc.defineDoubleParam(name);
    p->setDefault(def);
    p->setRange(min, max);
    p->setDisplayRange(dispMin, dispMax);
    p->setIncrement((dispMax - dispMin) / 100.0);
    p->setDigits(3);
    return finish(p, label, hint, group);
}

IntParamDescriptor* defineInt(ImageEffectDescriptor& desc, const char* name, const char* label, const char* hint,
                              GroupParamDescriptor* group, int def, int dispMin, int dispMax,
                              int min = INT_MIN, int max = INT_MAX)
{
    IntParamDescriptor* p = desc.defineIntParam(name);
    p->setDefault(def);
    p->setRange(min, max);
    p->setDisplayRange(dispMin, dispMax);
    return finish(p, label, hint, group);
}

// Normalised 0..1 point (y up), stored as plain values so it is resolution-independent.
Double2DParamDescriptor* definePoint(ImageEffectDescriptor& desc, const char* name, const char* label, const char* hint,
                                     GroupParamDescriptor* group, double dx, double dy)
{
    Double2DParamDescriptor* p = desc.defineDouble2DParam(name);
    p->setDefault(dx, dy);
    p->setDisplayRange(0.0, 0.0, 1.0, 1.0);
    p->setIncrement(0.001);
    p->setDigits(3);
    p->setDimensionLabels("x", "y");
    return finish(p, label, hint, group);
}

ChoiceParamDescriptor* defineChoice(ImageEffectDescriptor& desc, const char* name, const char* label, const char* hint,
                                    GroupParamDescriptor* group, std::initializer_list<const char*> options, int def)
{
    ChoiceParamDescriptor* p = desc.defineChoiceParam(name);
    for (const char* o : options) p->appendOption(o);
    p->setDefault(def);
    return finish(p, label, hint, group);
}

} // namespace

void FacetGlassFactory::describeInContext(ImageEffectDescriptor& desc, ContextEnum context)
{
    // Clips
    ClipDescriptor* src = desc.defineClip(kOfxImageEffectSimpleSourceClipName);
    src->addSupportedComponent(ePixelComponentRGBA);
    src->setTemporalClipAccess(false);
    src->setSupportsTiles(kSupportsTiles);
    src->setIsMask(false);

    if (context == eContextGeneral) {
        ClipDescriptor* gls = desc.defineClip(kClipRefract);
        gls->setLabels("Refract Source", "Refract", "Refract Source");
        gls->addSupportedComponent(ePixelComponentRGBA);
        gls->setTemporalClipAccess(false);
        gls->setSupportsTiles(kSupportsTiles);
        gls->setIsMask(false);
        gls->setOptional(true);
    }

    ClipDescriptor* dst = desc.defineClip(kOfxImageEffectOutputClipName);
    dst->addSupportedComponent(ePixelComponentRGBA);
    dst->setSupportsTiles(kSupportsTiles);

    g_page = desc.definePageParam("Controls");

    // ---------------------------------------------------------------- Glass
    GroupParamDescriptor* glass = defineGroup(desc, "glassGroup", "Glass");
    defineChoice(desc, kParamShape, "Shape", "The glass element in front of the lens.", glass,
                 { "Linear Prism", "Radial Prism", "Split Diopter", "Round Diopter", "Crystal" }, FG_SHAPE_LINEAR);
    definePoint(desc, kParamCenter, "Center", "Centre of the glass, normalised to the frame (0..1, y up).", glass, 0.5, 0.5);
    DoubleParamDescriptor* angle = defineDouble(desc, kParamAngle, "Angle", "Rotation of the glass, in degrees.",
                                                glass, 0.0, -180.0, 180.0);
    angle->setDoubleType(eDoubleTypeAngle);
    defineDouble(desc, kParamSize, "Size", "Size of the glass element (fraction of frame height).", glass,
                 0.35, 0.05, 1.0, 0.01);
    defineInt(desc, kParamFacets, "Facets", "Number of facets (prism steps, radial segments or crystal cells).",
              glass, 1, 1, 12, 1, 64);
    BooleanParamDescriptor* sym = desc.defineBooleanParam(kParamSymmetric);
    sym->setDefault(false);
    finish(sym, "Both Sides", "Linear Prism: add facets on both sides of the centre line.", glass);
    defineDouble(desc, kParamInner, "Clear Centre", "Radial Prism / Crystal: radius of the clear centre.", glass,
                 0.35, 0.0, 2.0, 0.0);
    defineDouble(desc, kParamPrism, "Prism Angle", "Wedge angle of each prism facet, in degrees.", glass,
                 8.0, 0.0, 30.0);
    defineDouble(desc, kParamPower, "Diopter Power", "Diopter strength; positive is convex (magnifies).", glass,
                 2.0, -20.0, 20.0);
    defineDouble(desc, kParamIOR, "Strength (IOR)", "Index of refraction of the glass.", glass,
                 1.5, 1.0, 2.0, 1.0, 3.0);
    defineDouble(desc, kParamDispersion, "Chromatic Aberration", "Spectral dispersion of the glass.", glass,
                 0.3, 0.0, 1.0, 0.0);
    defineInt(desc, kParamSeed, "Seed", "Random seed for sampling noise and hand-held wobble.", glass,
              1, 0, 100, 0);

    // ---------------------------------------------------------------- Optics
    GroupParamDescriptor* optics = defineGroup(desc, "opticsGroup", "Optics");
    defineDouble(desc, kParamFStop, "F-Stop", "Lens aperture; lower values blur the out-of-focus glass more.",
                 optics, 4.0, 0.95, 22.0, 0.5);
    defineDouble(desc, kParamFocus, "Focus", "Focus distance: 0 = on the glass, 1 = on the footage.", optics,
                 1.0, 0.0, 1.0, 0.0, 1.0);
    defineDouble(desc, kParamFocusD, "Diopter Focus",
                 "Focus for rays through a diopter (0 = glass, 1 = footage). Lets each half hold its own focus.",
                 optics, 1.0, 0.0, 1.0, 0.0, 1.0);
    defineDouble(desc, kParamGlassDist, "Glass Distance", "Distance of the glass from the lens (footage is at 1).",
                 optics, 0.05, 0.01, 0.3, 0.005, 0.9);
    defineChoice(desc, kParamEdge, "Outside Frame", "What refracted rays see beyond the frame edge.", optics,
                 { "Clamp", "Mirror", "Repeat", "Black" }, FG_EDGE_MIRROR);
    ChoiceParamDescriptor* quality =
        defineChoice(desc, kParamQuality, "Quality", "Rays per pixel.", optics,
                     { "Draft (8)", "Good (24)", "Final (64)", "Custom" }, kQualityGood);
    quality->setAnimates(false);
    IntParamDescriptor* samples =
        defineInt(desc, kParamSamples, "Samples", "Rays per pixel when Quality is Custom.", optics, 96, 1, 256, 1, 1024);
    samples->setAnimates(false);
    samples->setIsSecret(true);

    // ---------------------------------------------------------------- Look
    GroupParamDescriptor* look = defineGroup(desc, "lookGroup", "Look");
    defineDouble(desc, kParamBoost, "Highlight Boost", "Brightens highlights seen through the glass.", look,
                 0.5, 0.0, 4.0, 0.0);
    defineDouble(desc, kParamThreshold, "Highlight Threshold", "Luminance above which highlights are boosted.", look,
                 0.8, 0.0, 2.0);
    defineDouble(desc, kParamRefl, "Reflection", "Fresnel reflection off the glass faces.", look,
                 0.0, 0.0, 1.0, 0.0);
    definePoint(desc, kParamLight, "Light Position", "Position of the light that glints off the glass (0..1, y up).",
                look, 0.3, 0.75);
    defineDouble(desc, kParamLightSize, "Light Size", "Size of the glint light.", look, 0.15, 0.01, 1.0, 0.001);
    defineDouble(desc, kParamLightInt, "Light Intensity", "Brightness of the glint.", look, 0.0, 0.0, 10.0, 0.0);
    defineDouble(desc, kParamWobble, "Hand-held Wobble", "Amount of hand-held drift of the glass over time.", look,
                 0.0, 0.0, 1.0, 0.0);
    defineDouble(desc, kParamWobbleSpeed, "Wobble Speed", "Speed of the hand-held drift.", look, 1.0, 0.0, 5.0, 0.0);
    defineDouble(desc, kParamMix, "Mix", "Blend between the original (0) and the glass (1).", look, 1.0, 0.0, 1.0);

    g_page = nullptr;
}

ImageEffect* FacetGlassFactory::createInstance(OfxImageEffectHandle handle, ContextEnum /*context*/)
{
    return new FacetGlassPlugin(handle);
}

void OFX::Plugin::getPluginIDs(PluginFactoryArray& ids)
{
    static FacetGlassFactory factory;
    ids.push_back(&factory);
}
