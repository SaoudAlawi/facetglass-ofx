// FacetGlass test host: a small command-line OpenFX host built on HostSupport.
// SPDX-License-Identifier: MIT
//
// Loads the FacetGlass bundle from one directory only (never the system OFX
// folders), sets parameters, feeds float RGBA images from raw files and writes
// the rendered frame. Used by tests/compare_ofx.py to check the plugin against
// the golden renders without Resolve.
//
// Usage:
//   fg_ofxhost --plugins <dir containing FacetGlass.ofx.bundle>
//              --in <rgba.raw> <W> <H>  --out <rgba.raw>
//              [--refract <rgba.raw> <W> <H>]   connect Refract Source (General context)
//              [--context filter|general]       default: general if --refract, else filter
//              [--scale <s>]                     render scale; images are given at that scale
//              [--tiles <n>]                     render in an n x n grid of render windows
//              [--time <t>]                      frame number (default 0)
//              [--negstride]                     store images top-down (negative rowBytes)
//              [--minimal-host]                  no message suite V2, like DaVinci Resolve
//              [--repeat <n>]                    render the frame n times in this instance (playback cache)
//              [--look <n>]                      pick Look preset n as a user would (instance changed)
//              [--interactive]                   mark renders as interactive (viewer), not final
//              [--metal]                         macOS: images are MTLBuffers, render gets a Metal queue
//              [name=value | name=x,y ...]       parameter values by script name
//              [name:=text]                      text parameter (e.g. a file path)
//
// Raw images are float32 RGBA, row 0 = bottom row (OFX y-up). The output has the
// same layout. Diagnostics go to stdout as "key: value" lines.

#include <algorithm>
#include <atomic>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <map>
#include <memory>
#include <mutex>
#include <sstream>
#include <string>
#include <thread>
#include <vector>

#include "ofxCore.h"
#include "ofxImageEffect.h"
#include "ofxPixels.h"
#include "ofxMessage.h"

// HostSupport headers depend on each other: keep this order (same as hostDemo).
#include "ofxhBinary.h"
#include "ofxhPropertySuite.h"
#include "ofxGPURender.h"

#ifdef __APPLE__
extern "C" void* fgmQueue();
extern "C" void* fgmWrap(const void* data, size_t bytes);
extern "C" void fgmReadBack(void* buf, void* data, size_t bytes);
extern "C" void fgmRelease(void* buf);
#endif
#include "ofxhClip.h"
#include "ofxhParam.h"
#include "ofxhMemory.h"
#include "ofxhImageEffect.h"
#include "ofxhPluginAPICache.h"
#include "ofxhPluginCache.h"
#include "ofxhHost.h"
#include "ofxhImageEffectAPI.h"

namespace OH = OFX::Host;

static const char* kPluginId = "com.facetglass.FacetGlass";

static void fail(const std::string& msg)
{
    std::fprintf(stderr, "fg_ofxhost: %s\n", msg.c_str());
    std::exit(2);
}

static void check(OfxStatus st, const char* what)
{
    if (st != kOfxStatOK && st != kOfxStatReplyDefault) {
        std::ostringstream ss;
        ss << what << " failed with status " << st;
        fail(ss.str());
    }
}

////////////////////////////////////////////////////////////////////////////////
// Pixel buffers

struct Buffer {
    int w = 0, h = 0;
    bool negStride = false;
    std::vector<float> mem;
    void* mtl = nullptr;  // --metal: id<MTLBuffer> holding a copy of mem

    void alloc(int W, int H, bool neg)
    {
        w = W; h = H; negStride = neg;
        mem.assign((size_t)W * H * 4, 0.0f);
    }
    // Pointer to pixel (0,0) (bottom-left) and row stride in bytes.
    float* base() { return negStride ? &mem[(size_t)(h - 1) * w * 4] : mem.data(); }
    int rowBytes() const { return (negStride ? -1 : 1) * w * 4 * (int)sizeof(float); }
    float* row(int y) { return base() + (ptrdiff_t)y * (rowBytes() / (int)sizeof(float)); }

    void load(const std::string& path, int W, int H, bool neg)
    {
        alloc(W, H, neg);
        std::ifstream f(path, std::ios::binary);
        if (!f) fail("cannot open " + path);
        std::vector<float> tmp((size_t)W * H * 4);
        f.read(reinterpret_cast<char*>(tmp.data()), tmp.size() * sizeof(float));
        if (!f) fail("short read " + path);
        for (int y = 0; y < H; ++y) std::memcpy(row(y), &tmp[(size_t)y * W * 4], (size_t)W * 16);
    }
    void save(const std::string& path)
    {
        std::ofstream f(path, std::ios::binary);
        if (!f) fail("cannot write " + path);
        for (int y = 0; y < h; ++y) f.write(reinterpret_cast<const char*>(row(y)), (size_t)w * 16);
    }
};

struct Options {
    std::string pluginDir, out;
    std::string inPath, refPath;
    int inW = 0, inH = 0, refW = 0, refH = 0;
    std::string context;
    double scale = 1.0;
    int tiles = 1;
    double time = 0.0;
    bool negStride = false;
    bool minimalHost = false;  // no message suite V2 (like DaVinci Resolve)
    bool metal = false;        // images as MTLBuffers + Metal render (macOS)
    bool interactive = false;  // renderAction interactive flag
    int look = -1;             // --look: Look menu index chosen by the "user"
    int repeat = 1;            // --repeat: renders of the same frame
    std::vector<std::pair<std::string, std::vector<double>>> params;
    std::vector<std::pair<std::string, std::string>> strParams;  // name:=text
};

static Options g_opt;
static Buffer g_src, g_ref, g_out;

////////////////////////////////////////////////////////////////////////////////
// Images and clips

class TestImage : public OH::ImageEffect::Image {
public:
    TestImage(OH::ImageEffect::ClipInstance& clip, Buffer& b, double scale)
        : OH::ImageEffect::Image(clip, scale, scale, b.mtl ? b.mtl : b.base(), OfxRectI{ 0, 0, b.w, b.h },
                                 OfxRectI{ 0, 0, b.w, b.h }, b.rowBytes(), kOfxImageFieldNone, "fgtest")
    {
    }
};

class TestEffect;

class TestClip : public OH::ImageEffect::ClipInstance {
public:
    TestClip(OH::ImageEffect::Instance* effect, OH::ImageEffect::ClipDescriptor* desc)
        : OH::ImageEffect::ClipInstance(effect, *desc), _name(desc->getName()) {}

    Buffer* buffer() const
    {
        if (_name == kOfxImageEffectOutputClipName) return &g_out;
        if (_name == kOfxImageEffectSimpleSourceClipName) return &g_src;
        if (_name == "RefractSource") return g_ref.w > 0 ? &g_ref : nullptr;
        return nullptr;
    }

    const std::string& getUnmappedBitDepth() const override
    {
        static const std::string f(kOfxBitDepthFloat), n(kOfxBitDepthNone);
        return buffer() ? f : n;
    }
    const std::string& getUnmappedComponents() const override
    {
        static const std::string c(kOfxImageComponentRGBA), n(kOfxImageComponentNone);
        return buffer() ? c : n;
    }
    const std::string& getPremult() const override
    {
        static const std::string v(kOfxImagePreMultiplied);
        return v;
    }
    double getAspectRatio() const override { return 1.0; }
    double getFrameRate() const override { return 24.0; }
    void getFrameRange(double& a, double& b) const override { a = 0; b = 1000; }
    const std::string& getFieldOrder() const override
    {
        static const std::string v(kOfxImageFieldNone);
        return v;
    }
    bool getConnected() const override { return buffer() != nullptr; }
    double getUnmappedFrameRate() const override { return 24.0; }
    void getUnmappedFrameRange(double& a, double& b) const override { a = 0; b = 1000; }
    bool getContinuousSamples() const override { return false; }

    OH::ImageEffect::Image* getImage(OfxTime, const OfxRectD*) override
    {
        Buffer* b = buffer();
        if (!b) return nullptr;
        return new TestImage(*this, *b, g_opt.scale); // ref count 1, released by the plugin
    }

#ifdef OFX_SUPPORTS_OPENGLRENDER
    OH::ImageEffect::Texture* loadTexture(OfxTime, const char*, const OfxRectD*) override { return nullptr; }
#endif

    // Canonical RoD: the buffer is at render scale, canonical is full scale.
    OfxRectD getRegionOfDefinition(OfxTime) const override
    {
        Buffer* b = buffer();
        if (b == &g_out || !b) b = &g_src;
        return OfxRectD{ 0.0, 0.0, b->w / g_opt.scale, b->h / g_opt.scale };
    }

private:
    std::string _name;
};

////////////////////////////////////////////////////////////////////////////////
// Parameters: store a single (non-animated) value, starting from the default.

template <class ParamBase>
class ValueParam : public ParamBase {
public:
    ValueParam(OH::Param::Descriptor& d, OH::Param::SetInstance* set, int dims, bool isDouble)
        : ParamBase(d, set), _v(dims, 0.0)
    {
        for (int i = 0; i < dims; ++i)
            _v[i] = isDouble ? d.getProperties().getDoubleProperty(kOfxParamPropDefault, i)
                             : (double)d.getProperties().getIntProperty(kOfxParamPropDefault, i);
    }
    std::vector<double> _v;
};

class IntP : public ValueParam<OH::Param::IntegerInstance> {
public:
    IntP(OH::Param::Descriptor& d, OH::Param::SetInstance* s) : ValueParam(d, s, 1, false) {}
    OfxStatus get(int& v) override { v = (int)_v[0]; return kOfxStatOK; }
    OfxStatus get(OfxTime, int& v) override { return get(v); }
    OfxStatus set(int v) override { _v[0] = v; return kOfxStatOK; }
    OfxStatus set(OfxTime, int v) override { return set(v); }
};

class ChoiceP : public ValueParam<OH::Param::ChoiceInstance> {
public:
    ChoiceP(OH::Param::Descriptor& d, OH::Param::SetInstance* s) : ValueParam(d, s, 1, false) {}
    OfxStatus get(int& v) override { v = (int)_v[0]; return kOfxStatOK; }
    OfxStatus get(OfxTime, int& v) override { return get(v); }
    OfxStatus set(int v) override { _v[0] = v; return kOfxStatOK; }
    OfxStatus set(OfxTime, int v) override { return set(v); }
};

class BoolP : public ValueParam<OH::Param::BooleanInstance> {
public:
    BoolP(OH::Param::Descriptor& d, OH::Param::SetInstance* s) : ValueParam(d, s, 1, false) {}
    OfxStatus get(bool& v) override { v = _v[0] != 0.0; return kOfxStatOK; }
    OfxStatus get(OfxTime, bool& v) override { return get(v); }
    OfxStatus set(bool v) override { _v[0] = v ? 1.0 : 0.0; return kOfxStatOK; }
    OfxStatus set(OfxTime, bool v) override { return set(v); }
};

class DoubleP : public ValueParam<OH::Param::DoubleInstance> {
public:
    DoubleP(OH::Param::Descriptor& d, OH::Param::SetInstance* s) : ValueParam(d, s, 1, true) {}
    OfxStatus get(double& v) override { v = _v[0]; return kOfxStatOK; }
    OfxStatus get(OfxTime, double& v) override { return get(v); }
    OfxStatus set(double v) override { _v[0] = v; return kOfxStatOK; }
    OfxStatus set(OfxTime, double v) override { return set(v); }
    OfxStatus derive(OfxTime, double& v) override { v = 0.0; return kOfxStatOK; }
    OfxStatus integrate(OfxTime a, OfxTime b, double& v) override { v = _v[0] * (b - a); return kOfxStatOK; }
};

class Double2DP : public ValueParam<OH::Param::Double2DInstance> {
public:
    Double2DP(OH::Param::Descriptor& d, OH::Param::SetInstance* s) : ValueParam(d, s, 2, true) {}
    OfxStatus get(double& x, double& y) override { x = _v[0]; y = _v[1]; return kOfxStatOK; }
    OfxStatus get(OfxTime, double& x, double& y) override { return get(x, y); }
    OfxStatus set(double x, double y) override { _v[0] = x; _v[1] = y; return kOfxStatOK; }
    OfxStatus set(OfxTime, double x, double y) override { return set(x, y); }
};

class RGBP : public ValueParam<OH::Param::RGBInstance> {
public:
    RGBP(OH::Param::Descriptor& d, OH::Param::SetInstance* s) : ValueParam(d, s, 3, true) {}
    OfxStatus get(double& r, double& g, double& b) override { r = _v[0]; g = _v[1]; b = _v[2]; return kOfxStatOK; }
    OfxStatus get(OfxTime, double& r, double& g, double& b) override { return get(r, g, b); }
    OfxStatus set(double r, double g, double b) override { _v[0] = r; _v[1] = g; _v[2] = b; return kOfxStatOK; }
    OfxStatus set(OfxTime, double r, double g, double b) override { return set(r, g, b); }
};

class StringP : public OH::Param::StringInstance {
public:
    StringP(OH::Param::Descriptor& d, OH::Param::SetInstance* s)
        : OH::Param::StringInstance(d, s), _v(d.getProperties().getStringProperty(kOfxParamPropDefault)) {}
    OfxStatus get(std::string& v) override { v = _v; return kOfxStatOK; }
    OfxStatus get(OfxTime, std::string& v) override { v = _v; return kOfxStatOK; }
    OfxStatus set(const char* v) override { _v = v; return kOfxStatOK; }
    OfxStatus set(OfxTime, const char* v) override { _v = v; return kOfxStatOK; }
    std::string _v;
};

////////////////////////////////////////////////////////////////////////////////
// Effect instance

class TestEffect : public OH::ImageEffect::Instance {
public:
    TestEffect(OH::ImageEffect::ImageEffectPlugin* plugin, OH::ImageEffect::Descriptor& desc, const std::string& ctx)
        : OH::ImageEffect::Instance(plugin, desc, ctx, false) {}

    const std::string& getDefaultOutputFielding() const override
    {
        static const std::string v(kOfxImageFieldNone);
        return v;
    }
    OH::ImageEffect::ClipInstance* newClipInstance(OH::ImageEffect::Instance* plugin,
                                                   OH::ImageEffect::ClipDescriptor* d, int) override
    {
        return new TestClip(plugin, d);
    }
    OfxStatus vmessage(const char* type, const char*, const char* fmt, va_list args) override
    {
        std::printf("message[%s]: ", type);
        std::vprintf(fmt, args);
        std::printf("\n");
        return kOfxStatOK;
    }
    OfxStatus setPersistentMessage(const char* t, const char* id, const char* f, va_list a) override
    {
        return vmessage(t, id, f, a);
    }
    OfxStatus clearPersistentMessage() override { return kOfxStatOK; }

    void getProjectSize(double& x, double& y) const override { x = g_src.w / g_opt.scale; y = g_src.h / g_opt.scale; }
    void getProjectOffset(double& x, double& y) const override { x = y = 0.0; }
    void getProjectExtent(double& x, double& y) const override { getProjectSize(x, y); }
    double getProjectPixelAspectRatio() const override { return 1.0; }
    double getEffectDuration() const override { return 1000.0; }
    double getFrameRate() const override { return 24.0; }
    double getFrameRecursive() const override { return g_opt.time; }
    void getRenderScaleRecursive(double& x, double& y) const override { x = y = g_opt.scale; }

#ifdef __APPLE__
    // --metal: render in Metal mode on our queue, as Resolve does.
    void setCustomInArgs(const std::string& action, OFX::Host::Property::Set& inArgs) override
    {
        if (!g_opt.metal || action != kOfxImageEffectActionRender) return;
        static const OFX::Host::Property::PropSpec en = { kOfxImageEffectPropMetalEnabled, OFX::Host::Property::eInt, 1, false, "1" };
        static const OFX::Host::Property::PropSpec q = { kOfxImageEffectPropMetalCommandQueue, OFX::Host::Property::ePointer, 1, false, "0" };
        inArgs.createProperty(en);
        inArgs.createProperty(q);
        inArgs.setIntProperty(kOfxImageEffectPropMetalEnabled, 1);
        inArgs.setPointerProperty(kOfxImageEffectPropMetalCommandQueue, fgmQueue());
    }
#endif

    OH::Param::Instance* newParam(const std::string&, OH::Param::Descriptor& d) override
    {
        const std::string& t = d.getType();
        if (t == kOfxParamTypeInteger) return new IntP(d, this);
        if (t == kOfxParamTypeDouble) return new DoubleP(d, this);
        if (t == kOfxParamTypeBoolean) return new BoolP(d, this);
        if (t == kOfxParamTypeChoice) return new ChoiceP(d, this);
        if (t == kOfxParamTypeDouble2D) return new Double2DP(d, this);
        if (t == kOfxParamTypeRGB) return new RGBP(d, this);
        if (t == kOfxParamTypeString) return new StringP(d, this);
        if (t == kOfxParamTypeGroup) return new OH::Param::GroupInstance(d, this);
        if (t == kOfxParamTypePage) return new OH::Param::PageInstance(d, this);
        fail("unsupported parameter type " + t + " for " + d.getName());
        return nullptr;
    }
    OfxStatus editBegin(const std::string&) override { return kOfxStatOK; }
    OfxStatus editEnd() override { return kOfxStatOK; }

    void progressStart(const std::string&, const std::string&) override {}
    void progressEnd() override {}
    bool progressUpdate(double) override { return true; }
    double timeLineGetTime() override { return g_opt.time; }
    void timeLineGotoTime(double) override {}
    void timeLineGetBounds(double& a, double& b) override { a = 0; b = 1000; }
};

////////////////////////////////////////////////////////////////////////////////
// Host (with a real multithread suite)

static thread_local unsigned int t_threadIndex = 0;
static thread_local bool t_spawned = false;

class TestHost : public OH::ImageEffect::Host {
public:
    TestHost()
    {
        _properties.setIntProperty(kOfxPropAPIVersion, 1, 0);
        _properties.setIntProperty(kOfxPropAPIVersion, 4, 1);
        _properties.setStringProperty(kOfxPropName, "com.facetglass.TestHost");
        _properties.setStringProperty(kOfxPropLabel, "FacetGlass Test Host");
        _properties.setIntProperty(kOfxPropVersion, 1, 0);
        _properties.setIntProperty(kOfxPropVersion, 0, 1);
        _properties.setStringProperty(kOfxPropVersionLabel, "1.0");
        _properties.setIntProperty(kOfxImageEffectHostPropIsBackground, 1);
        _properties.setIntProperty(kOfxImageEffectPropSupportsOverlays, 0);
        _properties.setIntProperty(kOfxImageEffectPropSupportsMultiResolution, 1);
        _properties.setIntProperty(kOfxImageEffectPropSupportsTiles, 1);
        _properties.setIntProperty(kOfxImageEffectPropTemporalClipAccess, 0);
        _properties.setStringProperty(kOfxImageEffectPropSupportedComponents, kOfxImageComponentRGBA, 0);
        _properties.setStringProperty(kOfxImageEffectPropSupportedContexts, kOfxImageEffectContextFilter, 0);
        _properties.setStringProperty(kOfxImageEffectPropSupportedContexts, kOfxImageEffectContextGeneral, 1);
        _properties.setStringProperty(kOfxImageEffectPropSupportedPixelDepths, kOfxBitDepthFloat, 0);
        _properties.setIntProperty(kOfxImageEffectPropSupportsMultipleClipDepths, 0);
        _properties.setIntProperty(kOfxImageEffectPropSupportsMultipleClipPARs, 0);
        _properties.setIntProperty(kOfxImageEffectPropSetableFrameRate, 0);
        _properties.setIntProperty(kOfxImageEffectPropSetableFielding, 0);
        _properties.setIntProperty(kOfxParamHostPropSupportsCustomInteract, 0);
        _properties.setIntProperty(kOfxParamHostPropSupportsStringAnimation, 0);
        _properties.setIntProperty(kOfxParamHostPropSupportsChoiceAnimation, 0);
        _properties.setIntProperty(kOfxParamHostPropSupportsBooleanAnimation, 0);
        _properties.setIntProperty(kOfxParamHostPropSupportsCustomAnimation, 0);
        _properties.setIntProperty(kOfxParamHostPropMaxParameters, -1);
        _properties.setIntProperty(kOfxParamHostPropMaxPages, 0);
        _properties.setIntProperty(kOfxParamHostPropPageRowColumnCount, 0, 0);
        _properties.setIntProperty(kOfxParamHostPropPageRowColumnCount, 0, 1);
    }

    // --minimal-host: hide optional suites that some hosts (e.g. Resolve) lack.
    const void* fetchSuite(const char* name, int version) override
    {
        if (g_opt.minimalHost && std::strcmp(name, kOfxMessageSuite) == 0 && version >= 2) return nullptr;
        return OH::ImageEffect::Host::fetchSuite(name, version);
    }

    OH::ImageEffect::Instance* newInstance(void*, OH::ImageEffect::ImageEffectPlugin* plugin,
                                           OH::ImageEffect::Descriptor& desc, const std::string& ctx) override
    {
        return new TestEffect(plugin, desc, ctx);
    }
    OH::ImageEffect::Descriptor* makeDescriptor(OH::ImageEffect::ImageEffectPlugin* plugin) override
    {
        return new OH::ImageEffect::Descriptor(plugin);
    }
    OH::ImageEffect::Descriptor* makeDescriptor(const OH::ImageEffect::Descriptor& root,
                                                OH::ImageEffect::ImageEffectPlugin* plugin) override
    {
        return new OH::ImageEffect::Descriptor(root, plugin);
    }
    OH::ImageEffect::Descriptor* makeDescriptor(const std::string& bundlePath,
                                                OH::ImageEffect::ImageEffectPlugin* plugin) override
    {
        return new OH::ImageEffect::Descriptor(bundlePath, plugin);
    }
    OfxStatus vmessage(const char* type, const char*, const char* fmt, va_list args) override
    {
        std::printf("message[%s]: ", type);
        std::vprintf(fmt, args);
        std::printf("\n");
        return kOfxStatOK;
    }
    OfxStatus setPersistentMessage(const char* t, const char* id, const char* f, va_list a) override
    {
        return vmessage(t, id, f, a);
    }
    OfxStatus clearPersistentMessage() override { return kOfxStatOK; }

    OfxStatus multiThread(OfxThreadFunctionV1 func, unsigned int nThreads, void* arg) override
    {
        if (!func) return kOfxStatFailed;
        unsigned int ncpu = 1;
        multiThreadNumCPUS(&ncpu);
        if (nThreads == 0) nThreads = ncpu;
        if (nThreads == 1) { func(0, 1, arg); return kOfxStatOK; }
        std::vector<std::thread> pool;
        for (unsigned int i = 0; i < nThreads; ++i)
            pool.emplace_back([=] { t_threadIndex = i; t_spawned = true; func(i, nThreads, arg); });
        for (auto& t : pool) t.join();
        return kOfxStatOK;
    }
    OfxStatus multiThreadNumCPUS(unsigned int* n) const override
    {
        *n = std::max(1u, std::thread::hardware_concurrency());
        return kOfxStatOK;
    }
    OfxStatus multiThreadIndex(unsigned int* i) const override { *i = t_threadIndex; return kOfxStatOK; }
    int multiThreadIsSpawnedThread() const override { return t_spawned ? 1 : 0; }
    OfxStatus mutexCreate(OfxMutexHandle* m, int lockCount) override
    {
        auto* mx = new std::recursive_mutex;
        for (int i = 0; i < lockCount; ++i) mx->lock();
        *m = reinterpret_cast<OfxMutexHandle>(mx);
        return kOfxStatOK;
    }
    OfxStatus mutexDestroy(const OfxMutexHandle m) override
    {
        delete reinterpret_cast<std::recursive_mutex*>(m);
        return kOfxStatOK;
    }
    OfxStatus mutexLock(const OfxMutexHandle m) override
    {
        reinterpret_cast<std::recursive_mutex*>(m)->lock();
        return kOfxStatOK;
    }
    OfxStatus mutexUnLock(const OfxMutexHandle m) override
    {
        reinterpret_cast<std::recursive_mutex*>(m)->unlock();
        return kOfxStatOK;
    }
    OfxStatus mutexTryLock(const OfxMutexHandle m) override
    {
        return reinterpret_cast<std::recursive_mutex*>(m)->try_lock() ? kOfxStatOK : kOfxStatFailed;
    }
#ifdef OFX_SUPPORTS_OPENGLRENDER
    OfxStatus flushOpenGLResources() const override { return kOfxStatFailed; }
#endif
};

// A plugin cache that only looks in the directory we give it: the default
// constructor also adds the system OFX folders, which we must not load from.
class IsolatedPluginCache : public OH::PluginCache {
public:
    static IsolatedPluginCache* install(const std::string& dir)
    {
        IsolatedPluginCache* c = new IsolatedPluginCache;
        c->_pluginPath.clear();
        c->_nonrecursePath.clear();
        c->_pluginPath.push_back(dir);
        gPluginCachePtr = c;
        return c;
    }
};

////////////////////////////////////////////////////////////////////////////////

static std::vector<double> parseNumbers(const std::string& s)
{
    std::vector<double> v;
    std::stringstream ss(s);
    std::string item;
    while (std::getline(ss, item, ',')) v.push_back(std::atof(item.c_str()));
    return v;
}

static void parseArgs(int argc, char** argv)
{
    Options& o = g_opt;
    for (int i = 1; i < argc; ++i) {
        std::string a = argv[i];
        auto need = [&](int n) { if (i + n >= argc) fail("missing value for " + a); };
        if (a == "--plugins") { need(1); o.pluginDir = argv[++i]; }
        else if (a == "--in") { need(3); o.inPath = argv[++i]; o.inW = std::atoi(argv[++i]); o.inH = std::atoi(argv[++i]); }
        else if (a == "--refract") { need(3); o.refPath = argv[++i]; o.refW = std::atoi(argv[++i]); o.refH = std::atoi(argv[++i]); }
        else if (a == "--out") { need(1); o.out = argv[++i]; }
        else if (a == "--context") { need(1); o.context = argv[++i]; }
        else if (a == "--scale") { need(1); o.scale = std::atof(argv[++i]); }
        else if (a == "--tiles") { need(1); o.tiles = std::max(1, std::atoi(argv[++i])); }
        else if (a == "--time") { need(1); o.time = std::atof(argv[++i]); }
        else if (a == "--negstride") { o.negStride = true; }
        else if (a == "--minimal-host") { o.minimalHost = true; }
        else if (a == "--interactive") { o.interactive = true; }
        else if (a == "--look") { need(1); o.look = std::atoi(argv[++i]); }
        else if (a == "--repeat") { need(1); o.repeat = std::max(1, std::atoi(argv[++i])); }
        else if (a == "--metal") {
#ifdef __APPLE__
            o.metal = true;
#else
            fail("--metal needs macOS");
#endif
        }
        else if (a.find(":=") != std::string::npos) {
            const size_t eq = a.find(":=");
            o.strParams.emplace_back(a.substr(0, eq), a.substr(eq + 2));
        }
        else if (a.find('=') != std::string::npos) {
            const size_t eq = a.find('=');
            o.params.emplace_back(a.substr(0, eq), parseNumbers(a.substr(eq + 1)));
        }
        else fail("unknown argument " + a);
    }
    if (o.pluginDir.empty() || o.inPath.empty() || o.out.empty()) fail("need --plugins, --in and --out");
    if (o.context.empty()) o.context = o.refPath.empty() ? "filter" : "general";
}

static void setParam(OH::ImageEffect::Instance& inst, const std::string& name, const std::vector<double>& v)
{
    OH::Param::Instance* p = inst.getParam(name);
    if (!p) fail("no parameter named " + name);
    if (v.empty()) fail("no value for " + name);
    OfxStatus st = kOfxStatFailed;
    if (auto* q = dynamic_cast<IntP*>(p)) st = q->set((int)std::lround(v[0]));
    else if (auto* q = dynamic_cast<ChoiceP*>(p)) st = q->set((int)std::lround(v[0]));
    else if (auto* q = dynamic_cast<BoolP*>(p)) st = q->set(v[0] != 0.0);
    else if (auto* q = dynamic_cast<DoubleP*>(p)) st = q->set(v[0]);
    else if (auto* q = dynamic_cast<Double2DP*>(p)) {
        if (v.size() < 2) fail(name + " needs x,y");
        st = q->set(v[0], v[1]);
    }
    else if (auto* q = dynamic_cast<RGBP*>(p)) {
        if (v.size() < 3) fail(name + " needs r,g,b");
        st = q->set(v[0], v[1], v[2]);
    }
    if (st != kOfxStatOK) fail("cannot set " + name);
}

int main(int argc, char** argv)
{
    parseArgs(argc, argv);
    const Options& o = g_opt;

    g_src.load(o.inPath, o.inW, o.inH, o.negStride);
    if (!o.refPath.empty()) g_ref.load(o.refPath, o.refW, o.refH, o.negStride);
    g_out.alloc(o.inW, o.inH, o.negStride);
    std::fill(g_out.mem.begin(), g_out.mem.end(), -1.0f); // unwritten pixels are easy to spot
#ifdef __APPLE__
    if (o.metal) {
        for (Buffer* b : { &g_src, &g_ref, &g_out })
            if (b->w > 0) b->mtl = fgmWrap(b->mem.data(), b->mem.size() * sizeof(float));
        std::printf("metal: on\n");
    }
#endif

    IsolatedPluginCache::install(o.pluginDir);
    OH::PluginCache* cache = OH::PluginCache::getPluginCache();
    cache->setCacheVersion("FacetGlassTestHostV1");
    TestHost host;
    OH::ImageEffect::PluginCache effects(host);
    effects.registerInCache(*cache);
    cache->scanPluginFiles();

    OH::ImageEffect::ImageEffectPlugin* plugin = effects.getPluginById(kPluginId);
    if (!plugin) fail(std::string("plugin ") + kPluginId + " not found in " + o.pluginDir);
    std::printf("plugin: %s %d.%d\n", plugin->getIdentifier().c_str(), plugin->getVersionMajor(), plugin->getVersionMinor());

    const std::string ctx = o.context == "general" ? kOfxImageEffectContextGeneral : kOfxImageEffectContextFilter;
    std::unique_ptr<OH::ImageEffect::Instance> inst(plugin->createInstance(ctx, nullptr));
    if (!inst) fail("createInstance failed for context " + o.context);

    for (const auto& kv : o.params) setParam(*inst, kv.first, kv.second);
    for (const auto& kv : o.strParams) {
        auto* sp = dynamic_cast<StringP*>(inst->getParam(kv.first));
        if (!sp) fail("no text parameter named " + kv.first);
        sp->set(kv.second.c_str());
    }

    check(inst->createInstanceAction(), "createInstanceAction");
    if (!inst->getClipPreferences()) fail("getClipPreferences failed");

    const OfxPointD rs = { o.scale, o.scale };
    const OfxTime t = o.time;

    if (o.look >= 0) {
        // As if the user picked a Look: set the menu, then send the change.
        auto* lp = dynamic_cast<ChoiceP*>(inst->getParam("look"));
        if (!lp) fail("no look parameter");
        lp->set(o.look);
        check(inst->beginInstanceChangedAction(kOfxChangeUserEdited), "beginInstanceChanged");
        check(inst->paramInstanceChangedAction("look", kOfxChangeUserEdited, t, rs), "paramInstanceChanged");
        check(inst->endInstanceChangedAction(kOfxChangeUserEdited), "endInstanceChanged");
        int now = -1;
        lp->get(now);
        std::printf("look: %d\n", now);
    }

    OfxRectD rod;
    check(inst->getRegionOfDefinitionAction(t, rs, rod), "getRegionOfDefinitionAction");
    std::printf("rod: %g %g %g %g\n", rod.x1, rod.y1, rod.x2, rod.y2);

    check(inst->beginRenderAction(t, t, 1.0, false, rs, true, false), "beginRenderAction");

    // Identity check, as a real host does before rendering.
    {
        OfxTime it = t;
        std::string clipName;
        const OfxRectI full = { 0, 0, o.inW, o.inH };
        OfxStatus st = inst->isIdentityAction(it, kOfxImageFieldNone, full, rs, clipName);
        if (st == kOfxStatOK && !clipName.empty()) {
            std::printf("identity: %s\n", clipName.c_str());
            Buffer* b = clipName == "RefractSource" ? &g_ref : &g_src;
            for (int y = 0; y < o.inH; ++y) std::memcpy(g_out.row(y), b->row(y), (size_t)o.inW * 16);
            check(inst->endRenderAction(t, t, 1.0, false, rs, true, false), "endRenderAction");
            g_out.save(o.out);
            inst.reset();
            OH::PluginCache::clearPluginCache();
            return 0;
        }
        std::printf("identity: none\n");
    }

    // Render as an n x n grid of render windows; report the RoIs requested for each.
    int nTiles = 0;
    bool roiFull = true;
    for (int rep = 0; rep < o.repeat; ++rep)
    for (int ty = 0; ty < o.tiles; ++ty) {
        for (int tx = 0; tx < o.tiles; ++tx) {
            OfxRectI win;
            win.x1 = o.inW * tx / o.tiles;
            win.x2 = o.inW * (tx + 1) / o.tiles;
            win.y1 = o.inH * ty / o.tiles;
            win.y2 = o.inH * (ty + 1) / o.tiles;
            if (win.x2 <= win.x1 || win.y2 <= win.y1) continue;
            const OfxRectD winCanon = { win.x1 / o.scale, win.y1 / o.scale, win.x2 / o.scale, win.y2 / o.scale };
            std::map<OH::ImageEffect::ClipInstance*, OfxRectD> rois;
            check(inst->getRegionOfInterestAction(t, rs, winCanon, rois), "getRegionOfInterestAction");
            for (auto& r : rois) {
                const OfxRectD want = r.first->getRegionOfDefinition(t);
                const bool full = r.second.x1 <= want.x1 && r.second.y1 <= want.y1 &&
                                  r.second.x2 >= want.x2 && r.second.y2 >= want.y2;
                roiFull = roiFull && full;
                if (nTiles == 0 || !full)
                    std::printf("roi %s: %g %g %g %g%s\n", r.first->getName().c_str(), r.second.x1, r.second.y1,
                                r.second.x2, r.second.y2, full ? " (full)" : " (PARTIAL)");
            }
            check(inst->renderAction(t, kOfxImageFieldNone, win, rs, !o.interactive, o.interactive, false), "renderAction");
            ++nTiles;
        }
    }
    std::printf("tiles: %d\n", nTiles);
    std::printf("roi_full: %d\n", roiFull ? 1 : 0);

    check(inst->endRenderAction(t, t, 1.0, false, rs, true, false), "endRenderAction");
#ifdef __APPLE__
    if (o.metal) {
        fgmReadBack(g_out.mtl, g_out.mem.data(), g_out.mem.size() * sizeof(float));
        for (Buffer* b : { &g_src, &g_ref, &g_out }) { fgmRelease(b->mtl); b->mtl = nullptr; }
    }
#endif

    size_t unwritten = 0;
    for (size_t i = 0; i < g_out.mem.size(); i += 4)
        unwritten += (g_out.mem[i] == -1.0f && g_out.mem[i + 1] == -1.0f && g_out.mem[i + 2] == -1.0f &&
                      g_out.mem[i + 3] == -1.0f);
    std::printf("unwritten: %zu\n", unwritten);

    g_out.save(o.out);
    inst.reset();
    OH::PluginCache::clearPluginCache();
    return 0;
}
