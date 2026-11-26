// Metal helpers for the test host's --metal mode: images become shared MTLBuffers
// and render gets a command queue, as DaVinci Resolve does on macOS.
// SPDX-License-Identifier: MIT
#import <Metal/Metal.h>

#include <cstring>

static id<MTLDevice> device()
{
    static id<MTLDevice> d = MTLCreateSystemDefaultDevice();
    return d;
}

extern "C" void* fgmQueue()
{
    static id<MTLCommandQueue> q = [device() newCommandQueue];
    return (__bridge void*)q;
}

// Shared buffer holding a copy of `bytes` bytes at `data`. Owned by the caller.
extern "C" void* fgmWrap(const void* data, size_t bytes)
{
    id<MTLBuffer> b = [device() newBufferWithBytes:data length:bytes options:MTLResourceStorageModeShared];
    return (__bridge_retained void*)b;
}

// Waits for all work on the queue, then copies the buffer back.
extern "C" void fgmReadBack(void* buf, void* data, size_t bytes)
{
    id<MTLCommandBuffer> cb = [(__bridge id<MTLCommandQueue>)fgmQueue() commandBuffer];
    [cb commit];
    [cb waitUntilCompleted];
    std::memcpy(data, ((__bridge id<MTLBuffer>)buf).contents, bytes);
}

extern "C" void fgmRelease(void* buf)
{
    if (buf) CFRelease(buf);
}
