/*
 * libyue2ane — minimal C API over the private AppleNeuralEngine in-memory model
 * route (no Core ML). Programs are MIL text + a weight blob; tensors travel in
 * IOSurfaces so layer outputs can feed the next program with no copy.
 *
 * Build: clang -O2 -fobjc-arc -dynamiclib libyue2ane.m -o libyue2ane.dylib \
 *            -framework Foundation -framework IOSurface
 */
#import <Foundation/Foundation.h>
#import <IOSurface/IOSurface.h>
#import <objc/runtime.h>
#import <objc/message.h>
#import <dlfcn.h>
#import <mach/mach_time.h>

typedef struct { IOSurfaceRef surface; id wrapper; size_t bytes; } ane_surface;
typedef struct { id model; NSString *tmp; } ane_program;

static Class gDesc, gIMM, gReq, gSurf;
static int ensure(char *err, int errlen) {
    if (gDesc) return 0;
    if (!dlopen("/System/Library/PrivateFrameworks/AppleNeuralEngine.framework/AppleNeuralEngine", RTLD_NOW)) {
        snprintf(err, errlen, "dlopen AppleNeuralEngine failed"); return -1;
    }
    gDesc = NSClassFromString(@"_ANEInMemoryModelDescriptor"); gIMM = NSClassFromString(@"_ANEInMemoryModel");
    gReq = NSClassFromString(@"_ANERequest"); gSurf = NSClassFromString(@"_ANEIOSurfaceObject");
    if (!gDesc || !gIMM || !gReq || !gSurf) { snprintf(err, errlen, "private ANE classes missing"); gDesc = nil; return -1; }
    return 0;
}
// The whole error, including its chain of underlying errors (NSUnderlyingErrorKey, recursively):
// the compiler's actual reason, e.g. com.apple.appleneuralengine.compiler Code=22 "Compiler internal
// error: Couldn't do topological sort", is one or two levels below the error that the API returns.
// Python logs the text as it is; the buffer sizes in runtime.py allow for it.
static void seterr(char *err, int errlen, NSError *e, const char *what) {
    @autoreleasepool {
        NSMutableString *s = [NSMutableString stringWithFormat:@"%s: %@", what, e ? [e description] : @"unknown error"];
        NSError *u = e;
        for (int depth = 0; depth < 8; depth++) {
            id next = u.userInfo[NSUnderlyingErrorKey];
            if (![next isKindOfClass:[NSError class]]) break;
            u = next;
            [s appendFormat:@"\n  caused by: %@ Code=%ld \"%@\"", u.domain, (long)u.code, u.localizedDescription];
            NSString *reason = u.userInfo[NSLocalizedFailureReasonErrorKey];
            if ([reason isKindOfClass:[NSString class]] && ![reason isEqualToString:u.localizedDescription])
                [s appendFormat:@" (%@)", reason];
        }
        snprintf(err, errlen, "%s", [s UTF8String] ?: "unknown error");
    }
}

ane_surface *ane_surface_create(size_t bytes) {
    char err[128]; if (ensure(err, sizeof err)) return NULL;
    ane_surface *s = calloc(1, sizeof *s);
    s->bytes = bytes;
    s->surface = IOSurfaceCreate((__bridge CFDictionaryRef)@{(id)kIOSurfaceWidth: @(bytes), (id)kIOSurfaceHeight: @1,
        (id)kIOSurfaceBytesPerElement: @1, (id)kIOSurfaceBytesPerRow: @(bytes), (id)kIOSurfaceAllocSize: @(bytes), (id)kIOSurfacePixelFormat: @0});
    if (!s->surface) { free(s); return NULL; }
    s->wrapper = ((id(*)(Class,SEL,IOSurfaceRef))objc_msgSend)(gSurf, @selector(objectWithIOSurface:), s->surface);
    return s;
}
size_t ane_surface_bytes(ane_surface *s) { return s->bytes; }
void ane_surface_write(ane_surface *s, const void *src, size_t bytes) {
    IOSurfaceLock(s->surface, 0, NULL); memcpy(IOSurfaceGetBaseAddress(s->surface), src, bytes); IOSurfaceUnlock(s->surface, 0, NULL);
}
void ane_surface_read(ane_surface *s, void *dst, size_t bytes) {
    IOSurfaceLock(s->surface, kIOSurfaceLockReadOnly, NULL); memcpy(dst, IOSurfaceGetBaseAddress(s->surface), bytes); IOSurfaceUnlock(s->surface, kIOSurfaceLockReadOnly, NULL);
}
void ane_surface_free(ane_surface *s) { if (!s) return; s->wrapper = nil; CFRelease(s->surface); free(s); }

// Compile a program and, if do_load, load it into the engine (mapping its arena in the process's
// ~3.5 GiB address window). Background compiles pass do_load=0 so a second big program is never
// mapped beside the one that is running: that overflowed the window and failed the running
// inference (status 0x2). Load later with ane_program_reload.
ane_program *ane_program_compile(const char *dir, int do_load, double *compile_seconds, char *err, int errlen) {
    if (ensure(err, errlen)) return NULL;
    @autoreleasepool {
        NSString *d = [NSString stringWithUTF8String:dir];
        NSString *milPath = [d stringByAppendingPathComponent:@"model.mil"];
        NSString *weightPath = [d stringByAppendingPathComponent:@"weights/weight.bin"];
        NSData *mil = [NSData dataWithContentsOfFile:milPath];
        // Memory-mapped: the weight blob stays file-backed (clean, evictable pages) instead of a
        // private copy that the descriptor would hold for the life of the program.
        NSData *wb = [NSData dataWithContentsOfFile:weightPath options:NSDataReadingMappedAlways error:nil];
        if (!mil || !wb) { snprintf(err, errlen, "missing model.mil or weights/weight.bin in %s", dir); return NULL; }
        id desc = ((id(*)(Class,SEL,id,id,id))objc_msgSend)(gDesc, @selector(modelWithMILText:weights:optionsPlist:),
                     mil, @{@"@model_path/weights/weight.bin": @{@"offset": @0, @"data": wb}}, nil);
        if (!desc) { snprintf(err, errlen, "descriptor creation failed"); return NULL; }
        id model = ((id(*)(Class,SEL,id))objc_msgSend)(gIMM, @selector(inMemoryModelWithDescriptor:), desc);
        NSString *hexId = ((id(*)(id,SEL))objc_msgSend)(model, @selector(hexStringIdentifier));
        // The framework keys its working directory on the model identifier under the temporary
        // directory: it reads the MIL/weights there and writes its compiled output (net.plist, data)
        // beside them. (A retained directory does not make a later compile any faster.)
        NSString *tmp = [NSTemporaryDirectory() stringByAppendingPathComponent:hexId];
        NSFileManager *fm = [NSFileManager defaultManager];
        [fm removeItemAtPath:tmp error:nil];
        [fm createDirectoryAtPath:[tmp stringByAppendingPathComponent:@"weights"] withIntermediateDirectories:YES attributes:nil error:nil];
        [mil writeToFile:[tmp stringByAppendingPathComponent:@"model.mil"] atomically:YES];
        // APFS clone (no extra space, no copy); fall back to a byte copy on other file systems.
        NSString *dst = [tmp stringByAppendingPathComponent:@"weights/weight.bin"];
        if (![fm copyItemAtPath:weightPath toPath:dst error:nil]) [wb writeToFile:dst atomically:YES];
        NSError *e = nil;
        NSDate *t0 = [NSDate date];
        if (!((BOOL(*)(id,SEL,unsigned int,id,NSError**))objc_msgSend)(model, @selector(compileWithQoS:options:error:), 21, @{}, &e)) {
            seterr(err, errlen, e, "compile"); [fm removeItemAtPath:tmp error:nil]; return NULL;
        }
        if (compile_seconds) *compile_seconds = -[t0 timeIntervalSinceNow];
        if (do_load && !((BOOL(*)(id,SEL,unsigned int,id,NSError**))objc_msgSend)(model, @selector(loadWithQoS:options:error:), 21, @{}, &e)) {
            seterr(err, errlen, e, "load"); [fm removeItemAtPath:tmp error:nil]; return NULL;
        }
        ane_program *p = calloc(1, sizeof *p);
        p->model = model; p->tmp = tmp;
        return p;
    }
}

ane_program *ane_program_load(const char *dir, double *compile_seconds, char *err, int errlen) {
    return ane_program_compile(dir, 1, compile_seconds, err, errlen);
}

int ane_program_eval(ane_program *p, ane_surface **ins, int n_in, ane_surface **outs, int n_out, char *err, int errlen) {
    @autoreleasepool {
        NSMutableArray *i = [NSMutableArray array], *ii = [NSMutableArray array], *o = [NSMutableArray array], *oi = [NSMutableArray array];
        for (int k = 0; k < n_in; k++) { [i addObject:ins[k]->wrapper]; [ii addObject:@(k)]; }
        for (int k = 0; k < n_out; k++) { [o addObject:outs[k]->wrapper]; [oi addObject:@(k)]; }
        id req = ((id(*)(Class,SEL,id,id,id,id,id,id,id))objc_msgSend)(gReq,
            @selector(requestWithInputs:inputIndices:outputs:outputIndices:weightsBuffer:perfStats:procedureIndex:), i, ii, o, oi, nil, nil, @0);
        NSError *e = nil;
        if (!((BOOL(*)(id,SEL,unsigned int,id,id,NSError**))objc_msgSend)(p->model, @selector(evaluateWithQoS:options:request:error:), 21, @{}, req, &e)) {
            seterr(err, errlen, e, "evaluate"); return -1;
        }
        return 0;
    }
}

int ane_program_unload(ane_program *p, char *err, int errlen) {
    @autoreleasepool {
        NSError *e = nil;
        if (!((BOOL(*)(id,SEL,unsigned int,NSError**))objc_msgSend)(p->model, @selector(unloadWithQoS:error:), 21, &e)) { seterr(err, errlen, e, "unload"); return -1; }
        return 0;
    }
}

int ane_program_reload(ane_program *p, char *err, int errlen) {
    @autoreleasepool {
        NSError *e = nil;
        if (!((BOOL(*)(id,SEL,unsigned int,id,NSError**))objc_msgSend)(p->model, @selector(loadWithQoS:options:error:), 21, @{}, &e)) { seterr(err, errlen, e, "load"); return -1; }
        return 0;
    }
}

void ane_program_free(ane_program *p) {
    if (!p) return;
    @autoreleasepool {
        NSError *e = nil;
        ((BOOL(*)(id,SEL,unsigned int,NSError**))objc_msgSend)(p->model, @selector(unloadWithQoS:error:), 21, &e);
        [[NSFileManager defaultManager] removeItemAtPath:p->tmp error:nil];
        p->model = nil; p->tmp = nil;
    }
    free(p);
}
