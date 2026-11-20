/*
 * libyue2ane — minimal C API over the private AppleNeuralEngine in-memory model
 * route (no Core ML). Programs are MIL text + a weight blob; tensors travel in
 * IOSurfaces so layer outputs can feed the next program with no copy.
 * The route (descriptor, compile/load, IOSurface I/O, the model.mil temp-dir
 * quirk) follows https://github.com/maderix/ANE.
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

// ---- Compiling ---------------------------------------------------------------------------------
// Two routes. (1) In-process: ANECompiler.framework's ANECCompile() is called directly with the MIL
// and writes model.hwx (the engine's program binary) to a directory we choose; the daemon later
// loads that file. This is what Core ML's own E5RT cache holds too. It is 3x faster than the XPC
// service for our graphs with the flags Python passes (the service ignores compiler options), runs
// in parallel across processes (the service is one machine-wide queue), and the file is reusable
// across launches and reboots. (2) The service: compileWithQoS: on the in-memory model, kept as the
// fallback when the in-process compile fails.

static NSString *device_string(SEL sel) {
    Class info = NSClassFromString(@"_ANEDeviceInfo");
    if (!info || ![info respondsToSelector:sel]) return nil;
    id v = ((id(*)(Class,SEL))objc_msgSend)(info, sel);
    return [v isKindOfClass:[NSString class]] ? v : nil;
}

// "h16g-25F84": the engine generation and the OS build, which together decide whether a model.hwx
// compiled earlier is still the right one for this machine.
int ane_program_tag(char *out, int outlen) {
    @autoreleasepool {
        if (ensure(out, outlen)) return -1;
        NSString *arch = device_string(@selector(aneSubTypeAndVariant)), *build = device_string(@selector(buildVersion));
        if (!arch) { snprintf(out, outlen, "aneSubTypeAndVariant unavailable"); return -1; }
        snprintf(out, outlen, "%s-%s", [arch UTF8String], build ? [build UTF8String] : "unknown");
        return 0;
    }
}

// Compile dir/model.mil (+ weights) to out_dir/model.hwx in this process. flags_json: a JSON object of
// ANE compiler options (CamelCase names such as MemCacheStrategy, DisableCpAllocation) merged over
// {"TargetArchitecture": <this engine>}. Returns 0 on success.
int ane_compile_hwx(const char *dir, const char *out_dir, const char *flags_json, double *seconds, char *err, int errlen) {
    @autoreleasepool {
        if (ensure(err, errlen)) return -1;
        static int (*compile)(NSDictionary *, NSDictionary *, void *);
        if (!compile) {
            void *h = dlopen("/System/Library/PrivateFrameworks/ANECompiler.framework/ANECompiler", RTLD_NOW);
            compile = h ? dlsym(h, "ANECCompile") : NULL;
            if (!compile) { snprintf(err, errlen, "ANECompiler.framework / ANECCompile unavailable"); return -1; }
        }
        NSString *arch = device_string(@selector(aneSubTypeAndVariant));
        if (!arch) { snprintf(err, errlen, "cannot determine the engine generation"); return -1; }
        NSMutableDictionary *flags = [@{@"TargetArchitecture": arch} mutableCopy];
        if (flags_json && *flags_json) {
            NSError *je = nil;
            id parsed = [NSJSONSerialization JSONObjectWithData:[NSData dataWithBytes:flags_json length:strlen(flags_json)] options:0 error:&je];
            if (![parsed isKindOfClass:[NSDictionary class]]) { snprintf(err, errlen, "compiler flags are not a JSON object"); return -1; }
            [flags addEntriesFromDictionary:parsed];
        }
        NSString *d = [NSString stringWithUTF8String:dir], *o = [NSString stringWithUTF8String:out_dir];
        NSFileManager *fm = [NSFileManager defaultManager];
        [fm createDirectoryAtPath:o withIntermediateDirectories:YES attributes:nil error:nil];
        NSString *hwx = [o stringByAppendingPathComponent:@"model.hwx"];
        [fm removeItemAtPath:hwx error:nil];
        NSDictionary *input = @{@"InputNetworks": @[@{@"NetworkSourceFileName": @"model.mil", @"NetworkSourcePath": [d stringByAppendingString:@"/"]}],
                                @"OutputFileName": @"model.hwx", @"OutputFilePath": [o stringByAppendingString:@"/"]};
        NSDate *t0 = [NSDate date];
        int ret = compile(input, flags, NULL);
        if (seconds) *seconds = -[t0 timeIntervalSinceNow];
        unsigned long long size = [[fm attributesOfItemAtPath:hwx error:nil] fileSize];
        if (ret != 0 || size == 0) {
            [fm removeItemAtPath:hwx error:nil];
            snprintf(err, errlen, "ANECCompile returned %d (%llu-byte model.hwx) for %s with flags %s", ret, size, dir, flags_json ? flags_json : "{}");
            return ret ? ret : -1;
        }
        return 0;
    }
}

// Open a program: the in-memory model for dir/model.mil with the weight blob bound. With hwx (a
// model.hwx from ane_compile_hwx) the engine loads that file and no compile happens; without it the
// compiler service compiles (the original route). do_load=0 leaves the program unloaded (no engine
// mappings) until ane_program_reload: a second big program mapped beside the running one overflowed
// the ~3.5 GiB window and failed the running inference (status 0x2).
ane_program *ane_program_open(const char *dir, const char *hwx, int do_load, double *compile_seconds, char *err, int errlen) {
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
        // beside them. (A retained directory does not make a later service compile any faster.)
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
        if (hwx && *hwx) {
            Class ANEModel = NSClassFromString(@"_ANEModel");
            id m = ANEModel ? ((id(*)(Class,SEL,id,id))objc_msgSend)(ANEModel, @selector(modelAtURL:key:),
                                   [NSURL fileURLWithPath:[NSString stringWithUTF8String:hwx]], @"") : nil;
            if (!m) { snprintf(err, errlen, "precompiled: _ANEModel modelAtURL: failed for %s", hwx); [fm removeItemAtPath:tmp error:nil]; return NULL; }
            ((void(*)(id,SEL,id))objc_msgSend)(model, @selector(setModel:), m);
            if (compile_seconds) *compile_seconds = 0.0;
        } else {
            if (!((BOOL(*)(id,SEL,unsigned int,id,NSError**))objc_msgSend)(model, @selector(compileWithQoS:options:error:), 21, @{}, &e)) {
                seterr(err, errlen, e, "compile"); [fm removeItemAtPath:tmp error:nil]; return NULL;
            }
            if (compile_seconds) *compile_seconds = -[t0 timeIntervalSinceNow];
        }
        if (do_load && !((BOOL(*)(id,SEL,unsigned int,id,NSError**))objc_msgSend)(model, @selector(loadWithQoS:options:error:), 21, @{}, &e)) {
            seterr(err, errlen, e, hwx ? "load (precompiled)" : "load"); [fm removeItemAtPath:tmp error:nil]; return NULL;
        }
        ane_program *p = calloc(1, sizeof *p);
        p->model = model; p->tmp = tmp;
        return p;
    }
}

ane_program *ane_program_compile(const char *dir, int do_load, double *compile_seconds, char *err, int errlen) {
    return ane_program_open(dir, NULL, do_load, compile_seconds, err, errlen);
}

ane_program *ane_program_load(const char *dir, double *compile_seconds, char *err, int errlen) {
    return ane_program_open(dir, NULL, 1, compile_seconds, err, errlen);
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
