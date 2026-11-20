// Where does the in-memory route keep its compiled program, and does it survive a process?
// usage: cacheprobe <program dir> [options.plist] [--no-purge]
#import <Foundation/Foundation.h>
#import <objc/runtime.h>
#import <objc/message.h>
#include <dlfcn.h>
static id call0(id o, SEL s) { return ((id(*)(id,SEL))objc_msgSend)(o, s); }
static void show(id model, const char *when) {
  printf("[%s] compiledModelExists=%d modelURL=%s localModelPath=%s compilerOptionsFileName=%s state=%ld programHandle=%llu\n", when,
    (int)((BOOL(*)(id,SEL))objc_msgSend)(model, @selector(compiledModelExists)),
    [[call0(model, @selector(modelURL)) description] UTF8String] ?: "(nil)",
    [[call0(model, @selector(localModelPath)) description] UTF8String] ?: "(nil)",
    [[call0(model, @selector(compilerOptionsFileName)) description] UTF8String] ?: "(nil)",
    (long)((long(*)(id,SEL))objc_msgSend)(model, @selector(state)),
    (unsigned long long)((unsigned long long(*)(id,SEL))objc_msgSend)(model, @selector(programHandle)));
  id attrs = call0(model, @selector(modelAttributes));
  if (attrs) { NSString *d = [attrs description]; if (d.length > 1500) d = [d substringToIndex:1500]; printf("  modelAttributes: %s\n", [d UTF8String]); }
}
int main(int argc, char **argv) {
  @autoreleasepool {
    dlopen("/System/Library/PrivateFrameworks/AppleNeuralEngine.framework/AppleNeuralEngine", RTLD_NOW);
    Class Desc = objc_getClass("_ANEInMemoryModelDescriptor"), IMM = objc_getClass("_ANEInMemoryModel");
    NSString *dir = [NSString stringWithUTF8String:argv[1]];
    NSData *opts = (argc > 2 && argv[2][0] != '-') ? [NSData dataWithContentsOfFile:[NSString stringWithUTF8String:argv[2]]] : nil;
    BOOL purge = YES; for (int i = 2; i < argc; i++) if (!strcmp(argv[i], "--no-purge")) purge = NO;
    NSData *mil = [NSData dataWithContentsOfFile:[dir stringByAppendingPathComponent:@"model.mil"]];
    NSData *wb = [NSData dataWithContentsOfFile:[dir stringByAppendingPathComponent:@"weights/weight.bin"] options:NSDataReadingMappedAlways error:nil];
    id desc = ((id(*)(Class,SEL,id,id,id))objc_msgSend)(Desc, @selector(modelWithMILText:weights:optionsPlist:),
                 mil, @{@"@model_path/weights/weight.bin": @{@"offset": @0, @"data": wb}}, opts);
    id model = ((id(*)(Class,SEL,id))objc_msgSend)(IMM, @selector(inMemoryModelWithDescriptor:), desc);
    NSString *hexId = call0(model, @selector(hexStringIdentifier));
    NSString *tmp = [NSTemporaryDirectory() stringByAppendingPathComponent:hexId];
    NSFileManager *fm = [NSFileManager defaultManager];
    [fm createDirectoryAtPath:[tmp stringByAppendingPathComponent:@"weights"] withIntermediateDirectories:YES attributes:nil error:nil];
    [mil writeToFile:[tmp stringByAppendingPathComponent:@"model.mil"] atomically:YES];
    [wb writeToFile:[tmp stringByAppendingPathComponent:@"weights/weight.bin"] atomically:YES];
    show(model, "before compile");
    NSError *e = nil; NSDate *t0 = [NSDate date];
    BOOL ok = ((BOOL(*)(id,SEL,unsigned int,id,NSError**))objc_msgSend)(model, @selector(compileWithQoS:options:error:), 21, @{}, &e);
    printf("compile ok=%d in %.2f s %s\n", ok, -[t0 timeIntervalSinceNow], ok ? "" : [[e description] UTF8String]);
    show(model, "after compile");
    t0 = [NSDate date];
    ok = ((BOOL(*)(id,SEL,unsigned int,id,NSError**))objc_msgSend)(model, @selector(loadWithQoS:options:error:), 21, @{}, &e);
    printf("load ok=%d in %.2f s\n", ok, -[t0 timeIntervalSinceNow]);
    show(model, "after load");
    ((BOOL(*)(id,SEL,unsigned int,NSError**))objc_msgSend)(model, @selector(unloadWithQoS:error:), 21, &e);
    if (!purge) { printf("leaving compiled model in place (no purge)\n"); return 0; }
    printf("purgeCompiledModel -> %d\n", (int)((BOOL(*)(id,SEL))objc_msgSend)(model, @selector(purgeCompiledModel)));
    show(model, "after purge");
  }
  return 0;
}
