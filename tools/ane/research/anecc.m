// Call the ANE compiler library in-process. usage: anecc <program dir> <out dir> <arch> [--opts options.plist] [--flag key=value ...] [--in key=value ...]
#import <Foundation/Foundation.h>
#include <dlfcn.h>
static id parse(NSString *v) {
  if ([v isEqualToString:@"true"]) return @YES; if ([v isEqualToString:@"false"]) return @NO;
  NSScanner *sc = [NSScanner scannerWithString:v]; long long n; if ([sc scanLongLong:&n] && sc.isAtEnd) return @(n); return v; }
int main(int argc, char **argv) {
  @autoreleasepool {
    void *h = dlopen("/System/Library/PrivateFrameworks/ANECompiler.framework/ANECompiler", RTLD_NOW);
    int (*ANECCompile)(NSDictionary *, NSDictionary *, void *) = dlsym(h, "ANECCompile");
    NSString *dir = [NSString stringWithUTF8String:argv[1]], *out = [NSString stringWithUTF8String:argv[2]], *arch = [NSString stringWithUTF8String:argv[3]];
    [[NSFileManager defaultManager] createDirectoryAtPath:out withIntermediateDirectories:YES attributes:nil error:nil];
    NSMutableDictionary *input = [@{@"InputNetworks": @[@{@"NetworkSourceFileName": @"model.mil", @"NetworkSourcePath": [dir stringByAppendingString:@"/"]}],
                                    @"OutputFileName": @"model.hwx", @"OutputFilePath": [out stringByAppendingString:@"/"]} mutableCopy];
    NSMutableDictionary *flags = [@{@"TargetArchitecture": arch} mutableCopy];
    for (int i = 4; i < argc; i++) {
      NSString *a = [NSString stringWithUTF8String:argv[i]];
      if ([a isEqualToString:@"--opts"] && i + 1 < argc) { input[@"OptionsFilePath"] = [NSString stringWithUTF8String:argv[++i]]; continue; }
      if (([a isEqualToString:@"--flag"] || [a isEqualToString:@"--in"]) && i + 1 < argc) {
        NSString *kv = [NSString stringWithUTF8String:argv[++i]]; NSRange r = [kv rangeOfString:@"="];
        NSMutableDictionary *d = [a isEqualToString:@"--flag"] ? flags : input;
        d[[kv substringToIndex:r.location]] = parse([kv substringFromIndex:r.location + 1]);
      }
    }
    fprintf(stderr, "flags=%s\n", [[flags description] UTF8String]);
    NSDate *t0 = [NSDate date];
    int ret = ANECCompile(input, flags, NULL);
    printf("ret=%d time=%.2fs", ret, -[t0 timeIntervalSinceNow]);
    for (NSString *f in [[NSFileManager defaultManager] contentsOfDirectoryAtPath:out error:nil]) {
      NSDictionary *a = [[NSFileManager defaultManager] attributesOfItemAtPath:[out stringByAppendingPathComponent:f] error:nil];
      printf("  %s=%llu", [f UTF8String], [a fileSize]);
    }
    printf("\n");
  }
  return 0;
}
