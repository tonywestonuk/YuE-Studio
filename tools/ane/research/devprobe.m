#import <Foundation/Foundation.h>
#import <objc/runtime.h>
#import <objc/message.h>
#include <dlfcn.h>
static void dump(const char *cls) {
  Class c = objc_getClass(cls); if (!c) { printf("%s: no class\n", cls); return; }
  Class meta = object_getClass(c); unsigned n = 0; Method *ms = class_copyMethodList(meta, &n);
  printf("== +%s (%u class methods)\n", cls, n);
  for (unsigned i = 0; i < n; i++) {
    SEL s = method_getName(ms[i]); const char *name = sel_getName(s);
    if (strchr(name, ':')) { printf("  %s (args)\n", name); continue; }
    char ret[16]; method_getReturnType(ms[i], ret, sizeof ret);
    if (ret[0] == '@') { id v = nil; @try { v = ((id(*)(Class,SEL))objc_msgSend)(c, s); } @catch (id e) { v = @"<threw>"; }
      NSString *d = [v description]; if (d.length > 120) d = [[d substringToIndex:120] stringByAppendingString:@"…"]; printf("  %s = %s\n", name, [d UTF8String]); }
    else if (ret[0] == 'B' || ret[0] == 'c') printf("  %s = %d\n", name, (int)((char(*)(Class,SEL))objc_msgSend)(c, s));
    else if (ret[0] == 'q' || ret[0] == 'Q' || ret[0] == 'i' || ret[0] == 'I' || ret[0] == 'l' || ret[0] == 'L') printf("  %s = %lld\n", name, (long long)((long long(*)(Class,SEL))objc_msgSend)(c, s));
    else printf("  %s (returns %s)\n", name, ret);
  }
}
int main() { @autoreleasepool { dlopen("/System/Library/PrivateFrameworks/AppleNeuralEngine.framework/AppleNeuralEngine", RTLD_NOW); dump("_ANEDeviceInfo"); dump("_ANEStrings"); } return 0; }
