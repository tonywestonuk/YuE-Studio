// Does the options plist reach the in-memory descriptor? Print the identifiers it derives.
#import <Foundation/Foundation.h>
#import <objc/runtime.h>
#import <objc/message.h>
#include <dlfcn.h>
int main(int argc, char **argv) {
  @autoreleasepool {
    dlopen("/System/Library/PrivateFrameworks/AppleNeuralEngine.framework/AppleNeuralEngine", RTLD_NOW);
    Class Desc = objc_getClass("_ANEInMemoryModelDescriptor"), IMM = objc_getClass("_ANEInMemoryModel");
    NSString *dir = [NSString stringWithUTF8String:argv[1]];
    NSData *mil = [NSData dataWithContentsOfFile:[dir stringByAppendingPathComponent:@"model.mil"]];
    NSData *wb = [NSData dataWithContentsOfFile:[dir stringByAppendingPathComponent:@"weights/weight.bin"]];
    NSData *opts = argc > 2 ? [NSData dataWithContentsOfFile:[NSString stringWithUTF8String:argv[2]]] : nil;
    printf("opts bytes: %lu\n", (unsigned long)opts.length);
    id desc = ((id(*)(Class,SEL,id,id,id))objc_msgSend)(Desc, @selector(modelWithMILText:weights:optionsPlist:),
                 mil, @{@"@model_path/weights/weight.bin": @{@"offset": @0, @"data": wb}}, opts);
    printf("descriptor: %s\n", [[desc description] UTF8String]);
    // dump every property of the descriptor
    unsigned n = 0; objc_property_t *props = class_copyPropertyList(Desc, &n);
    for (unsigned i = 0; i < n; i++) {
      const char *name = property_getName(props[i]);
      id v = nil; @try { v = [desc valueForKey:[NSString stringWithUTF8String:name]]; } @catch (id e) { v = @"<n/a>"; }
      NSString *s = [v description]; if (s.length > 160) s = [[s substringToIndex:160] stringByAppendingString:@"…"];
      printf("  %s = %s\n", name, [s UTF8String]);
    }
    id model = ((id(*)(Class,SEL,id))objc_msgSend)(IMM, @selector(inMemoryModelWithDescriptor:), desc);
    printf("hexStringIdentifier: %s\n", [((id(*)(id,SEL))objc_msgSend)(model, @selector(hexStringIdentifier)) UTF8String]);
    n = 0; props = class_copyPropertyList(IMM, &n);
    for (unsigned i = 0; i < n; i++) printf("  model prop: %s\n", property_getName(props[i]));
    // methods of the descriptor and model (names only) for later use
    unsigned m = 0; Method *ms = class_copyMethodList(Desc, &m);
    printf("descriptor methods:"); for (unsigned i = 0; i < m; i++) printf(" %s", sel_getName(method_getName(ms[i]))); printf("\n");
    m = 0; ms = class_copyMethodList(IMM, &m);
    printf("model methods:"); for (unsigned i = 0; i < m; i++) printf(" %s", sel_getName(method_getName(ms[i]))); printf("\n");
  }
  return 0;
}
