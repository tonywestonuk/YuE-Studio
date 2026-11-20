#import <Foundation/Foundation.h>
#import <objc/runtime.h>
#import <objc/message.h>
#include <dlfcn.h>
int main(int argc, char **argv) {
  @autoreleasepool {
    void *h = dlopen("/System/Library/PrivateFrameworks/AppleNeuralEngine.framework/AppleNeuralEngine", RTLD_NOW);
    const char *keys[] = {"kANEFCompilerOptionsFilenameKey","kANEFInMemoryModelIsCachedKey","kANEFModelPreCompiled","kANEFSkipPreparePhaseKey","kANEFPerformanceStatsMask","kANEFModelType","kANEFModelMIL","kANEFIsInMemoryModelTypeKey","kANEFKeepModelMemoryWiredKey","kANEFEnablePowerSavingKey","kANEFModelIdentityStrKey","kANEFAOTCacheUrlIdentifierKey","kANEFModelCacheIdentifierUsingSourceURLKey","kANEFRetainModelsWithoutSourceURLKey","kANEFCompilationInitiatedByE5MLKey","kANEModelKeyEspressoTranslationOptions", NULL};
    for (int i = 0; keys[i]; i++) { NSString *__strong *p = (NSString *__strong *)dlsym(h, keys[i]); printf("%s = %s\n", keys[i], p ? [[*p description] UTF8String] : "(no symbol)"); }
    Class Desc = objc_getClass("_ANEInMemoryModelDescriptor"), IMM = objc_getClass("_ANEInMemoryModel");
    printf("defaultCompilerOptionsFilename = %s\n", [[((id(*)(Class,SEL))objc_msgSend)(IMM, @selector(defaultCompilerOptionsFilename)) description] UTF8String]);
    for (int i = 1; i < argc; i++) {
      NSString *dir = [NSString stringWithUTF8String:argv[i]];
      NSData *mil = [NSData dataWithContentsOfFile:[dir stringByAppendingPathComponent:@"model.mil"]];
      NSData *wb = [NSData dataWithContentsOfFile:[dir stringByAppendingPathComponent:@"weights/weight.bin"] options:NSDataReadingMappedAlways error:nil];
      id desc = ((id(*)(Class,SEL,id,id,id))objc_msgSend)(Desc, @selector(modelWithMILText:weights:optionsPlist:), mil, @{@"@model_path/weights/weight.bin": @{@"offset": @0, @"data": wb}}, nil);
      id model = ((id(*)(Class,SEL,id))objc_msgSend)(IMM, @selector(inMemoryModelWithDescriptor:), desc);
      printf("%s: compiledModelExists=%d\n", [[dir lastPathComponent].stringByDeletingLastPathComponent UTF8String] ?: argv[i], (int)((BOOL(*)(id,SEL))objc_msgSend)(model, @selector(compiledModelExists)));
    }
  }
  return 0;
}
