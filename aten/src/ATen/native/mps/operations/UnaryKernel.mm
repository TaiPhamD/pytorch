#define TORCH_ASSERT_ONLY_METHOD_OPERATORS
#include <ATen/native/DispatchStub.h>
#include <ATen/native/UnaryOps.h>
#ifndef AT_PER_OPERATOR_HEADERS
#include <ATen/Functions.h>
#include <ATen/NativeFunctions.h>
#else
#include <ATen/ops/erfinv_native.h>
#endif

#import <Foundation/Foundation.h>
#import <Metal/Metal.h>
#include <torch/mps.h>

#include <fmt/format.h>

namespace at::native {


// https://github.com/boostorg/math/blob/c56f334348d5476783fba996d604fc5c6ae980c3/include/boost/math/special_functions/detail/erf_inv.hpp

static const char* ERFINV_OPS_TEMPLATE = R"METAL(
 #include <metal_stdlib>
 using namespace metal;

struct Uniforms {{
  /*float a[4] = {{  0.886226899f, -1.645349621f,  0.914624893f, -0.140543331f }};
  float b[4] = {{ -2.118377725f,  1.442710462f, -0.329097515f,  0.012229801f }};
  float c[4] = {{ -1.970840454f, -1.624906493f,  3.429567803f,  1.641345311f }};
  float d[2] = {{  3.543889200f,  1.637067800f }};*/
  float a[4];
  float b[4];
  float c[4];
  float d[2];
}};


 kernel void erfinv_mps_kernel(device {0}  *output [[buffer(0)]],
                               device {1}  *input [[buffer(1)]],
                               constant Uniforms& uniforms [[buffer(2)]],
                         uint index [[thread_position_in_grid]])  {{


  const float *a = uniforms.a;
  const float *b = uniforms.b;
  const float *c = uniforms.c;
  const float *d = uniforms.d;
  float y = input[index];
  float x, z, num, dem; /*working variables */
  /* coefficients in rational expansion */

  float y_abs = abs(y);
  if(y_abs > 1.0f){{
    output[index] = NAN;
    return;
  }}
  if(y_abs == 1.0f){{
    output[index] = copysign(INFINITY, y);
    return;
  }}
  if(y_abs <= 0.7f) {{
    z = y * y;
    num = (((a[3]*z + a[2])*z + a[1])*z + a[0]);
    dem = ((((b[3]*z + b[2])*z + b[1])*z +b[0]) * z + 1.0f);
    x = y * num / dem;
  }}
  else{{
    z = sqrt(-1.0f*log((1.0-y_abs)/2.0));
    num = ((c[3]*z + c[2])*z + c[1]) * z + c[0];
    dem = (d[1]*z + d[0])*z + 1.0f;
    x = copysign(num, y) / dem;
  }} 

  // 2 round newton - erf(x)
  //

  output[index] = x;
}})METAL";

struct Uniforms {
  float a[4] = { 0.886226899f, -1.645349621f,  0.914624893f, -0.140543331f };
  float b[4] = { -2.118377725f,  1.442710462f, -0.329097515f,  0.012229801f };
  float c[4] = { -1.970840454f, -1.624906493f,  3.429567803f,  1.641345311f };
  float d[2] = {  3.543889200f,  1.637067800f };

};
const std::string& getMetalType(const c10::ScalarType& t) {
  // Mapping from c10::ScalarType to integral type that can be used for bitwise ops
  // As bitwise ops sign-agnostic map signed/unsigned char and boolean to the same type
  static std::unordered_map<c10::ScalarType, std::string> scalar_to_metal_type = {
      // int8, uint8, int16, int32, int64, float16, float32
      {c10::ScalarType::Half, "half"},
      {c10::ScalarType::Float, "float"},
      {c10::ScalarType::Long, "long"},
      {c10::ScalarType::Int, "int"},
      {c10::ScalarType::Short, "short"},
      {c10::ScalarType::Bool, "char"},
  };

  auto it = scalar_to_metal_type.find(t);
  TORCH_CHECK(it != scalar_to_metal_type.end(), "Unsupported type ", t);
  return it->second;
}

const std::string& getMetalType(const Tensor& t) {
  return getMetalType(t.scalar_type());
}

const std::string& getMetalType(const c10::Scalar& s) {
  return getMetalType(s.type());
}
static inline id<MTLBuffer> getMTLBufferStorage(const Tensor& tensor) {
  return __builtin_bit_cast(id<MTLBuffer>, tensor.storage().data());
}

static id<MTLLibrary> compileErfinvOpsLibrary(id<MTLDevice> device,
                                              const std::string& t1,
                                              const std::string& t2,
                                              const std::string& t3) {
  auto key = t1 + t2 + t3;
  static std::unordered_map<std::string, id<MTLLibrary>> libMap;
  auto it = libMap.find(key);
  if (it != libMap.end()) {
    return it->second;
  }
  NSError* error = nil;
  MTLCompileOptions* options = [[MTLCompileOptions new] autorelease];
  [options setLanguageVersion:MTLLanguageVersion2_3];
  auto rc =
      [device newLibraryWithSource:[NSString stringWithUTF8String:fmt::format(ERFINV_OPS_TEMPLATE, t1, t2, t3).c_str()]
                           options:options
                             error:&error];
  TORCH_CHECK(rc != nil && error == nil, "Failed to compile library: ", [[error localizedDescription] UTF8String]);
  libMap[key] = rc;
  return rc;
}

static id<MTLComputePipelineState> getCPLState(id<MTLDevice> device,
                                               const std::string& t1,
                                               const std::string& t2,
                                               const std::string& t3,
                                               const std::string& fname) {
  auto key = t1 + t2 + t3 + fname;
  static std::unordered_map<std::string, id<MTLComputePipelineState>> cplMap;
  auto it = cplMap.find(key);
  if (it != cplMap.end()) {
    return it->second;
  }
  NSError* error = nil;
  auto library = compileErfinvOpsLibrary(device, t1, t2, t3);
  id<MTLFunction> func = [library newFunctionWithName:[NSString stringWithUTF8String:fname.c_str()]];
  TORCH_CHECK(func != nil, "Can't get function ", fname);
  auto rc = [device newComputePipelineStateWithFunction:func error:&error];
  TORCH_CHECK(
      rc != nil && error == nil, "Failed to construct pipeline state: ", [[error localizedDescription] UTF8String]);
  cplMap[key] = rc;
  return rc;
}

TORCH_IMPL_FUNC(erfinv_out_mps)(const Tensor& self, const Tensor& output_) {
  // hanlde erfinv ops
  Tensor input = self; 
  Tensor output = output_; 
  // do type check to make sure it's a scalar bool, int16, int32, float32,
  TORCH_CHECK(input.scalar_type() != ScalarType::Double, "MPS does not support erfinv op with scalar type: Double");
  if (input.dim() == 0) {
    return;
  }
  uint32_t length = output.numel();
  if (length == 0) {
    return;
  }
  using namespace torch::mps;
  @autoreleasepool {
    id<MTLDevice> device = MTLCreateSystemDefaultDevice();
    id<MTLComputePipelineState> cplState =
        getCPLState(device, getMetalType(output), getMetalType(input), getMetalType(input), "erfinv_mps_kernel");

    // Get a reference of the MPSStream MTLCommandBuffer.
    id<MTLCommandBuffer> commandBuffer = torch::mps::get_command_buffer();
    TORCH_CHECK(commandBuffer, "Failed to retrieve command buffer reference");

    // Get a reference of the MPSStream dispatch_queue. This is used for CPU side synchronization while encoding.
    dispatch_queue_t serialQueue = torch::mps::get_dispatch_queue();
    dispatch_sync(serialQueue, ^() {
      // getMPSProfiler().beginProfileKernel(cplState, "erfinv_mps_kernel", {self});

      id<MTLComputeCommandEncoder> computeEncoder = [commandBuffer computeCommandEncoder];
      id<MTLBuffer> outBuf = getMTLBufferStorage(output);
      id<MTLBuffer> selfBuf = getMTLBufferStorage(input);

      //[computeEncoder pushDebugGroup:@"Dispatch erfinv_mps_kernel"];
      [computeEncoder setComputePipelineState:cplState];
      [computeEncoder setBuffer:getMTLBufferStorage(output) offset:0 atIndex:0];
      [computeEncoder setBuffer:getMTLBufferStorage(input) offset:0 atIndex:1];
      MTLSize gridSize = MTLSizeMake(length, 1, 1);
      uint32_t maxThreadsPerGroup = [cplState maxTotalThreadsPerThreadgroup];
      // Calculate a thread group size.
      NSUInteger threadsPerGroupSize = std::min(maxThreadsPerGroup, length);
      MTLSize threadGroupSize = MTLSizeMake(threadsPerGroupSize, 1, 1);
      // Encode the compute command.
      [computeEncoder dispatchThreads:gridSize threadsPerThreadgroup:threadGroupSize];
      [computeEncoder endEncoding];
      torch::mps::commit();

      // getMPSProfiler().endProfileKernel(cplState);
    });
  }
  torch::mps::synchronize();
}
} // namespace at::native