#define TORCH_ASSERT_ONLY_METHOD_OPERATORS
#include <ATen/native/DispatchStub.h>
#include <ATen/native/UnaryOps.h>
#include <ATen/native/mps/UnaryConstants.h>
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

static const char* ERFINV_OPS_TEMPLATE = R"METAL(
 #include <metal_stdlib>
 using namespace metal;

struct ErfinvConstant
{{
  float Y[7];
  float P[7][11];
  float Q[7][11];
}};

float evaluate_polynomial(constant const float *coefficients, float x, int count)
{{
  // Horner's implementation
  float result = coefficients[count - 1];
  for (int i = count - 2; i >= 0; --i)
  {{
    result = result * x + coefficients[i];
  }}
  return result;
}}


 kernel void erfinv_mps_kernel(device {0}  *output [[buffer(0)]],
                               device {1}  *input [[buffer(1)]],
                               constant ErfinvConstant& uniforms [[buffer(2)]],
                         uint index [[thread_position_in_grid]])  {{
  // Algorithm ported from BOOST
  // https://github.com/boostorg/math/blob/c56f334348d5476783fba996d604fc5c6ae980c3/include/boost/math/special_functions/detail/erf_inv.hpp

  float y = input[index];
  float y_abs = abs(y);
  if (y_abs > 1.0f)
  {{
    output[index] = NAN;
    return;
  }}
  if (y_abs == 1.0f)
  {{
    output[index] = copysign(INFINITY, y);
    return;
  }}
  if (y_abs == 0.0f)
  {{
    output[index] = copysign(0.0f, y);
    return;
  }}

  float p, q, s;
  if (y < 0)
  {{
    p = -y;
    q = 1 - p;
    s = -1;
  }}
  else
  {{
    p = y;
    q = 1 - p;
    s = 1;
  }}
  float Y, g, r, xs, x, R, result;
  if (p <= 0.5f)
  {{
    Y = uniforms.Y[0];
    g = p * (p + 10.0f);
    r = evaluate_polynomial(uniforms.P[0], p, 8) / evaluate_polynomial(uniforms.Q[0], p, 10);
    result = g * Y + g * r;
  }}
  else if (q >= 0.25f)
  {{
    Y = uniforms.Y[1];
    float g = sqrt(-2.0f * log(q));
    xs = q - 0.25;
    r = evaluate_polynomial(uniforms.P[1], xs, 9) / evaluate_polynomial(uniforms.Q[1], xs, 9);
    result = g / (Y + r);
  }}
  else
  {{
    x = sqrt(-log(q));
    if (x < 3)
    {{
      Y = uniforms.Y[2];
      xs = x - 1.125;
      R = evaluate_polynomial(uniforms.P[2], xs, 11) / evaluate_polynomial(uniforms.Q[2], xs, 8);
    }}
    else if (x < 6.0f)
    {{
      Y = uniforms.Y[3];
      xs = x - 3.0f;
      R = evaluate_polynomial(uniforms.P[3], xs, 9) / evaluate_polynomial(uniforms.Q[3], xs, 7);
    }}
    else if (x < 18.0f)
    {{
      Y = uniforms.Y[4];
      xs = x - 6.0f;
      R = evaluate_polynomial(uniforms.P[4], xs, 9) / evaluate_polynomial(uniforms.Q[4], xs, 7);
    }}
    else if (x < 44.0f)
    {{
      Y = uniforms.Y[5];
      xs = x - 18.0f;
      R = evaluate_polynomial(uniforms.P[5], xs, 8) / evaluate_polynomial(uniforms.Q[5], xs, 7);
    }}
    else
    {{
      Y = uniforms.Y[6];
      xs = x - 44.0f;
      R = evaluate_polynomial(uniforms.P[6], xs, 8) / evaluate_polynomial(uniforms.Q[6], xs, 7);
    }}
    result = Y * x + R * x;
  }}
  output[index] = s * result;
}})METAL";
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

static id<MTLLibrary> compileErfinvOpsLibrary(id<MTLDevice> device, const std::string& t1, const std::string& t2) {
  auto key = t1 + t2;
  static std::unordered_map<std::string, id<MTLLibrary>> libMap;
  auto it = libMap.find(key);
  if (it != libMap.end()) {
    return it->second;
  }
  NSError* error = nil;
  MTLCompileOptions* options = [[MTLCompileOptions new] autorelease];
  [options setLanguageVersion:MTLLanguageVersion2_3];
  auto rc =
      [device newLibraryWithSource:[NSString stringWithUTF8String:fmt::format(ERFINV_OPS_TEMPLATE, t1, t2).c_str()]
                           options:options
                             error:&error];
  TORCH_CHECK(rc != nil && error == nil, "Failed to compile library: ", [[error localizedDescription] UTF8String]);
  libMap[key] = rc;
  return rc;
}

static id<MTLComputePipelineState> getCPLState(id<MTLDevice> device,
                                               const std::string& t1,
                                               const std::string& t2,
                                               const std::string& fname) {
  auto key = t1 + t2 + fname;
  static std::unordered_map<std::string, id<MTLComputePipelineState>> cplMap;
  auto it = cplMap.find(key);
  if (it != cplMap.end()) {
    return it->second;
  }
  NSError* error = nil;
  auto library = compileErfinvOpsLibrary(device, t1, t2);
  id<MTLFunction> func = [library newFunctionWithName:[NSString stringWithUTF8String:fname.c_str()]];
  TORCH_CHECK(func != nil, "Can't get function ", fname);
  auto rc = [device newComputePipelineStateWithFunction:func error:&error];
  TORCH_CHECK(
      rc != nil && error == nil, "Failed to construct pipeline state: ", [[error localizedDescription] UTF8String]);
  cplMap[key] = rc;
  return rc;
}

TORCH_IMPL_FUNC(erfinv_out_mps)(const Tensor& self, const Tensor& output_) {
  // handle erfinv ops using metal kernel
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
        getCPLState(device, getMetalType(output), getMetalType(input), "erfinv_mps_kernel");

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
      // Set uniform buffer for the kernel polynomial constants
      struct ErfinvConstant uniforms;
      id<MTLBuffer> constantBuffer = [device newBufferWithLength:sizeof(ErfinvConstant)
                                                         options:MTLResourceStorageModePrivate];
      memcpy([constantBuffer contents], &uniforms, sizeof(ErfinvConstant));
      [computeEncoder setBuffer:constantBuffer offset:0 atIndex:2];

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