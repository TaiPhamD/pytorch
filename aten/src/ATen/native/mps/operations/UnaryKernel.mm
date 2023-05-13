#include <ATen/native/DispatchStub.h>
#include <ATen/native/UnaryOps.h>
#include <ATen/ops/erfinv_native.h>

#import <Foundation/Foundation.h>
#import <Metal/Metal.h>
#include <torch/torch.h>

#include <fmt/format.h>

namespace at::native {
static const char* ERFINV_OPS_TEMPLATE = R"METAL(
 #include <metal_stdlib>
 using namespace metal;
 kernel void erfinv_mps_kernel(device {0}  *output [[buffer(0)]],
                               device {1}  *input [[buffer(1)]],
                         uint index [[thread_position_in_grid]])  {{

  // copysign
  // INFINITY
  // NAN
  // M_PI_F
  // M_2_SQRTPI_F
  constexpr float M_PI = 3.14159265358979323846264338327950288f;
  constexpr float A = 0.147f;
  const float x_squared = input[index] * input[index];
  const float log_term = log(1.0f - x_squared);
  const float common_term = 2.0f / (M_PI * A) + log_term * 0.5f;

  float term = sqrt(sqrt(common_term * common_term - log_term / A) - common_term);

  float sign = input[index] > 0.0f ? 1.0f : -1.0f;
  output[index] = sign * term;
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