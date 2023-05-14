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

static const char* ERFINV_OPS_TEMPLATE = R"METAL(
 #include <metal_stdlib>
 using namespace metal;

struct UniformsErfinv
{{
  float Y1;
  float P1[8];
  float Q1[10];
  float Y2;
  float P2[9];
  float Q2[9];
  float Y3;
  float P3[11];
  float Q3[8];
  float Y4;
  float P4[9];
  float Q4[7];
  float Y5;
  float P5[9];
  float Q5[7];
  float Y6;
  float P6[8];
  float Q6[7];
  float Y7;
  float P7[8];
  float Q7[7];
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
                               constant UniformsErfinv& uniforms [[buffer(2)]],
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
    Y = uniforms.Y1;
    g = p * (p + 10.0f);
    r = evaluate_polynomial(uniforms.P1, p, 8) / evaluate_polynomial(uniforms.Q1, p, 10);
    result = g * Y + g * r;
  }}
  else if (q >= 0.25f)
  {{
    Y = uniforms.Y2;
    float g = sqrt(-2.0f * log(q));
    xs = q - 0.25;
    r = evaluate_polynomial(uniforms.P2, xs, 9) / evaluate_polynomial(uniforms.Q2, xs, 9);
    result = g / (Y + r);
  }}
  else
  {{
    x = sqrt(-log(q));
    if (x < 3)
    {{
      Y = uniforms.Y3;
      xs = x - 1.125;
      R = evaluate_polynomial(uniforms.P3, xs, 11) / evaluate_polynomial(uniforms.Q3, xs, 8);
    }}
    else if (x < 6.0f)
    {{
      Y = uniforms.Y4;
      xs = x - 3.0f;
      R = evaluate_polynomial(uniforms.P4, xs, 9) / evaluate_polynomial(uniforms.Q4, xs, 7);
    }}
    else if (x < 18.0f)
    {{
      Y = uniforms.Y5;
      xs = x - 6.0f;
      R = evaluate_polynomial(uniforms.P5, xs, 9) / evaluate_polynomial(uniforms.Q5, xs, 7);
    }}
    else if (x < 44.0f)
    {{
      Y = uniforms.Y6;
      xs = x - 18.0f;
      R = evaluate_polynomial(uniforms.P6, xs, 8) / evaluate_polynomial(uniforms.Q6, xs, 7);
    }}
    else
    {{
      Y = uniforms.Y7;
      xs = x - 44.0f;
      R = evaluate_polynomial(uniforms.P7, xs, 8) / evaluate_polynomial(uniforms.Q7, xs, 7);
    }}
    result = Y * x + R * x;
  }}
  output[index] = s * result;
}})METAL";

struct UniformsErfinv {
  float Y1 = 0.0891314744949340820313f;
  float P1[8] = {-0.000508781949658280665617,
                 -0.00836874819741736770379,
                 0.0334806625409744615033,
                 -0.0126926147662974029034,
                 -0.0365637971411762664006,
                 0.0219878681111168899165,
                 0.00822687874676915743155,
                 -0.00538772965071242932965};
  float Q1[10] = {1.0,
                  -0.970005043303290640362,
                  -1.56574558234175846809,
                  1.56221558398423026363,
                  0.662328840472002992063,
                  -0.71228902341542847553,
                  -0.0527396382340099713954,
                  0.0795283687341571680018,
                  -0.00233393759374190016776,
                  0.000886216390456424707504};

  float Y2 = 2.249481201171875f;
  float P2[9] = {-0.202433508355938759655,
                 0.105264680699391713268,
                 8.37050328343119927838,
                 17.6447298408374015486,
                 -18.8510648058714251895,
                 -44.6382324441786960818,
                 17.445385985570866523,
                 21.1294655448340526258,
                 -3.67192254707729348546};
  float Q2[9] = {1.0,
                 6.24264124854247537712,
                 3.9713437953343869095,
                 -28.6608180499800029974,
                 -20.1432634680485188801,
                 48.5609213108739935468,
                 10.8268667355460159008,
                 -22.6436933413139721736,
                 1.72114765761200282724};
  float Y3 = 0.807220458984375f;
  float P3[11] = {-0.131102781679951906451,
                  -0.163794047193317060787,
                  0.117030156341995252019,
                  0.387079738972604337464,
                  0.337785538912035898924,
                  0.142869534408157156766,
                  0.0290157910005329060432,
                  0.00214558995388805277169,
                  -0.679465575181126350155e-6,
                  0.285225331782217055858e-7,
                  -0.681149956853776992068e-9};
  float Q3[8] = {1.0,
                 3.46625407242567245975,
                 5.38168345707006855425,
                 4.77846592945843778382,
                 2.59301921623620271374,
                 0.848854343457902036425,
                 0.152264338295331783612,
                 0.01105924229346489121};
  float Y4 = 0.93995571136474609375f;
  float P4[9] = {-0.0350353787183177984712,
                 -0.00222426529213447927281,
                 0.0185573306514231072324,
                 0.00950804701325919603619,
                 0.00187123492819559223345,
                 0.000157544617424960554631,
                 0.460469890584317994083e-5,
                 -0.230404776911882601748e-9,
                 0.266339227425782031962e-11};
  float Q4[7] = {1.0,
                 1.3653349817554063097,
                 0.762059164553623404043,
                 0.220091105764131249824,
                 0.0341589143670947727934,
                 0.00263861676657015992959,
                 0.764675292302794483503e-4};
  float Y5 = 0.98362827301025390625f;
  float P5[9] = {-0.0167431005076633737133,
                 -0.00112951438745580278863,
                 0.00105628862152492910091,
                 0.000209386317487588078668,
                 0.149624783758342370182e-4,
                 0.449696789927706453732e-6,
                 0.462596163522878599135e-8,
                 -0.281128735628831791805e-13,
                 0.99055709973310326855e-16};
  float Q5[7] = {1.0,
                 0.591429344886417493481,
                 0.138151865749083321638,
                 0.0160746087093676504695,
                 0.000964011807005165528527,
                 0.275335474764726041141e-4,
                 0.282243172016108031869e-6};
  float Y6 = 0.99714565277099609375f;
  float P6[8] = {-0.0024978212791898131227,
                 -0.779190719229053954292e-5,
                 0.254723037413027451751e-4,
                 0.162397777342510920873e-5,
                 0.396341011304801168516e-7,
                 0.411632831190944208473e-9,
                 0.145596286718675035587e-11,
                 -0.116765012397184275695e-17};
  float Q6[7] = {1.0,
                 0.207123112214422517181,
                 0.0169410838120975906478,
                 0.000690538265622684595676,
                 0.145007359818232637924e-4,
                 0.144437756628144157666e-6,
                 0.509761276599778486139e-9};
  float Y7 = 0.99941349029541015625f;
  float P7[8] = {-0.000539042911019078575891,
                 -0.28398759004727721098e-6,
                 0.899465114892291446442e-6,
                 0.229345859265920864296e-7,
                 0.225561444863500149219e-9,
                 0.947846627503022684216e-12,
                 0.135880130108924861008e-14,
                 -0.348890393399948882918e-21};
  float Q7[7] = {1.0,
                 0.0845746234001899436914,
                 0.00282092984726264681981,
                 0.468292921940894236786e-4,
                 0.399968812193862100054e-6,
                 0.161809290887904476097e-8,
                 0.231558608310259605225e-11};
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
      struct UniformsErfinv uniforms;
      id<MTLBuffer> constantBuffer = [device newBufferWithLength:sizeof(UniformsErfinv)
                                                         options:MTLResourceStorageModePrivate];
      memcpy([constantBuffer contents], &uniforms, sizeof(UniformsErfinv));
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