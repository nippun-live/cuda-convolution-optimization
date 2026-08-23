#include "cuda_conv/conv2d.hpp"

#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <mma.h>

#include <chrono>
#include <cstddef>
#include <sstream>
#include <stdexcept>
#include <utility>

namespace cuda_conv {
namespace {

constexpr int kThreads = 256;
constexpr int kTile = 16;
constexpr int kTensorWarps = 4;
constexpr int kConstantWeightCapacity = 16384;

__constant__ float kConstantWeights[kConstantWeightCapacity];

void check_cuda(const cudaError_t status, const char* operation) {
  if (status != cudaSuccess) {
    std::ostringstream message;
    message << operation << ": " << cudaGetErrorString(status);
    throw std::runtime_error(message.str());
  }
}

class DeviceBuffer {
 public:
  explicit DeviceBuffer(const std::size_t bytes) {
    check_cuda(cudaMalloc(&pointer_, bytes), "cudaMalloc");
  }

  ~DeviceBuffer() {
    if (pointer_ != nullptr) {
      cudaFree(pointer_);
    }
  }

  DeviceBuffer(const DeviceBuffer&) = delete;
  DeviceBuffer& operator=(const DeviceBuffer&) = delete;

  template <typename T>
  T* as() {
    return static_cast<T*>(pointer_);
  }

 private:
  void* pointer_ = nullptr;
};

class Event {
 public:
  Event() { check_cuda(cudaEventCreate(&event_), "cudaEventCreate"); }
  ~Event() { cudaEventDestroy(event_); }
  operator cudaEvent_t() const { return event_; }

 private:
  cudaEvent_t event_{};
};

std::vector<__half> convert_to_half(const std::vector<float>& values) {
  std::vector<__half> converted;
  converted.reserve(values.size());
  for (const float value : values) {
    converted.push_back(__float2half(value));
  }
  return converted;
}

std::vector<__nv_bfloat16> convert_to_bfloat16(
    const std::vector<float>& values) {
  std::vector<__nv_bfloat16> converted;
  converted.reserve(values.size());
  for (const float value : values) {
    converted.push_back(__float2bfloat16(value));
  }
  return converted;
}

__global__ void direct_kernel(float* __restrict__ output,
                              const float* __restrict__ input,
                              const float* __restrict__ weights,
                              const std::size_t output_count,
                              const int n_count,
                              const int channels,
                              const int height,
                              const int width,
                              const int maps,
                              const int kernel,
                              const int stride,
                              const int out_h,
                              const int out_w,
                              const bool use_constant_weights) {
  const std::size_t index =
      static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (index >= output_count) {
    return;
  }

  std::size_t remainder = index;
  const int out_x = static_cast<int>(remainder % out_w);
  remainder /= out_w;
  const int out_y = static_cast<int>(remainder % out_h);
  remainder /= out_h;
  const int map = static_cast<int>(remainder % maps);
  const int batch = static_cast<int>(remainder / maps);
  if (batch >= n_count) {
    return;
  }

  float accumulator = 0.0f;
  for (int channel = 0; channel < channels; ++channel) {
    for (int ky = 0; ky < kernel; ++ky) {
      for (int kx = 0; kx < kernel; ++kx) {
        const std::size_t input_index =
            ((static_cast<std::size_t>(batch) * channels + channel) * height +
             out_y * stride + ky) *
                width +
            out_x * stride + kx;
        const std::size_t weight_index =
            ((static_cast<std::size_t>(map) * channels + channel) * kernel +
             ky) *
                kernel +
            kx;
        const float weight = use_constant_weights
                                 ? kConstantWeights[weight_index]
                                 : weights[weight_index];
        accumulator += input[input_index] * weight;
      }
    }
  }
  output[index] = accumulator;
}

__global__ void tiled_gemm_kernel(float* __restrict__ output,
                                  const float* __restrict__ input,
                                  const float* __restrict__ weights,
                                  const int channels,
                                  const int height,
                                  const int width,
                                  const int maps,
                                  const int kernel,
                                  const int stride,
                                  const int out_h,
                                  const int out_w,
                                  const int output_positions) {
  __shared__ float weight_tile[kTile][kTile];
  __shared__ float input_tile[kTile][kTile];

  const int local_x = threadIdx.x;
  const int local_y = threadIdx.y;
  const int map = blockIdx.y * kTile + local_y;
  const int position = blockIdx.x * kTile + local_x;
  const int reduction_size = channels * kernel * kernel;
  float accumulator = 0.0f;

  for (int base = 0; base < reduction_size; base += kTile) {
    const int weight_k = base + local_x;
    if (map < maps && weight_k < reduction_size) {
      weight_tile[local_y][local_x] =
          weights[static_cast<std::size_t>(map) * reduction_size + weight_k];
    } else {
      weight_tile[local_y][local_x] = 0.0f;
    }

    const int input_k = base + local_y;
    if (position < output_positions && input_k < reduction_size) {
      const int spatial_size = out_h * out_w;
      const int batch = position / spatial_size;
      const int spatial = position % spatial_size;
      const int out_y = spatial / out_w;
      const int out_x = spatial % out_w;
      const int channel = input_k / (kernel * kernel);
      const int kernel_offset = input_k % (kernel * kernel);
      const int ky = kernel_offset / kernel;
      const int kx = kernel_offset % kernel;
      const std::size_t input_index =
          ((static_cast<std::size_t>(batch) * channels + channel) * height +
           out_y * stride + ky) *
              width +
          out_x * stride + kx;
      input_tile[local_y][local_x] = input[input_index];
    } else {
      input_tile[local_y][local_x] = 0.0f;
    }
    __syncthreads();

#pragma unroll
    for (int k = 0; k < kTile; ++k) {
      accumulator += weight_tile[local_y][k] * input_tile[k][local_x];
    }
    __syncthreads();
  }

  if (map < maps && position < output_positions) {
    const int spatial_size = out_h * out_w;
    const int batch = position / spatial_size;
    const int spatial = position % spatial_size;
    output[(static_cast<std::size_t>(batch) * maps + map) * spatial_size +
           spatial] = accumulator;
  }
}

template <typename Storage>
__global__ void tensor_core_kernel(
    float* __restrict__ output,
    const Storage* __restrict__ input,
    const Storage* __restrict__ weights,
    const int channels,
    const int height,
    const int width,
    const int maps,
    const int kernel,
    const int stride,
    const int out_h,
    const int out_w,
    const int output_positions) {
  using namespace nvcuda;
  __shared__ __align__(16) Storage weight_tile[kTile * kTile];
  __shared__ __align__(16) Storage
      input_tile[kTensorWarps][kTile * kTile];
  __shared__ __align__(16) float
      output_tile[kTensorWarps][kTile * kTile];

  const int warp = threadIdx.x / 32;
  const int lane = threadIdx.x % 32;
  const int map_base = blockIdx.y * kTile;
  const int position_base =
      (blockIdx.x * kTensorWarps + warp) * kTile;
  const int reduction_size = channels * kernel * kernel;

  wmma::fragment<wmma::matrix_a, kTile, kTile, kTile, Storage,
                 wmma::row_major>
      weight_fragment;
  wmma::fragment<wmma::matrix_b, kTile, kTile, kTile, Storage,
                 wmma::row_major>
      input_fragment;
  wmma::fragment<wmma::accumulator, kTile, kTile, kTile, float>
      accumulator_fragment;
  wmma::fill_fragment(accumulator_fragment, 0.0f);

  for (int base = 0; base < reduction_size; base += kTile) {
    for (int element = threadIdx.x; element < kTile * kTile;
         element += blockDim.x) {
      const int row = element / kTile;
      const int column = element % kTile;
      const int map = map_base + row;
      const int weight_k = base + column;
      weight_tile[element] =
          map < maps && weight_k < reduction_size
              ? weights[static_cast<std::size_t>(map) * reduction_size +
                        weight_k]
              : Storage{};

    }
    for (int element = lane; element < kTile * kTile; element += 32) {
      const int row = element / kTile;
      const int column = element % kTile;
      const int input_k = base + row;
      const int position = position_base + column;
      if (position < output_positions && input_k < reduction_size) {
        const int spatial_size = out_h * out_w;
        const int batch = position / spatial_size;
        const int spatial = position % spatial_size;
        const int out_y = spatial / out_w;
        const int out_x = spatial % out_w;
        const int channel = input_k / (kernel * kernel);
        const int kernel_offset = input_k % (kernel * kernel);
        const int ky = kernel_offset / kernel;
        const int kx = kernel_offset % kernel;
        const std::size_t input_index =
            ((static_cast<std::size_t>(batch) * channels + channel) * height +
             out_y * stride + ky) *
                width +
            out_x * stride + kx;
        input_tile[warp][element] = input[input_index];
      } else {
        input_tile[warp][element] = Storage{};
      }
    }
    __syncthreads();
    wmma::load_matrix_sync(weight_fragment, weight_tile, kTile);
    wmma::load_matrix_sync(input_fragment, input_tile[warp], kTile);
    wmma::mma_sync(accumulator_fragment, weight_fragment, input_fragment,
                   accumulator_fragment);
    __syncthreads();
  }

  wmma::store_matrix_sync(output_tile[warp], accumulator_fragment, kTile,
                          wmma::mem_row_major);
  __syncthreads();
  for (int element = lane; element < kTile * kTile; element += 32) {
    const int row = element / kTile;
    const int column = element % kTile;
    const int map = map_base + row;
    const int position = position_base + column;
    if (map < maps && position < output_positions) {
      const int spatial_size = out_h * out_w;
      const int batch = position / spatial_size;
      const int spatial = position % spatial_size;
      output[(static_cast<std::size_t>(batch) * maps + map) * spatial_size +
             spatial] = output_tile[warp][element];
    }
  }
}

Algorithm select_algorithm(const Conv2DShape& shape,
                           const Algorithm requested) {
  if (requested != Algorithm::Adaptive) {
    return requested;
  }
  return shape.output_elements() < 4096 ? Algorithm::Direct
                                        : Algorithm::TiledGemm;
}

void launch(float* output,
            const void* input,
            const void* weights,
            const Conv2DShape& shape,
            const Algorithm algorithm,
            const bool use_constant_weights,
            cudaStream_t stream = nullptr) {
  const int out_h = shape.output_height();
  const int out_w = shape.output_width();
  if (algorithm == Algorithm::Direct) {
    const std::size_t count = shape.output_elements();
    const unsigned int blocks =
        static_cast<unsigned int>((count + kThreads - 1) / kThreads);
    direct_kernel<<<blocks, kThreads, 0, stream>>>(
        output, static_cast<const float*>(input),
        static_cast<const float*>(weights), count, shape.batch,
        shape.in_channels, shape.height, shape.width, shape.out_channels,
        shape.kernel, shape.stride, out_h, out_w, use_constant_weights);
  } else if (algorithm == Algorithm::TiledGemm) {
    const int positions = shape.batch * out_h * out_w;
    const dim3 block(kTile, kTile);
    const dim3 grid((positions + kTile - 1) / kTile,
                    (shape.out_channels + kTile - 1) / kTile);
    tiled_gemm_kernel<<<grid, block, 0, stream>>>(
        output, static_cast<const float*>(input),
        static_cast<const float*>(weights), shape.in_channels, shape.height,
        shape.width, shape.out_channels, shape.kernel, shape.stride, out_h,
        out_w, positions);
  } else {
    const int positions = shape.batch * out_h * out_w;
    const int position_tiles = (positions + kTile - 1) / kTile;
    const dim3 grid((position_tiles + kTensorWarps - 1) / kTensorWarps,
                    (shape.out_channels + kTile - 1) / kTile);
    if (algorithm == Algorithm::TensorCoreFp16) {
      tensor_core_kernel<__half><<<grid, kTensorWarps * 32, 0, stream>>>(
          output, static_cast<const __half*>(input),
          static_cast<const __half*>(weights), shape.in_channels, shape.height,
          shape.width, shape.out_channels, shape.kernel, shape.stride, out_h,
          out_w, positions);
    } else {
      tensor_core_kernel<__nv_bfloat16>
          <<<grid, kTensorWarps * 32, 0, stream>>>(
              output, static_cast<const __nv_bfloat16*>(input),
              static_cast<const __nv_bfloat16*>(weights), shape.in_channels,
              shape.height, shape.width, shape.out_channels, shape.kernel,
              shape.stride, out_h, out_w, positions);
    }
  }
  check_cuda(cudaGetLastError(), "kernel launch");
}

float measure_end_to_end(const std::vector<float>& input,
                         const std::vector<float>& weights,
                         const Conv2DShape& shape,
                         const Algorithm algorithm) {
  const auto start = std::chrono::steady_clock::now();
  {
    const bool fp16 = algorithm == Algorithm::TensorCoreFp16;
    const bool bf16 = algorithm == Algorithm::TensorCoreBf16;
    const bool mixed_precision = fp16 || bf16;
    const std::vector<__half> half_input =
        fp16 ? convert_to_half(input) : std::vector<__half>{};
    const std::vector<__half> half_weights =
        fp16 ? convert_to_half(weights) : std::vector<__half>{};
    const std::vector<__nv_bfloat16> bfloat_input =
        bf16 ? convert_to_bfloat16(input) : std::vector<__nv_bfloat16>{};
    const std::vector<__nv_bfloat16> bfloat_weights =
        bf16 ? convert_to_bfloat16(weights) : std::vector<__nv_bfloat16>{};
    const std::size_t input_bytes =
        input.size() * (mixed_precision ? sizeof(__half) : sizeof(float));
    const std::size_t weight_bytes =
        weights.size() * (mixed_precision ? sizeof(__half) : sizeof(float));
    const void* host_input = fp16
                                 ? static_cast<const void*>(half_input.data())
                             : bf16
                                 ? static_cast<const void*>(bfloat_input.data())
                                 : static_cast<const void*>(input.data());
    const void* host_weights =
        fp16 ? static_cast<const void*>(half_weights.data())
        : bf16
            ? static_cast<const void*>(bfloat_weights.data())
            : static_cast<const void*>(weights.data());
    DeviceBuffer device_input(input_bytes);
    DeviceBuffer device_weights(weight_bytes);
    DeviceBuffer device_output(shape.output_elements() * sizeof(float));
    check_cuda(cudaMemcpy(device_input.as<void>(), host_input, input_bytes,
                          cudaMemcpyHostToDevice),
               "copy input to device");
    check_cuda(cudaMemcpy(device_weights.as<void>(), host_weights, weight_bytes,
                          cudaMemcpyHostToDevice),
               "copy weights to device");

    const bool use_constant =
        algorithm == Algorithm::Direct &&
        weights.size() <= static_cast<std::size_t>(kConstantWeightCapacity);
    if (use_constant) {
      check_cuda(cudaMemcpyToSymbol(kConstantWeights, weights.data(),
                                    weights.size() * sizeof(float)),
                 "copy weights to constant memory");
    }
    launch(device_output.as<float>(), device_input.as<void>(),
           device_weights.as<void>(), shape, algorithm, use_constant);

    std::vector<float> scratch(shape.output_elements());
    check_cuda(cudaMemcpy(scratch.data(), device_output.as<float>(),
                          scratch.size() * sizeof(float), cudaMemcpyDeviceToHost),
               "copy output to host");
  }
  const auto stop = std::chrono::steady_clock::now();
  return std::chrono::duration<float, std::milli>(stop - start).count();
}

}  // namespace

DeviceInfo current_device_info() {
  int device = 0;
  check_cuda(cudaGetDevice(&device), "cudaGetDevice");
  cudaDeviceProp properties{};
  check_cuda(cudaGetDeviceProperties(&properties, device),
             "cudaGetDeviceProperties");
  DeviceInfo info;
  info.name = properties.name;
  info.compute_major = properties.major;
  info.compute_minor = properties.minor;
  info.global_memory_bytes = properties.totalGlobalMem;
  return info;
}

RunResult run_cuda(const std::vector<float>& input,
                   const std::vector<float>& weights,
                   const Conv2DShape& shape,
                   const Algorithm requested,
                   const int warmup_iterations,
                   const int timed_iterations) {
  shape.validate();
  if (input.size() != shape.input_elements()) {
    throw std::invalid_argument("input size does not match shape");
  }
  if (weights.size() != shape.weight_elements()) {
    throw std::invalid_argument("weight size does not match shape");
  }
  if (warmup_iterations < 0 || timed_iterations <= 0) {
    throw std::invalid_argument("iteration counts are invalid");
  }

  const Algorithm executed = select_algorithm(shape, requested);
  const bool fp16 = executed == Algorithm::TensorCoreFp16;
  const bool bf16 = executed == Algorithm::TensorCoreBf16;
  const bool mixed_precision = fp16 || bf16;
  if (mixed_precision) {
    const DeviceInfo device = current_device_info();
    if (fp16 && device.compute_major < 7) {
      throw std::runtime_error("tensor-fp16 requires Tensor Core support");
    }
    if (bf16 && device.compute_major < 8) {
      throw std::runtime_error("tensor-bf16 requires Ampere or newer");
    }
  }
  const std::vector<__half> half_input =
      fp16 ? convert_to_half(input) : std::vector<__half>{};
  const std::vector<__half> half_weights =
      fp16 ? convert_to_half(weights) : std::vector<__half>{};
  const std::vector<__nv_bfloat16> bfloat_input =
      bf16 ? convert_to_bfloat16(input) : std::vector<__nv_bfloat16>{};
  const std::vector<__nv_bfloat16> bfloat_weights =
      bf16 ? convert_to_bfloat16(weights) : std::vector<__nv_bfloat16>{};
  const std::size_t input_bytes =
      input.size() * (mixed_precision ? sizeof(__half) : sizeof(float));
  const std::size_t weight_bytes =
      weights.size() * (mixed_precision ? sizeof(__half) : sizeof(float));
  const void* host_input = fp16
                               ? static_cast<const void*>(half_input.data())
                           : bf16
                               ? static_cast<const void*>(bfloat_input.data())
                               : static_cast<const void*>(input.data());
  const void* host_weights =
      fp16 ? static_cast<const void*>(half_weights.data())
      : bf16
          ? static_cast<const void*>(bfloat_weights.data())
          : static_cast<const void*>(weights.data());
  DeviceBuffer device_input(input_bytes);
  DeviceBuffer device_weights(weight_bytes);
  DeviceBuffer device_output(shape.output_elements() * sizeof(float));
  check_cuda(cudaMemcpy(device_input.as<void>(), host_input, input_bytes,
                        cudaMemcpyHostToDevice),
             "copy input to device");
  check_cuda(cudaMemcpy(device_weights.as<void>(), host_weights, weight_bytes,
                        cudaMemcpyHostToDevice),
             "copy weights to device");

  const bool use_constant =
      executed == Algorithm::Direct &&
      weights.size() <= static_cast<std::size_t>(kConstantWeightCapacity);
  if (use_constant) {
    check_cuda(cudaMemcpyToSymbol(kConstantWeights, weights.data(),
                                  weights.size() * sizeof(float)),
               "copy weights to constant memory");
  }

  for (int i = 0; i < warmup_iterations; ++i) {
    launch(device_output.as<float>(), device_input.as<void>(),
           device_weights.as<void>(), shape, executed, use_constant);
  }
  check_cuda(cudaDeviceSynchronize(), "warmup synchronization");

  Event start;
  Event stop;
  check_cuda(cudaEventRecord(start), "record start event");
  for (int i = 0; i < timed_iterations; ++i) {
    launch(device_output.as<float>(), device_input.as<void>(),
           device_weights.as<void>(), shape, executed, use_constant);
  }
  check_cuda(cudaEventRecord(stop), "record stop event");
  check_cuda(cudaEventSynchronize(stop), "synchronize stop event");
  float elapsed_ms = 0.0f;
  check_cuda(cudaEventElapsedTime(&elapsed_ms, start, stop),
             "measure kernel time");

  RunResult result;
  result.output.resize(shape.output_elements());
  check_cuda(cudaMemcpy(result.output.data(), device_output.as<float>(),
                        result.output.size() * sizeof(float),
                        cudaMemcpyDeviceToHost),
             "copy output to host");
  result.kernel_ms = elapsed_ms / static_cast<float>(timed_iterations);
  result.end_to_end_ms =
      measure_end_to_end(input, weights, shape, executed);
  result.requested = requested;
  result.executed = executed;
  return result;
}

}  // namespace cuda_conv
