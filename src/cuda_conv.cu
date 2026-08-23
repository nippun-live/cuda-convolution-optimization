#include "cuda_conv/conv2d.hpp"

#include <cuda_runtime.h>

#include <chrono>
#include <cstddef>
#include <sstream>
#include <stdexcept>
#include <utility>

namespace cuda_conv {
namespace {

constexpr int kThreads = 256;
constexpr int kTile = 16;
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

  float* get() { return static_cast<float*>(pointer_); }

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

Algorithm select_algorithm(const Conv2DShape& shape,
                           const Algorithm requested) {
  if (requested != Algorithm::Adaptive) {
    return requested;
  }
  return shape.output_elements() < 4096 ? Algorithm::Direct
                                        : Algorithm::TiledGemm;
}

void launch(float* output,
            const float* input,
            const float* weights,
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
        output, input, weights, count, shape.batch, shape.in_channels,
        shape.height, shape.width, shape.out_channels, shape.kernel,
        shape.stride, out_h, out_w, use_constant_weights);
  } else {
    const int positions = shape.batch * out_h * out_w;
    const dim3 block(kTile, kTile);
    const dim3 grid((positions + kTile - 1) / kTile,
                    (shape.out_channels + kTile - 1) / kTile);
    tiled_gemm_kernel<<<grid, block, 0, stream>>>(
        output, input, weights, shape.in_channels, shape.height, shape.width,
        shape.out_channels, shape.kernel, shape.stride, out_h, out_w,
        positions);
  }
  check_cuda(cudaGetLastError(), "kernel launch");
}

float measure_end_to_end(const std::vector<float>& input,
                         const std::vector<float>& weights,
                         const Conv2DShape& shape,
                         const Algorithm algorithm) {
  const auto start = std::chrono::steady_clock::now();
  {
    DeviceBuffer device_input(input.size() * sizeof(float));
    DeviceBuffer device_weights(weights.size() * sizeof(float));
    DeviceBuffer device_output(shape.output_elements() * sizeof(float));
    check_cuda(cudaMemcpy(device_input.get(), input.data(),
                          input.size() * sizeof(float), cudaMemcpyHostToDevice),
               "copy input to device");
    check_cuda(cudaMemcpy(device_weights.get(), weights.data(),
                          weights.size() * sizeof(float), cudaMemcpyHostToDevice),
               "copy weights to device");

    const bool use_constant =
        algorithm == Algorithm::Direct &&
        weights.size() <= static_cast<std::size_t>(kConstantWeightCapacity);
    if (use_constant) {
      check_cuda(cudaMemcpyToSymbol(kConstantWeights, weights.data(),
                                    weights.size() * sizeof(float)),
                 "copy weights to constant memory");
    }
    launch(device_output.get(), device_input.get(), device_weights.get(), shape,
           algorithm, use_constant);

    std::vector<float> scratch(shape.output_elements());
    check_cuda(cudaMemcpy(scratch.data(), device_output.get(),
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
  DeviceBuffer device_input(input.size() * sizeof(float));
  DeviceBuffer device_weights(weights.size() * sizeof(float));
  DeviceBuffer device_output(shape.output_elements() * sizeof(float));
  check_cuda(cudaMemcpy(device_input.get(), input.data(),
                        input.size() * sizeof(float), cudaMemcpyHostToDevice),
             "copy input to device");
  check_cuda(cudaMemcpy(device_weights.get(), weights.data(),
                        weights.size() * sizeof(float), cudaMemcpyHostToDevice),
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
    launch(device_output.get(), device_input.get(), device_weights.get(), shape,
           executed, use_constant);
  }
  check_cuda(cudaDeviceSynchronize(), "warmup synchronization");

  Event start;
  Event stop;
  check_cuda(cudaEventRecord(start), "record start event");
  for (int i = 0; i < timed_iterations; ++i) {
    launch(device_output.get(), device_input.get(), device_weights.get(), shape,
           executed, use_constant);
  }
  check_cuda(cudaEventRecord(stop), "record stop event");
  check_cuda(cudaEventSynchronize(stop), "synchronize stop event");
  float elapsed_ms = 0.0f;
  check_cuda(cudaEventElapsedTime(&elapsed_ms, start, stop),
             "measure kernel time");

  RunResult result;
  result.output.resize(shape.output_elements());
  check_cuda(cudaMemcpy(result.output.data(), device_output.get(),
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
