#pragma once

#include <cstddef>
#include <string>
#include <vector>

namespace cuda_conv {

struct Conv2DShape {
  int batch = 1;
  int in_channels = 1;
  int height = 1;
  int width = 1;
  int out_channels = 1;
  int kernel = 1;
  int stride = 1;

  int output_height() const;
  int output_width() const;
  std::size_t input_elements() const;
  std::size_t weight_elements() const;
  std::size_t output_elements() const;
  void validate() const;
  std::string to_string() const;
};

enum class Algorithm {
  Direct,
  TiledGemm,
  TensorCoreFp16,
  TensorCoreBf16,
  Adaptive,
};

const char* algorithm_name(Algorithm algorithm);
Algorithm parse_algorithm(const std::string& value);

struct RunResult {
  std::vector<float> output;
  float kernel_ms = 0.0f;
  float end_to_end_ms = 0.0f;
  Algorithm requested = Algorithm::Direct;
  Algorithm executed = Algorithm::Direct;
};

struct ErrorMetrics {
  float max_abs = 0.0f;
  float max_rel = 0.0f;
  float mean_abs = 0.0f;
};

struct DeviceInfo {
  std::string name;
  int compute_major = 0;
  int compute_minor = 0;
  std::size_t global_memory_bytes = 0;
};

DeviceInfo current_device_info();

void conv2d_cpu(const std::vector<float>& input,
                const std::vector<float>& weights,
                std::vector<float>& output,
                const Conv2DShape& shape);

RunResult run_cuda(const std::vector<float>& input,
                   const std::vector<float>& weights,
                   const Conv2DShape& shape,
                   Algorithm algorithm,
                   int warmup_iterations = 5,
                   int timed_iterations = 50);

ErrorMetrics compare_outputs(const std::vector<float>& expected,
                             const std::vector<float>& actual);

}  // namespace cuda_conv
