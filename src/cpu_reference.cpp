#include "cuda_conv/conv2d.hpp"

#include <algorithm>
#include <cmath>
#include <limits>
#include <sstream>
#include <stdexcept>

namespace cuda_conv {

namespace {

std::size_t checked_product(std::initializer_list<int> dimensions) {
  std::size_t product = 1;
  for (const int dimension : dimensions) {
    if (dimension <= 0) {
      throw std::invalid_argument("all tensor dimensions must be positive");
    }
    if (product > std::numeric_limits<std::size_t>::max() /
                      static_cast<std::size_t>(dimension)) {
      throw std::overflow_error("tensor size overflows size_t");
    }
    product *= static_cast<std::size_t>(dimension);
  }
  return product;
}

}  // namespace

int Conv2DShape::output_height() const {
  return (height - kernel) / stride + 1;
}

int Conv2DShape::output_width() const {
  return (width - kernel) / stride + 1;
}

std::size_t Conv2DShape::input_elements() const {
  return checked_product({batch, in_channels, height, width});
}

std::size_t Conv2DShape::weight_elements() const {
  return checked_product({out_channels, in_channels, kernel, kernel});
}

std::size_t Conv2DShape::output_elements() const {
  validate();
  return checked_product({batch, out_channels, output_height(), output_width()});
}

void Conv2DShape::validate() const {
  checked_product(
      {batch, in_channels, height, width, out_channels, kernel, stride});
  if (kernel > height || kernel > width) {
    throw std::invalid_argument("kernel must fit inside the input");
  }
}

std::string Conv2DShape::to_string() const {
  std::ostringstream stream;
  stream << "N" << batch << "_C" << in_channels << "_H" << height << "_W"
         << width << "_M" << out_channels << "_K" << kernel << "_S"
         << stride;
  return stream.str();
}

const char* algorithm_name(const Algorithm algorithm) {
  switch (algorithm) {
    case Algorithm::Direct:
      return "direct";
    case Algorithm::TiledGemm:
      return "tiled-gemm";
    case Algorithm::Adaptive:
      return "adaptive";
  }
  return "unknown";
}

Algorithm parse_algorithm(const std::string& value) {
  if (value == "direct") {
    return Algorithm::Direct;
  }
  if (value == "tiled-gemm" || value == "gemm") {
    return Algorithm::TiledGemm;
  }
  if (value == "adaptive") {
    return Algorithm::Adaptive;
  }
  throw std::invalid_argument("unknown algorithm: " + value);
}

void conv2d_cpu(const std::vector<float>& input,
                const std::vector<float>& weights,
                std::vector<float>& output,
                const Conv2DShape& shape) {
  shape.validate();
  if (input.size() != shape.input_elements()) {
    throw std::invalid_argument("input size does not match shape");
  }
  if (weights.size() != shape.weight_elements()) {
    throw std::invalid_argument("weight size does not match shape");
  }

  const int out_h = shape.output_height();
  const int out_w = shape.output_width();
  output.assign(shape.output_elements(), 0.0f);

  for (int n = 0; n < shape.batch; ++n) {
    for (int m = 0; m < shape.out_channels; ++m) {
      for (int h = 0; h < out_h; ++h) {
        for (int w = 0; w < out_w; ++w) {
          float accumulator = 0.0f;
          for (int c = 0; c < shape.in_channels; ++c) {
            for (int p = 0; p < shape.kernel; ++p) {
              for (int q = 0; q < shape.kernel; ++q) {
                const std::size_t input_index =
                    ((static_cast<std::size_t>(n) * shape.in_channels + c) *
                         shape.height +
                     h * shape.stride + p) *
                        shape.width +
                    w * shape.stride + q;
                const std::size_t weight_index =
                    ((static_cast<std::size_t>(m) * shape.in_channels + c) *
                         shape.kernel +
                     p) *
                        shape.kernel +
                    q;
                accumulator += input[input_index] * weights[weight_index];
              }
            }
          }
          const std::size_t output_index =
              ((static_cast<std::size_t>(n) * shape.out_channels + m) * out_h +
               h) *
                  out_w +
              w;
          output[output_index] = accumulator;
        }
      }
    }
  }
}

ErrorMetrics compare_outputs(const std::vector<float>& expected,
                             const std::vector<float>& actual) {
  if (expected.size() != actual.size()) {
    throw std::invalid_argument("output vectors have different sizes");
  }
  ErrorMetrics metrics;
  double absolute_sum = 0.0;
  for (std::size_t i = 0; i < expected.size(); ++i) {
    const float absolute = std::abs(expected[i] - actual[i]);
    const float denominator = std::max(std::abs(expected[i]), 1.0e-6f);
    metrics.max_abs = std::max(metrics.max_abs, absolute);
    metrics.max_rel = std::max(metrics.max_rel, absolute / denominator);
    absolute_sum += absolute;
  }
  if (!expected.empty()) {
    metrics.mean_abs =
        static_cast<float>(absolute_sum / static_cast<double>(expected.size()));
  }
  return metrics;
}

}  // namespace cuda_conv
