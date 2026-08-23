#include "cuda_conv/conv2d.hpp"

#include <iostream>
#include <random>
#include <stdexcept>
#include <vector>

namespace {

std::vector<float> random_vector(const std::size_t size,
                                 std::mt19937& generator) {
  std::uniform_real_distribution<float> distribution(-0.5f, 0.5f);
  std::vector<float> values(size);
  for (float& value : values) {
    value = distribution(generator);
  }
  return values;
}

}  // namespace

int main() {
  try {
    const std::vector<cuda_conv::Conv2DShape> shapes{
        {1, 1, 7, 7, 1, 3, 1},
        {2, 3, 11, 13, 5, 3, 2},
        {1, 7, 17, 15, 19, 5, 1},
        {3, 16, 12, 10, 17, 1, 2},
    };
    const std::vector<cuda_conv::Algorithm> algorithms{
        cuda_conv::Algorithm::Direct,
        cuda_conv::Algorithm::TiledGemm,
        cuda_conv::Algorithm::Adaptive,
    };

    std::mt19937 generator(437);
    bool passed = true;
    for (const auto& shape : shapes) {
      const auto input = random_vector(shape.input_elements(), generator);
      const auto weights = random_vector(shape.weight_elements(), generator);
      std::vector<float> reference;
      cuda_conv::conv2d_cpu(input, weights, reference, shape);

      for (const auto algorithm : algorithms) {
        const auto result =
            cuda_conv::run_cuda(input, weights, shape, algorithm, 1, 2);
        const auto error =
            cuda_conv::compare_outputs(reference, result.output);
        const bool case_passed = error.max_abs <= 1.0e-3f;
        std::cout << shape.to_string() << " "
                  << cuda_conv::algorithm_name(algorithm) << " -> "
                  << cuda_conv::algorithm_name(result.executed)
                  << " max_abs=" << error.max_abs << " "
                  << (case_passed ? "PASS" : "FAIL") << '\n';
        passed = passed && case_passed;
      }
    }

    bool rejected_invalid_shape = false;
    try {
      cuda_conv::Conv2DShape invalid{1, 3, 5, 5, 8, 7, 1};
      invalid.validate();
    } catch (const std::invalid_argument&) {
      rejected_invalid_shape = true;
    }
    std::cout << "invalid shape validation "
              << (rejected_invalid_shape ? "PASS" : "FAIL") << '\n';
    passed = passed && rejected_invalid_shape;
    return passed ? 0 : 2;
  } catch (const std::exception& error) {
    std::cerr << "error: " << error.what() << '\n';
    return 1;
  }
}
