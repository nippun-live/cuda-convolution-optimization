#include "cuda_conv/conv2d.hpp"

#include <chrono>
#include <cmath>
#include <cstdlib>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <random>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

using cuda_conv::Algorithm;
using cuda_conv::Conv2DShape;

struct Options {
  int warmup = 5;
  int iterations = 50;
  bool quick = false;
  std::string csv_path;
  std::vector<Algorithm> algorithms{
      Algorithm::Direct, Algorithm::TiledGemm, Algorithm::Adaptive};
};

Options parse_options(const int argc, char** argv) {
  Options options;
  for (int i = 1; i < argc; ++i) {
    const std::string argument = argv[i];
    if (argument == "--quick") {
      options.quick = true;
      options.warmup = 2;
      options.iterations = 10;
    } else if (argument == "--iterations" && i + 1 < argc) {
      options.iterations = std::stoi(argv[++i]);
    } else if (argument == "--warmup" && i + 1 < argc) {
      options.warmup = std::stoi(argv[++i]);
    } else if (argument == "--algorithm" && i + 1 < argc) {
      const std::string value = argv[++i];
      if (value == "all") {
        options.algorithms = {Algorithm::Direct, Algorithm::TiledGemm,
                              Algorithm::Adaptive};
      } else {
        options.algorithms = {cuda_conv::parse_algorithm(value)};
      }
    } else if (argument == "--csv" && i + 1 < argc) {
      options.csv_path = argv[++i];
    } else if (argument == "--help") {
      std::cout
          << "Usage: cuda-conv-bench [--quick] [--warmup N] "
             "[--iterations N]\n"
             "                       [--algorithm direct|tiled-gemm|adaptive|all]\n"
             "                       [--csv results.csv]\n";
      std::exit(0);
    } else {
      throw std::invalid_argument("unknown or incomplete argument: " +
                                  argument);
    }
  }
  return options;
}

std::vector<float> random_vector(const std::size_t size,
                                 std::mt19937& generator) {
  std::uniform_real_distribution<float> distribution(-0.25f, 0.25f);
  std::vector<float> values(size);
  for (float& value : values) {
    value = distribution(generator);
  }
  return values;
}

double operation_count(const Conv2DShape& shape) {
  return 2.0 * static_cast<double>(shape.output_elements()) *
         shape.in_channels * shape.kernel * shape.kernel;
}

}  // namespace

int main(int argc, char** argv) {
  try {
    const Options options = parse_options(argc, argv);
    const auto device = cuda_conv::current_device_info();
    std::cout << "Device: " << device.name << " (sm_" << device.compute_major
              << device.compute_minor << ", " << std::fixed
              << std::setprecision(1)
              << static_cast<double>(device.global_memory_bytes) /
                     (1024.0 * 1024.0 * 1024.0)
              << " GiB)\n\n";

    std::vector<Conv2DShape> shapes{
        {1, 1, 8, 8, 4, 3, 1},
        {8, 3, 32, 32, 16, 3, 1},
        {4, 16, 32, 32, 32, 3, 1},
        {4, 32, 31, 29, 64, 3, 2},
        {2, 64, 28, 28, 64, 5, 1},
    };
    if (options.quick) {
      shapes.resize(3);
    }

    std::ofstream csv;
    if (!options.csv_path.empty()) {
      csv.open(options.csv_path);
      if (!csv) {
        throw std::runtime_error("could not open CSV output: " +
                                 options.csv_path);
      }
      csv << "device,shape,requested,executed,kernel_ms,end_to_end_ms,"
             "cpu_ms,kernel_speedup,gflops,max_abs,max_rel,mean_abs,pass\n";
    }

    std::cout << std::left << std::setw(30) << "Shape" << std::setw(13)
              << "Requested" << std::setw(13) << "Executed" << std::right
              << std::setw(11) << "Kernel ms" << std::setw(11) << "GFLOP/s"
              << std::setw(12) << "CPU x" << std::setw(12) << "Max abs"
              << std::setw(8) << "Check" << '\n';
    std::cout << std::string(110, '-') << '\n';

    std::mt19937 generator(437);
    bool all_passed = true;
    for (const Conv2DShape& shape : shapes) {
      const auto input = random_vector(shape.input_elements(), generator);
      const auto weights = random_vector(shape.weight_elements(), generator);

      std::vector<float> reference;
      cuda_conv::conv2d_cpu(input, weights, reference, shape);
      constexpr int kCpuIterations = 3;
      const auto cpu_start = std::chrono::steady_clock::now();
      for (int iteration = 0; iteration < kCpuIterations; ++iteration) {
        cuda_conv::conv2d_cpu(input, weights, reference, shape);
      }
      const auto cpu_stop = std::chrono::steady_clock::now();
      const float cpu_ms =
          std::chrono::duration<float, std::milli>(cpu_stop - cpu_start)
              .count() /
          static_cast<float>(kCpuIterations);

      for (const Algorithm requested : options.algorithms) {
        const auto result =
            cuda_conv::run_cuda(input, weights, shape, requested,
                                options.warmup, options.iterations);
        const auto error =
            cuda_conv::compare_outputs(reference, result.output);
        const bool passed = error.max_abs <= 1.0e-3f;
        all_passed = all_passed && passed;
        const double gflops =
            operation_count(shape) / (result.kernel_ms * 1.0e6);
        const double speedup = cpu_ms / result.kernel_ms;

        std::cout << std::left << std::setw(30) << shape.to_string()
                  << std::setw(13) << cuda_conv::algorithm_name(requested)
                  << std::setw(13)
                  << cuda_conv::algorithm_name(result.executed) << std::right
                  << std::fixed << std::setprecision(4) << std::setw(11)
                  << result.kernel_ms << std::setprecision(1) << std::setw(11)
                  << gflops << std::setw(12) << speedup << std::scientific
                  << std::setprecision(2) << std::setw(12) << error.max_abs
                  << std::setw(8) << (passed ? "PASS" : "FAIL") << '\n';

        if (csv) {
          csv << '"' << device.name << "\"," << shape.to_string() << ','
              << cuda_conv::algorithm_name(requested) << ','
              << cuda_conv::algorithm_name(result.executed) << ','
              << std::fixed << std::setprecision(6) << result.kernel_ms << ','
              << result.end_to_end_ms << ',' << cpu_ms << ',' << speedup << ','
              << gflops << ',' << std::scientific << error.max_abs << ','
              << error.max_rel << ',' << error.mean_abs << ','
              << (passed ? "true" : "false") << '\n';
        }
      }
    }
    return all_passed ? 0 : 2;
  } catch (const std::exception& error) {
    std::cerr << "error: " << error.what() << '\n';
    return 1;
  }
}
