# CUDA Convolution Kernels: FP32 to Tensor Cores

A ground-up implementation of NCHW `Conv2D` that progresses from a direct
FP32 kernel to shared-memory implicit GEMM and four-warp FP16/BF16 Tensor Core
execution. The project focuses on the parts hidden by high-level libraries:
indexing, data movement, tiling, precision, dispatch, numerical validation,
and reproducible GPU timing.

**Peak measured result:** 428.4 GFLOP/s on an RTX 3050 Laptop GPU, with up to
1.89x the direct CUDA baseline and 1.67x the FP32 tiled kernel on matched
workloads.

![Grouped throughput bars for FP32, FP16, and BF16 kernels alongside an accuracy-throughput plot](docs/benchmark-summary.png)

The left panel compares kernel throughput across four convolution workloads.
The right panel shows the accuracy-throughput tradeoff for the largest tested
reduction. Values are five-trial medians from the committed reference CSV.

## Kernel stack

| Implementation | Execution model | Key optimization |
|---|---|---|
| `direct` | One CUDA thread per output | Register accumulation and constant-memory filters when the weight bank fits in 64 KiB |
| `tiled-gemm` | 16 x 16 FP32 thread block | Shared-memory tiles and implicit im2col without an expanded global-memory matrix |
| `tensor-fp16` | Four WMMA warps per block | FP16 input/weight storage, shared weight reuse, and FP32 accumulation |
| `tensor-bf16` | Four WMMA warps per block | BF16 storage for wider exponent range with FP32 accumulation |
| `adaptive` | Explicit FP32 heuristic | Direct execution for small outputs and tiled GEMM for larger layers |

## Tensor Core dataflow

The mixed-precision kernels do more than cast FP32 values inside a scalar
kernel. Inputs and weights are stored as true 16-bit tensors before transfer.
Each block then:

1. Loads one 16 x 16 weight tile into shared memory.
2. Reuses that tile across four warps processing different output-position
   tiles.
3. Generates implicit-im2col coordinates on demand for each warp.
4. Executes 16 x 16 x 16 WMMA operations with FP32 accumulators.
5. Zero-pads reduction and output tails so arbitrary channel counts and
   non-aligned spatial dimensions remain valid.

FP16 and BF16 are explicit algorithms rather than hidden inside adaptive
dispatch. Choosing a faster representation should also mean consciously
choosing its numerical behavior.

## Performance

Measurements use CUDA events for kernel time and a separate wall-clock path
for allocation, conversion, transfers, execution, and output retrieval. Every
row is the median of five trials with 100 timed launches per trial; deterministic
inputs are regenerated from a fixed seed and checked against the FP32 CPU
oracle before a result is accepted.

Reference system: NVIDIA GeForce RTX 3050 Laptop GPU (compute capability 8.6),
CUDA 12.8, and driver 596.08. The correctness oracle is single-threaded FP32
code compiled with MSVC `/O2`.

| Shape | Kernel | Kernel time | Throughput | Direct speedup | Max abs. error |
|---|---:|---:|---:|---:|---:|
| N1 C1 8x8, M4 K3 S1 | direct FP32 | 0.0157 ms | 0.2 GFLOP/s | 1.00x | 1.49e-8 |
| N8 C3 32x32, M16 K3 S1 | tensor FP16 | 0.0329 ms | 189.3 GFLOP/s | 1.23x | 1.47e-4 |
| N4 C16 32x32, M32 K3 S1 | tiled FP32 | 0.1319 ms | 251.5 GFLOP/s | 1.13x | 2.98e-7 |
| N4 C16 32x32, M32 K3 S1 | tensor FP16 | 0.0829 ms | 400.0 GFLOP/s | 1.80x | 2.93e-4 |
| N4 C16 32x32, M32 K3 S1 | tensor BF16 | 0.0790 ms | 419.9 GFLOP/s | 1.89x | 2.46e-3 |
| N2 C64 28x28, M64 K5 S1 | tiled FP32 | 0.6340 ms | 372.1 GFLOP/s | 1.23x | 1.31e-6 |
| N2 C64 28x28, M64 K5 S1 | tensor FP16 | 0.5593 ms | 421.9 GFLOP/s | 1.39x | 9.69e-4 |
| N2 C64 28x28, M64 K5 S1 | tensor BF16 | 0.5507 ms | 428.4 GFLOP/s | 1.41x | 7.65e-3 |

The peak BF16 measurement was also 355.8x the optimized single-thread CPU
oracle. The tiny workload is retained to show the launch-overhead regime where
GPU execution is not beneficial. Direct CUDA remains the primary optimization
baseline; the CPU implementation is principally a correctness oracle. Full
kernel and end-to-end measurements for all five execution modes are available
in [the reference CSV](results/reference-rtx3050.csv).

## Correctness and numerical behavior

The regression suite covers:

- Arbitrary batch sizes and channel counts.
- Rectangular inputs, configurable stride, and non-tile-aligned outputs.
- FP32 direct and tiled kernels against a scalar CPU oracle.
- FP16 and BF16 Tensor Core outputs with dtype-specific tolerances.
- A wide-range BF16 case that exercises values poorly suited to FP16's exponent
  range.
- Invalid shape rejection before launching a kernel.

Across the regression suite, maximum absolute error remained below 1.2e-3 for
FP16 and 7.9e-3 for BF16. The FP32 implementations remained within 1.4e-6 of
the oracle.

## Build and run

Requirements: an NVIDIA GPU, CUDA Toolkit 12.x or a compatible recent toolkit,
a C++17 compiler, and CMake 3.24+ or direct `nvcc`. FP16 Tensor Core execution
requires compute capability 7.0+; BF16 requires Ampere (8.0)+. The FP32 paths
remain available on older CUDA GPUs.

~~~bash
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build --config Release
ctest --test-dir build -C Release --output-on-failure
~~~

When cross-compiling, set the architecture explicitly, for example
`-DCMAKE_CUDA_ARCHITECTURES=86`.

On Windows, a Visual Studio Developer PowerShell can build directly with
`nvcc`:

~~~powershell
.\scripts\build_windows.ps1
.\build\cuda-conv-correctness.exe
~~~

Use `-BuildDirectory build-local` to choose a different output directory.

## Benchmark CLI

~~~bash
./build/cuda-conv-bench --quick
./build/cuda-conv-bench --iterations 100 --trials 5 --csv results/local.csv
./build/cuda-conv-bench --algorithm tensor-fp16
./build/cuda-conv-bench --algorithm tensor-bf16
./build/cuda-conv-bench --shape 8,3,32,32,16,3,1 --algorithm all
~~~

Custom shapes use `N,C,H,W,M,K,S`; repeat `--shape` to construct a sweep.
Console output reports speedup over direct CUDA, while CSV output includes
kernel time, end-to-end time, throughput, CPU and direct speedups, and error
metrics.

Regenerate the benchmark figure with Python, Matplotlib, and NumPy:

~~~bash
python scripts/plot_results.py
~~~

## Project structure

~~~text
include/cuda_conv/conv2d.hpp  public shape, algorithm, and result API
src/cpu_reference.cpp        scalar oracle and numerical error metrics
src/cuda_conv.cu             FP32 and Tensor Core kernels, dispatch, timing
app/benchmark.cpp            repeatable benchmark CLI and CSV export
tests/correctness.cpp        shape, precision, and validation regressions
scripts/plot_results.py      figure generation from committed measurements
docs/optimization-notes.md   design rationale and rejected prototypes
results/                     raw reference measurements
~~~

## Engineering decisions

The final implementation keeps only mechanisms supported by correctness tests
and measured behavior. In-kernel FP16 casting was replaced with true 16-bit
storage and WMMA execution. Precision-changing kernels remain opt-in. Fixed
stream slicing and oversized channel-parallel reductions were not retained;
their failure modes and the conditions required for better designs are recorded
in [the optimization notes](docs/optimization-notes.md).

## Scope

The current API implements valid, unpadded, undilated NCHW cross-correlation
with square kernels in FP32, FP16, and BF16. It is a ground-up kernel engineering
project, not a claim to outperform cuDNN; production libraries add autotuning,
architecture-specific engines, fusion, and broader operator coverage. All
performance claims here are limited to the committed hardware, shapes, code,
and raw measurements.

## Author and license

Designed and implemented by **Nippun Sabharwal**. Released under the MIT
License.
