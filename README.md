# CUDA Convolution Optimization Lab

A shape-general FP32 implementation of valid 2D convolution in NCHW layout,
built to make GPU optimization decisions measurable rather than anecdotal.
The project compares a scalar CPU oracle with direct CUDA convolution, an
implicit-im2col tiled GEMM kernel, and a transparent layer-adaptive dispatcher.

The repository grew from CUDA convolution experiments by **Nippun Sabharwal**
and was rebuilt as a standalone, reproducible systems project. It contains no
course harness, datasets, model checkpoints, or hidden evaluation code.

## What this demonstrates

- Correct indexing for arbitrary batch sizes, rectangular inputs, non-tile-
  aligned outputs, and configurable stride.
- Constant-memory filter reuse when a direct-convolution filter bank fits in
  CUDA's 64 KiB constant-memory budget.
- A 16 x 16 shared-memory GEMM formulation that generates im2col coordinates
  on demand instead of materializing the expanded matrix.
- Layer-adaptive dispatch using an explicit, inspectable heuristic.
- CPU-oracle validation, CUDA-event kernel timing, end-to-end timing, and
  deterministic benchmark generation.
- Documentation of FP16, channel-reduction, and stream experiments that were
  rejected or narrowed instead of being presented as unmeasured speedups.

## Repository layout

~~~text
include/cuda_conv/conv2d.hpp  public shape, algorithm, and result API
src/cpu_reference.cpp        scalar correctness oracle and error metrics
src/cuda_conv.cu             direct, tiled-GEMM, dispatch, and timing code
app/benchmark.cpp            reproducible benchmark CLI and CSV export
tests/correctness.cpp        shape and stride regression suite
docs/optimization-notes.md   retained designs, negative results, methodology
results/                     committed reference measurements
~~~

## Requirements

- NVIDIA GPU with CUDA support
- CUDA Toolkit 12.x or a compatible recent toolkit
- C++17 host compiler
- CMake 3.24+ (recommended) or direct nvcc

## Build

### CMake

~~~bash
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build --config Release
ctest --test-dir build -C Release --output-on-failure
~~~

Override architecture detection when cross-compiling:

~~~bash
cmake -S . -B build -DCMAKE_CUDA_ARCHITECTURES=86
~~~

### Windows without CMake

Run from a Visual Studio Developer PowerShell:

~~~powershell
.\scripts\build_windows.ps1
.\build\cuda-conv-correctness.exe
~~~

## Benchmark

~~~bash
./build/cuda-conv-bench --quick
./build/cuda-conv-bench --iterations 100 --csv results/local.csv
./build/cuda-conv-bench --algorithm tiled-gemm
~~~

The benchmark reports kernel-only latency separately from end-to-end latency.
Every timed configuration is checked against the CPU reference before it is
marked as passing.

## Reference results

Measured on an NVIDIA GeForce RTX 3050 Laptop GPU (compute capability 8.6),
CUDA 12.8, and driver 596.08. The CPU comparison is an optimized,
single-threaded scalar oracle compiled with MSVC /O2. Each row below uses the
explicit kernel named in the table rather than selecting the best result after
the fact.

| Shape | Kernel | Kernel time | Throughput | CPU speedup | Max abs. error |
|---|---:|---:|---:|---:|---:|
| N1 C1 8x8, M4 K3 S1 | direct | 0.0115 ms | 0.2 GFLOP/s | 0.2x | 1.49e-8 |
| N8 C3 32x32, M16 K3 S1 | tiled-gemm | 0.0342 ms | 181.8 GFLOP/s | 132.2x | 8.94e-8 |
| N4 C16 32x32, M32 K3 S1 | tiled-gemm | 0.1366 ms | 242.8 GFLOP/s | 197.6x | 2.98e-7 |
| N4 C32 31x29, M64 K3 S2 | tiled-gemm | 0.1349 ms | 229.6 GFLOP/s | 160.7x | 3.58e-7 |
| N2 C64 28x28, M64 K5 S1 | tiled-gemm | 0.6824 ms | 345.7 GFLOP/s | 308.0x | 1.31e-6 |

The tiny workload is intentionally retained: kernel-launch overhead makes both
GPU paths slower than the CPU oracle, and their difference is within ordinary
run-to-run noise. This prevents the larger-layer speedups from being presented
as universal. Raw measurements, including end-to-end latency and all three
dispatch modes, are in
[results/reference-rtx3050.csv](results/reference-rtx3050.csv).

## Scope

This is an educational kernel study, not a replacement for cuDNN. It currently
implements unpadded, undilated FP32 cross-correlation with square kernels.
Performance claims are limited to the committed shapes, hardware, and raw
results. See [the optimization notes](docs/optimization-notes.md) for the
rationale and the experiments that did not survive validation.

Nsight Compute source-level profiling is supported by the build, but hardware
performance counters were unavailable on the reference Windows configuration.
No peak-occupancy or peak-bandwidth claim is made without those counters.

## Authorship and license

Designed and implemented by **Nippun Sabharwal**. Released under the MIT
License.
