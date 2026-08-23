# Optimization notes

This repository keeps the optimization story explicit: each retained kernel has
a distinct memory-access or execution model, and each claim is tied to a
correctness check and measured timing.

## Retained kernels

| Kernel | Design | Intended regime |
|---|---|---|
| direct | One thread per output, register accumulation, read-only input, and constant-memory weights when the filter bank fits in 64 KiB | Small channel reductions and shallow layers |
| tiled-gemm | Convolution expressed as an implicit matrix multiplication with 16 x 16 shared-memory tiles; im2col coordinates are generated on demand | Larger channel reductions and output-map counts |
| adaptive | Transparent heuristic choosing direct below 4,096 output elements and tiled GEMM otherwise | Mixed layer sequences |

The tiled kernel does not materialize an im2col matrix. Each shared-memory
input tile is generated directly from NCHW coordinates, avoiding the extra
global-memory allocation and copy of explicit im2col.

## Experiments intentionally not presented as wins

Earlier prototypes also explored the following:

- **FP16 accumulation.** Converting FP32 loads to half precision inside the
  kernel reduces accumulator precision but does not halve global-memory
  traffic. A publishable mixed-precision path would store FP16 tensors or use
  Tensor Core-compatible fragments and report its numerical error.
- **Channel-parallel tree reduction.** A three-dimensional thread block can
  expose channel parallelism, but naively multiplying a 16 x 16 spatial block
  by the channel count exceeds the hardware thread limit for common layers.
- **Many-stream batch slicing.** Concurrent launches can help when kernels are
  too small to occupy the device, but fixed 100-image slices create unsafe tail
  behavior and stream overhead can dominate. The clean implementation instead
  batches work in a single shape-general launch.
- **Asynchronous copies without overlap.** An async copy followed immediately
  by stream synchronization is operationally synchronous. This repository does
  not claim transfer/compute overlap without pinned buffers and a measured
  pipeline.

These negative results are part of the project: they distinguish mechanisms
that sound fast from mechanisms that improve a measured workload.

## Timing methodology

- Inputs and weights are deterministic pseudo-random FP32 values.
- Every GPU result is compared against the scalar CPU reference.
- The optimized single-thread CPU reference is warmed up and averaged over
  three executions.
- Warm-up launches occur before timing.
- Kernel time is averaged over repeated launches using CUDA events.
- End-to-end time is measured separately and includes allocation, host/device
  transfers, one kernel launch, synchronization, and cleanup.
- Reported GFLOP/s uses 2 * N * M * H_out * W_out * C * K * K.

Results are hardware- and shape-specific. They should not be described as
peak occupancy, peak bandwidth, or general cuDNN speedups unless profiler or
cuDNN evidence is added.
