# Optimization notes

This repository keeps the optimization story explicit: each retained kernel has
a distinct memory-access or execution model, and each claim is tied to a
correctness check and measured timing.

## Retained kernels

| Kernel | Design | Intended regime |
|---|---|---|
| direct | One thread per output, register accumulation, read-only input, and constant-memory weights when the filter bank fits in 64 KiB | Small channel reductions and shallow layers |
| tiled-gemm | Convolution expressed as an implicit matrix multiplication with 16 x 16 shared-memory tiles; im2col coordinates are generated on demand | Larger channel reductions and output-map counts |
| tensor-fp16 | Four warps reuse each FP16 weight tile, generate per-warp implicit-im2col tiles, execute WMMA, and accumulate into FP32 | Tensor Core GPUs and accuracy-sensitive mixed precision |
| tensor-bf16 | The same Tensor Core dataflow with BF16 storage and FP32 accumulation | Ampere+ GPUs when exponent range matters more than mantissa precision |
| adaptive | Transparent heuristic choosing direct below 4,096 output elements and tiled GEMM otherwise | Mixed layer sequences |

The tiled kernel does not materialize an im2col matrix. Each shared-memory
input tile is generated directly from NCHW coordinates, avoiding the extra
global-memory allocation and copy of explicit im2col.

## FP16 versus BF16

Both Tensor Core paths store inputs and weights in 16 bits and accumulate in
FP32. FP16 retains more mantissa precision, while BF16 preserves FP32's wider
exponent range. The benchmark keeps them as explicit algorithms rather than
silently changing precision inside adaptive dispatch. On the reference shapes,
FP16 produced lower error; BF16 passed a dedicated wide-range regression and
was fastest on the two largest dense workloads. Neither format won every
shape because implicit-coordinate generation and tail work can dominate the
matrix multiply.

## Rejected or redesigned experiments

Earlier prototypes also explored the following:

- **Naive FP16 conversion.** Converting FP32 loads inside a scalar kernel did
  not reduce global-memory traffic and only reduced accumulator precision. It
  was replaced by the retained path: FP16 tensors are stored as 16-bit values,
  multiplied through WMMA Tensor Core fragments, and accumulated in FP32.
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
- Benchmark rows report the median kernel time across configurable trials.
- End-to-end time is measured separately and includes allocation, host/device
  transfers, one kernel launch, synchronization, and cleanup.
- Reported GFLOP/s uses 2 * N * M * H_out * W_out * C * K * K.

Results are hardware- and shape-specific. They should not be described as
peak occupancy, peak bandwidth, or general cuDNN speedups unless profiler or
cuDNN evidence is added.
