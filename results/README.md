# Results

Run:

~~~powershell
.\build\cuda-conv-bench.exe --csv results\local.csv
~~~

The benchmark prints a human-readable table and writes machine-readable
results containing:

- requested and executed algorithm;
- average kernel latency;
- one-shot end-to-end latency;
- scalar CPU latency and kernel speedup;
- effective GFLOP/s;
- maximum absolute, maximum relative, and mean absolute error.

reference-rtx3050.csv was generated from the repository's committed code on
2026-08-22 using an NVIDIA GeForce RTX 3050 Laptop GPU, CUDA 12.8, and driver
596.08. GPU boost state, thermals, driver version, and background load can
change timings, so regenerate results before making comparisons on another
machine.
