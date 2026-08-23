$ErrorActionPreference = "Stop"

$repositoryRoot = Split-Path -Parent $PSScriptRoot
$buildDirectory = Join-Path $repositoryRoot "build"
$cudaCompiler = Join-Path $env:CUDA_PATH "bin\nvcc.exe"

if (-not (Test-Path -LiteralPath $cudaCompiler)) {
    throw "nvcc was not found. Install the CUDA Toolkit and set CUDA_PATH."
}

New-Item -ItemType Directory -Force -Path $buildDirectory | Out-Null

$commonArguments = @(
    "-std=c++17",
    "-O3",
    "-Xcompiler=/O2",
    "-arch=native",
    "-I$repositoryRoot\include",
    "$repositoryRoot\src\cpu_reference.cpp",
    "$repositoryRoot\src\cuda_conv.cu"
)

& $cudaCompiler @commonArguments "$repositoryRoot\app\benchmark.cpp" -o "$buildDirectory\cuda-conv-bench.exe"
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }

& $cudaCompiler @commonArguments "$repositoryRoot\tests\correctness.cpp" -o "$buildDirectory\cuda-conv-correctness.exe"
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }

Write-Host "Built:"
Write-Host "  $buildDirectory\cuda-conv-bench.exe"
Write-Host "  $buildDirectory\cuda-conv-correctness.exe"
