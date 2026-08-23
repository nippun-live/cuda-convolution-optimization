param(
    [string]$BuildDirectory
)

$ErrorActionPreference = "Stop"

$repositoryRoot = Split-Path -Parent $PSScriptRoot
if ([string]::IsNullOrWhiteSpace($BuildDirectory)) {
    $BuildDirectory = Join-Path $repositoryRoot "build"
} elseif (-not [System.IO.Path]::IsPathRooted($BuildDirectory)) {
    $BuildDirectory = Join-Path $repositoryRoot $BuildDirectory
}
$cudaCompiler = Join-Path $env:CUDA_PATH "bin\nvcc.exe"

if (-not (Test-Path -LiteralPath $cudaCompiler)) {
    throw "nvcc was not found. Install the CUDA Toolkit and set CUDA_PATH."
}

New-Item -ItemType Directory -Force -Path $BuildDirectory | Out-Null

$commonArguments = @(
    "-std=c++17",
    "-O3",
    "-Xcompiler=/O2",
    "-arch=native",
    "-I$repositoryRoot\include",
    "$repositoryRoot\src\cpu_reference.cpp",
    "$repositoryRoot\src\cuda_conv.cu"
)

& $cudaCompiler @commonArguments "$repositoryRoot\app\benchmark.cpp" -o "$BuildDirectory\cuda-conv-bench.exe"
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }

& $cudaCompiler @commonArguments "$repositoryRoot\tests\correctness.cpp" -o "$BuildDirectory\cuda-conv-correctness.exe"
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }

Write-Host "Built:"
Write-Host "  $BuildDirectory\cuda-conv-bench.exe"
Write-Host "  $BuildDirectory\cuda-conv-correctness.exe"
