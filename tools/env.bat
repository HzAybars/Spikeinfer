@echo off
REM tools\env.bat -- environment setup for building/running SNN benchmarks.
REM
REM Usage (from repo root, cmd.exe):
REM     call tools\env.bat
REM     .venv\Scripts\python.exe bench\benchmark.py ...
REM
REM What it does:
REM   1. Loads the VS2022 BuildTools x64 dev environment (MSVC 14.44) so
REM      cl.exe is on PATH -- Triton needs it to JIT-compile kernels.
REM   2. Forces CUDA_HOME/CUDA_PATH to the CUDA 12.4 toolkit, since torch
REM      2.6.0+cu124 needs nvcc 12.4 but the PATH-default nvcc is 13.3.
REM   3. Prepends the CUDA 12.4 bin dir to PATH so `nvcc` resolves to 12.4.
REM
REM Do NOT use the VS "18" BuildTools install if present -- only 2022
REM (MSVC 14.44) is known-good here.

set "VCVARS64=C:\Program Files (x86)\Microsoft Visual Studio\2022\BuildTools\VC\Auxiliary\Build\vcvars64.bat"
set "CUDA_HOME=C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.4"

if not exist "%VCVARS64%" (
    echo [tools\env.bat] ERROR: vcvars64.bat not found at "%VCVARS64%"
    echo [tools\env.bat] Expected VS2022 BuildTools ^(MSVC 14.44^) at that path. Install/repair it and retry.
    exit /b 1
)

call "%VCVARS64%" >nul 2>&1
if errorlevel 1 (
    echo [tools\env.bat] ERROR: vcvars64.bat failed to run ^(errorlevel %errorlevel%^).
    exit /b 1
)

if not exist "%CUDA_HOME%" (
    echo [tools\env.bat] ERROR: CUDA 12.4 toolkit not found at "%CUDA_HOME%"
    exit /b 1
)

set "CUDA_PATH=%CUDA_HOME%"
set "PATH=%CUDA_HOME%\bin;%PATH%"

for /f "delims=" %%A in ('where cl.exe 2^>nul') do (
    set "CL_RESOLVED=%%A"
    goto :gotcl
)
:gotcl

for /f "tokens=*" %%A in ('nvcc --version 2^>nul ^| findstr /r "release"') do set "NVCC_RESOLVED=%%A"

echo [tools\env.bat] OK  cl.exe="%CL_RESOLVED%"  nvcc: %NVCC_RESOLVED%
