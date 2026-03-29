@echo off
REM PolarQuant-KV one-click installer for llama.cpp (Windows)
REM Usage: install.bat
REM Requires: CUDA Toolkit, Visual Studio, CMake, Ninja

echo === PolarQuant-KV Installer (Windows) ===

set SCRIPT_DIR=%~dp0
set PATCH_FILE=%SCRIPT_DIR%polarquant-kv.patch
set LLAMA_DIR=%SCRIPT_DIR%llama.cpp

REM Step 1: Clone
if not exist "%LLAMA_DIR%" (
    echo Cloning llama.cpp...
    git clone --depth 1 https://github.com/ggml-org/llama.cpp "%LLAMA_DIR%"
)

REM Step 2: Apply patch
echo Applying PolarQuant patch...
cd /d "%LLAMA_DIR%"
git checkout -- . 2>nul
git apply "%PATCH_FILE%"

REM Step 3: Build
echo Building...
call "%VS_PATH%\VC\Auxiliary\Build\vcvars64.bat" >nul 2>&1
cmake -B build -G Ninja -DGGML_CUDA=ON -DGGML_POLARQUANT=ON -DCMAKE_BUILD_TYPE=Release
cmake --build build -j %NUMBER_OF_PROCESSORS%

echo.
echo === Build complete ===
echo Binary: %LLAMA_DIR%\build\bin\llama-cli.exe
echo.
echo Usage:
echo   %LLAMA_DIR%\build\bin\llama-cli.exe -m MODEL.gguf --polarquant -p "Hello"
