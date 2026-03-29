@echo off
REM PolarQuant-KV llama.cpp 编译脚本 (Windows)
REM 环境: CUDA 13.2, MSVC 19.50, RTX 5060 Ti (sm_120)

echo === PolarQuant-KV llama.cpp Build ===

cd /d "%~dp0llama.cpp"

REM 配置 CMake
cmake -B build -G "Visual Studio 18 2025" ^
  -DGGML_CUDA=ON ^
  -DGGML_POLARQUANT=ON ^
  -DCMAKE_CUDA_ARCHITECTURES="120a" ^
  -DGGML_CUDA_FA=ON

if %ERRORLEVEL% NEQ 0 (
    echo CMake 配置失败
    exit /b 1
)

REM 编译
cmake --build build --config Release -j %NUMBER_OF_PROCESSORS%

if %ERRORLEVEL% NEQ 0 (
    echo 编译失败
    exit /b 1
)

echo === 编译成功 ===
echo 可执行文件: build\bin\Release\llama-cli.exe
echo.
echo 使用示例:
echo   build\bin\Release\llama-cli.exe -m models\Qwen3.5-9B.Q4_K_M.gguf --polarquant -p "Hello"
