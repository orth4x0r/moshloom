@echo off
rem Windows build. Same command as build.sh; see the comment there about -fno-finite-math-only.
rem Requires LLVM: winget install LLVM.LLVM
clang --target=wasm32 -O3 -ffast-math -fno-finite-math-only -msimd128 -nostdlib -ffreestanding -fvisibility=hidden -Wl,--no-entry -Wl,--export-dynamic -Wl,--export=__heap_base -Wl,-z,stack-size=1048576 -o engine.wasm engine.c
if errorlevel 1 exit /b 1
echo built engine.wasm
