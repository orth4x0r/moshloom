#!/bin/sh
# -fno-finite-math-only: with -ffast-math alone, clang folds `a<b?a:b` into a call to
# libm's fminf, which a freestanding build cannot link. It only changes NaN semantics,
# which the engine never relies on.
clang --target=wasm32 -O3 -ffast-math -fno-finite-math-only -msimd128 -nostdlib -ffreestanding -fvisibility=hidden \
  -Wl,--no-entry -Wl,--export-dynamic -Wl,--export=__heap_base -Wl,-z,stack-size=1048576 -o engine.wasm engine.c
