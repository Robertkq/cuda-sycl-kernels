#!/usr/bin/env bash
# Configure + build every kernel, then run the SYCL binaries on the CPU with
# full verification. Meant to run inside the ci/Dockerfile image (no GPU):
#
#   podman run --rm -v "$PWD":/src:Z ghcr.io/robertkq/cuda-sycl-kernels-ci:latest ci/build.sh
#
# CUDA binaries are compiled but never run -- there's no device for them.
# The numbers the SYCL runs print are CPU timings and mean nothing; the point
# is only that each kernel produces the right answer.
set -euo pipefail

BUILD_DIR=${BUILD_DIR:-build-ci}
# No GPU to detect, so pin the architectures. sm_86 is arbitrary but
# supported by current CUDA 13 toolkits; it just has to compile.
CUDA_ARCH=${CUDA_ARCH:-86}

cmake -S . -B "$BUILD_DIR" -G Ninja \
  -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_CUDA_ARCHITECTURES="$CUDA_ARCH" \
  -DSYCL_CUDA_ROOT=/opt/sycl \
  -DSYCL_CUDA_GPU_ARCH="sm_$CUDA_ARCH" \
  -DSYCL_EXTRA_TARGETS=spir64
cmake --build "$BUILD_DIR" --parallel

echo "--- built executables"
find "$BUILD_DIR" -maxdepth 1 -type f -executable -name '*-*' -printf '%f\n' | sort

if [[ "${RUN_SYCL:-1}" != 1 ]]; then
  exit 0
fi

# Force the OpenCL CPU device so the queue never tries the CUDA adapter.
export ONEAPI_DEVICE_SELECTOR=opencl:cpu
echo "--- SYCL devices"
# sycl-ls has no rpath (the project binaries do), so point it at the libs.
LD_LIBRARY_PATH=/opt/sycl/lib /opt/sycl/bin/sycl-ls

status=0
for exe in "$BUILD_DIR"/*-sycl; do
  [[ -x "$exe" ]] || continue
  echo "--- running $(basename "$exe")"
  # Small problem size: the kernels only need to be exercised, not timed.
  # 1<<16 is a perfect square so the transpose kernels get a 256x256 matrix.
  if ! "$exe" --count $((1 << 16)) --iterations 2 --warmups 1 \
       --verify Full --no-color; then
    echo "FAILED: $(basename "$exe")" >&2
    status=1
  fi
done
exit $status
