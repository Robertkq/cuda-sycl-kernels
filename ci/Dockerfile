# Build environment for cuda-sycl-kernels: Fedora 44 + CUDA toolkit +
# a prebuilt intel/llvm (DPC++) SYCL compiler with the CUDA backend.
#
# There is no GPU in here. The image can *compile* both backends, and it can
# *run* the SYCL binaries on the CPU via Intel's OpenCL CPU runtime (the SYCL
# targets get an extra spir64 device image in CI; see SYCL_EXTRA_TARGETS in
# the top-level CMakeLists.txt). CUDA binaries are build-only.
#
#   podman build -t ghcr.io/robertkq/cuda-sycl-kernels-ci:latest -f ci/Dockerfile .
#   podman run --rm -v "$PWD":/src:Z ghcr.io/robertkq/cuda-sycl-kernels-ci:latest ci/build.sh
#
# CI doesn't build this file; it runs inside the published image. After
# changing it, rebuild with the command above and push:
#
#   podman push ghcr.io/robertkq/cuda-sycl-kernels-ci:latest

FROM registry.fedoraproject.org/fedora:44

# Links the ghcr.io package to the repo, which lets the repo's workflows pull
# it with their GITHUB_TOKEN.
LABEL org.opencontainers.image.source=https://github.com/Robertkq/cuda-sycl-kernels

# CUDA major-minor as it appears in NVIDIA's package names (cuda-nvcc-13-4).
ARG CUDA_PKG_VERSION=13-4
# intel/llvm daily release tag. Pinned so a new nightly can't break CI
# silently; bump it on purpose.
ARG SYCL_RELEASE=nightly-2026-09-24

RUN dnf -y install dnf-plugins-core \
 && dnf config-manager addrepo --from-repofile=https://developer.download.nvidia.com/compute/cuda/repos/fedora44/x86_64/cuda-fedora44.repo \
 && printf '%s\n' \
      '[oneAPI]' \
      'name=Intel oneAPI repository' \
      'baseurl=https://yum.repos.intel.com/oneapi' \
      'enabled=1' \
      'gpgcheck=1' \
      'repo_gpgcheck=1' \
      'gpgkey=https://yum.repos.intel.com/intel-gpg-keys/GPG-PUB-KEY-INTEL-SW-PRODUCTS.PUB' \
      > /etc/yum.repos.d/oneAPI.repo \
 && dnf -y install --setopt=install_weak_deps=False \
      cmake ninja-build gcc-c++ git curl tar gzip which \
      cli11-devel json-devel \
      ocl-icd clinfo \
      intel-oneapi-runtime-opencl \
      cuda-nvcc-${CUDA_PKG_VERSION} \
      cuda-cudart-devel-${CUDA_PKG_VERSION} \
      cuda-crt-${CUDA_PKG_VERSION} \
 && dnf clean all

ENV PATH=/usr/local/cuda/bin:${PATH}

# Prebuilt DPC++ toolchain (same project the local intel/llvm --cuda build in
# BUILD.md comes from; the release tarballs ship the CUDA adapter too).
RUN mkdir -p /opt/sycl \
 && curl -fsSL "https://github.com/intel/llvm/releases/download/${SYCL_RELEASE}/sycl_linux.tar.gz" \
    | tar -xz -C /opt/sycl

WORKDIR /src
