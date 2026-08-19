#!/usr/bin/env bash
set -Eeuo pipefail

readonly CMAKE_MIN_VERSION=3.30
readonly PROJECT_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"

log() {
    echo "[bumblebee] $*"
}

die() {
    echo "[bumblebee] ERROR: $*" >&2
    exit 1
}

if [[ ${EUID} -eq 0 ]]; then
    SUDO=()
elif command -v sudo >/dev/null 2>&1; then
    SUDO=(sudo)
else
    die "Run as root or install sudo. Vast instances normally log in as root."
fi

command -v apt-get >/dev/null 2>&1 || die "This bootstrap supports Ubuntu/Debian CUDA images."

export DEBIAN_FRONTEND=noninteractive
log "Installing base development tools"
"${SUDO[@]}" apt-get update
"${SUDO[@]}" apt-get install -y --no-install-recommends \
    build-essential ca-certificates cmake curl git ninja-build \
    python3 python3-pip python3-venv software-properties-common tmux

install_gcc16() {
    if command -v g++-16 >/dev/null 2>&1; then
        return
    fi

    log "Installing GCC 16 for C++26 reflection"
    if "${SUDO[@]}" apt-get install -y --no-install-recommends gcc-16 g++-16; then
        return
    fi

    if [[ -r /etc/os-release ]]; then
        # shellcheck disable=SC1091
        source /etc/os-release
    fi
    [[ ${ID:-} == ubuntu ]] || die "GCC 16 is unavailable. Use an Ubuntu image or install gcc-16 and g++-16 manually."

    "${SUDO[@]}" add-apt-repository -y ppa:ubuntu-toolchain-r/test
    "${SUDO[@]}" apt-get update
    "${SUDO[@]}" apt-get install -y --no-install-recommends gcc-16 g++-16
}

install_gcc16

cmake_is_new_enough() {
    command -v cmake >/dev/null 2>&1 || return 1
    local installed
    installed="$(cmake --version | awk 'NR == 1 { print $3 }')"
    dpkg --compare-versions "${installed}" ge "${CMAKE_MIN_VERSION}"
}

if ! cmake_is_new_enough; then
    log "Installing CMake >= ${CMAKE_MIN_VERSION} in /opt/bumblebee-tools"
    "${SUDO[@]}" python3 -m venv /opt/bumblebee-tools
    "${SUDO[@]}" /opt/bumblebee-tools/bin/pip install --upgrade pip "cmake>=${CMAKE_MIN_VERSION}"
    for tool in cmake cpack ctest; do
        "${SUDO[@]}" ln -sf "/opt/bumblebee-tools/bin/${tool}" "/usr/local/bin/${tool}"
    done
    hash -r
fi

command -v nvidia-smi >/dev/null 2>&1 || die \
    "nvidia-smi is missing. Rent a Vast NVIDIA instance with GPU drivers enabled."
nvidia-smi >/dev/null || die "The NVIDIA GPU/driver is not available inside this instance."

command -v nvcc >/dev/null 2>&1 || die \
    "nvcc is missing. Recreate the instance with an NVIDIA CUDA development image (not a runtime-only image)."

# nvcc can support a different GCC range than the compiler used for .cpp files.
CUDA_HOST_CXX=""
CUDA_TEST_SOURCE="$(mktemp --suffix=.cu)"
CUDA_TEST_OBJECT="$(mktemp --suffix=.o)"
trap 'rm -f "${CUDA_TEST_SOURCE}" "${CUDA_TEST_OBJECT}"' EXIT
cat >"${CUDA_TEST_SOURCE}" <<'CUDA_TEST'
__global__ void kernel() {}
int main() { kernel<<<1, 1>>>(); }
CUDA_TEST

for candidate in g++-16 g++-15 g++-14 g++-13 g++-12 g++-11 g++; do
    command -v "${candidate}" >/dev/null 2>&1 || continue
    if nvcc -std=c++20 -ccbin "$(command -v "${candidate}")" -c \
        "${CUDA_TEST_SOURCE}" -o "${CUDA_TEST_OBJECT}" >/dev/null 2>&1; then
        CUDA_HOST_CXX="$(command -v "${candidate}")"
        break
    fi
done

if [[ -z ${CUDA_HOST_CXX} ]]; then
    log "Installing alternate CUDA host compilers"
    for version in 15 14 13 12 11; do
        if apt-cache show "g++-${version}" >/dev/null 2>&1; then
            "${SUDO[@]}" apt-get install -y --no-install-recommends "g++-${version}"
        fi
    done
    for candidate in g++-15 g++-14 g++-13 g++-12 g++-11; do
        command -v "${candidate}" >/dev/null 2>&1 || continue
        if nvcc -std=c++20 -ccbin "$(command -v "${candidate}")" -c \
            "${CUDA_TEST_SOURCE}" -o "${CUDA_TEST_OBJECT}" >/dev/null 2>&1; then
            CUDA_HOST_CXX="$(command -v "${candidate}")"
            break
        fi
    done
fi

[[ -n ${CUDA_HOST_CXX} ]] || die \
    "No installed GCC is accepted by nvcc. Use a newer CUDA development image."

GPU_ARCH="$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader | awk 'NR == 1 { gsub(/\./, ""); print }')"
[[ ${GPU_ARCH} =~ ^[0-9]+$ ]] || die "Could not determine the GPU compute capability."

log "Configuring for GPU compute capability ${GPU_ARCH}"
rm -rf "${PROJECT_ROOT}/build"
CC=gcc-16 CXX=g++-16 CUDAHOSTCXX="${CUDA_HOST_CXX}" \
    cmake -S "${PROJECT_ROOT}" -B "${PROJECT_ROOT}/build" -G Ninja \
    -DCMAKE_BUILD_TYPE=Release \
    -DCMAKE_CUDA_ARCHITECTURES="${GPU_ARCH}"

log "Building bumblebee"
cmake --build "${PROJECT_ROOT}/build" --parallel "$(nproc)"

log "Environment ready"
nvidia-smi --query-gpu=name,compute_cap,memory.total --format=csv,noheader
echo "C++ compiler: $(g++-16 --version | awk 'NR == 1')"
echo "CUDA compiler: $(nvcc --version | awk '/release/ { print; exit }')"
echo "CMake: $(cmake --version | awk 'NR == 1')"
echo "Run: ${PROJECT_ROOT}/build/bumblebee"
