#!/usr/bin/env bash
set -Eeuo pipefail

readonly PROJECT_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
readonly BUILD_DIR="${BUILD_DIR:-${PROJECT_ROOT}/build}"

cmake_args=(
    -S "${PROJECT_ROOT}"
    -B "${BUILD_DIR}"
    -DCMAKE_BUILD_TYPE="${BUILD_TYPE:-Release}"
)
if [[ ! -f "${BUILD_DIR}/CMakeCache.txt" ]]; then
    cmake_args+=(-G "${CMAKE_GENERATOR:-Ninja}")
fi

cmake "${cmake_args[@]}"
cmake --build "${BUILD_DIR}" --parallel

if (( $# > 0 )); then
    cd "${PROJECT_ROOT}"
    exec "$@"
fi
