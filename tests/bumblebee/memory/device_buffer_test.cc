#include "bumblebee/memory/device_buffer.h"
#include "bumblebee/memory/host_buffer.h"

#include <algorithm>
#include <cstddef>
#include <iostream>
#include <string_view>
#include <type_traits>
#include <utility>

namespace {

int failures = 0;

void expect(bool condition, std::string_view message)
{
    if (!condition) {
        std::cerr << "FAIL: " << message << '\n';
        ++failures;
    }
}

bool expect_cuda(cudaError_t status, std::string_view operation)
{
    if (status == cudaSuccess) {
        return true;
    }

    std::cerr << "FAIL: " << operation << ": " << cudaGetErrorString(status) << '\n';
    ++failures;
    return false;
}

} // namespace

int main()
{
    int device_count = 0;
    cudaError_t const device_status = cudaGetDeviceCount(&device_count);
    if (device_status != cudaSuccess || device_count == 0) {
        std::cout << "SKIP: no CUDA device available";
        if (device_status != cudaSuccess) {
            std::cout << ": " << cudaGetErrorString(device_status);
        }
        std::cout << '\n';
        return 77;
    }

    if (!expect_cuda(cudaSetDevice(0), "cudaSetDevice")) {
        return 1;
    }

    bmb::DeviceBuffer empty;
    expect(empty.is_empty(), "default buffer reports an empty state");
    expect(empty.data() == nullptr, "default buffer has a null pointer");
    expect(empty.size() == 0, "default buffer has zero bytes");
    expect(std::string_view{empty.error_state()}.empty(),
           "default buffer has no error");

    expect(!std::is_copy_constructible_v<bmb::DeviceBuffer>,
           "DeviceBuffer is not copy constructible");
    expect(!std::is_copy_assignable_v<bmb::DeviceBuffer>,
           "DeviceBuffer is not copy assignable");
    expect(std::is_move_constructible_v<bmb::DeviceBuffer>,
           "DeviceBuffer is move constructible");
    expect(std::is_move_assignable_v<bmb::DeviceBuffer>,
           "DeviceBuffer is move assignable");

    constexpr std::size_t byte_count = 16;
    bmb::HostBuffer input{byte_count};
    bmb::HostBuffer output{byte_count};
    for (std::size_t i = 0; i < byte_count; ++i) {
        input.data_mut()[i] = static_cast<std::byte>(i + 1);
        output.data_mut()[i] = std::byte{0};
    }

    bmb::DeviceBuffer device{byte_count};
    expect(!device.is_empty(), "sized device buffer reports a non-empty state");
    expect(device.data() != nullptr, "sized device buffer has a pointer");
    expect(device.size() == byte_count,
           "sized device buffer exposes the requested byte count");
    expect(std::string_view{device.error_state()}.empty(),
           "successful device allocation has no error");

    if (device.data() == nullptr) {
        std::cerr << "Device allocation failed: " << device.error_state() << '\n';
        return 1;
    }

    if (!expect_cuda(cudaMemcpy(device.data_mut(), input.data().data(), byte_count,
                                cudaMemcpyHostToDevice),
                     "host-to-device copy")) {
        return 1;
    }

    void* const original_pointer = device.data_mut();
    bmb::DeviceBuffer moved{std::move(device)};
    expect(moved.data_mut() == original_pointer,
           "move construction transfers the device allocation");
    expect(moved.size() == byte_count,
           "move construction preserves the device allocation size");
    expect(device.is_empty(), "move construction leaves the source empty");
    expect(device.data() == nullptr,
           "move construction clears the source pointer");

    bmb::DeviceBuffer destination{4};
    destination = std::move(moved);
    expect(destination.data_mut() == original_pointer,
           "move assignment transfers the device allocation");
    expect(destination.size() == byte_count,
           "move assignment preserves the device allocation size");
    expect(moved.is_empty(), "move assignment leaves the source empty");
    expect(moved.data() == nullptr,
           "move assignment clears the source pointer");

    if (!expect_cuda(cudaMemcpy(output.data_mut().data(), destination.data(), byte_count,
                                cudaMemcpyDeviceToHost),
                     "device-to-host copy")) {
        return 1;
    }

    expect(std::equal(input.data().begin(), input.data().end(), output.data().begin()),
           "host-device-host round trip preserves every byte");

    if (failures != 0) {
        std::cerr << failures << " DeviceBuffer test(s) failed\n";
        return 1;
    }

    std::cout << "All DeviceBuffer tests passed\n";
    return 0;
}
