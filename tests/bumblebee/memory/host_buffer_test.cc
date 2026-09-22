#include "bumblebee/memory/host_buffer.h"

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

template <typename Buffer>
void check_move_construction()
{
    if constexpr (std::is_move_constructible_v<Buffer>) {
        Buffer source{4};
        source.data_mut()[0] = std::byte{0x2a};

        Buffer destination{std::move(source)};
        expect(destination.data().size() == 4,
               "move construction preserves the allocation size");
        expect(destination.data()[0] == std::byte{0x2a},
               "move construction preserves stored bytes");
        expect(source.data().empty(),
               "move construction leaves the source empty");
    }
}

template <typename Buffer>
void check_move_assignment()
{
    if constexpr (std::is_move_assignable_v<Buffer>) {
        Buffer source{4};
        source.data_mut()[0] = std::byte{0x2a};
        Buffer destination{2};

        destination = std::move(source);
        expect(destination.data().size() == 4,
               "move assignment preserves the allocation size");
        expect(destination.data()[0] == std::byte{0x2a},
               "move assignment preserves stored bytes");
        expect(source.data().empty(),
               "move assignment leaves the source empty");
    }
}

} // namespace

int main()
{
    bmb::HostBuffer empty;
    expect(empty.data_mut().empty(), "default buffer has an empty mutable view");
    expect(empty.data().empty(), "default buffer has an empty const view");
    expect(empty.is_empty(), "default buffer reports an empty state");
    expect(empty.data_mut().data() == nullptr,
           "default buffer has a null data pointer");

    bmb::HostBuffer buffer{8};
    expect(!buffer.is_empty(), "sized buffer reports a non-empty state");
    expect(buffer.data_mut().size() == 8,
           "sized buffer exposes the requested byte count");
    expect(buffer.data_mut().data() != nullptr,
           "non-empty buffer has a data pointer");

    buffer.data_mut()[0] = std::byte{0x12};
    buffer.data_mut()[7] = std::byte{0x34};
    expect(buffer.data()[0] == std::byte{0x12},
           "const view observes a write to the first byte");
    expect(buffer.data()[7] == std::byte{0x34},
           "const view observes a write to the last byte");

    bmb::HostBuffer independent{8};
    independent.data_mut()[0] = std::byte{0x56};
    expect(buffer.data()[0] == std::byte{0x12},
           "separate buffers own separate allocations");

    expect(!std::is_copy_constructible_v<bmb::HostBuffer>,
           "HostBuffer is not copy constructible");
    expect(!std::is_copy_assignable_v<bmb::HostBuffer>,
           "HostBuffer is not copy assignable");
    expect(std::is_move_constructible_v<bmb::HostBuffer>,
           "HostBuffer is move constructible");
    expect(std::is_move_assignable_v<bmb::HostBuffer>,
           "HostBuffer is move assignable");

    check_move_construction<bmb::HostBuffer>();
    check_move_assignment<bmb::HostBuffer>();

    if (failures != 0) {
        std::cerr << failures << " HostBuffer test(s) failed\n";
        return 1;
    }

    std::cout << "All HostBuffer tests passed\n";
    return 0;
}
