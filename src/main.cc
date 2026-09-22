#include <print>
#include <meta>
#include "bumblebee/memory/host_buffer.h"

int hello();

enum Color { Red, Green, Blue };

auto main() -> int {
	bmb::HostBuffer buf(100);
	return hello();
};
