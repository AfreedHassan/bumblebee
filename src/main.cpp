#include <print>
#include <meta>

enum Color { Red, Green, Blue };

template <typename E> 
requires std::is_enum_v<E> 
constexpr std::string_view enum_to_string(E val) {
	template for (constexpr auto e: std::define_static_array(std::meta::enumerators_of(^^E))) {
		if ([:e:] == val) {
			return std::meta::identifier_of(e);
		};
		return "<unknown>";
	}
};

auto main() -> int {
	std::println("Hello from the Color {}!", enum_to_string(Color::Red));
	return 0;
};
