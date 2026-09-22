#pragma once

#include <cuda_runtime.h>
#include <cstddef>
#include <memory>
#include <span>

namespace bmb {
class HostBuffer {
	using View = std::span<std::byte>;
	using ConstView = std::span<const std::byte>;

	private:
		std::unique_ptr<std::byte[]> data_;
		std::size_t size_;

	public:
		explicit HostBuffer(std::size_t byte_count): data_(nullptr), size_(byte_count) {
			if (size_ > 0) {
				data_ = std::make_unique_for_overwrite<std::byte[]>(size_);
			};
		}
		HostBuffer() : HostBuffer(0) {};

		// delete copy assignments and constructors
		HostBuffer& operator=(const HostBuffer& other) = delete;
		HostBuffer(const HostBuffer& other) = delete;

		HostBuffer& operator=(HostBuffer&& other) noexcept {
			this->data_ = std::move(other.data_);
			this->size_ = other.size_; other.size_ = 0;
			return *this;
		};

		HostBuffer(HostBuffer&& other) noexcept
			: data_(std::move(other.data_)), size_(other.size_) {
			other.size_ = 0;
		};

		auto data_mut() noexcept -> View {
			return std::span(data_.get(), size_);
		};

		auto data() const noexcept -> ConstView {
			const std::byte* pbytes = data_.get();
			return std::span(pbytes, size_);
		};

		auto is_empty() const noexcept -> bool { return size_ == 0; }
};
} // namespace bmb
