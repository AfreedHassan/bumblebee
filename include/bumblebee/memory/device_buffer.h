#pragma once

#include <cuda_runtime.h>
#include <cstddef>
#include <stdio.h>
#include <memory>
#include <span>

namespace bmb {
class DeviceBuffer {
	using View = void*;
	using ConstView = const void*;

	private:
		void* data_;
		std::size_t byte_count_;
		const char* err_string_ = "";

	public:
		explicit DeviceBuffer(std::size_t byte_count)
			: data_(nullptr), byte_count_(byte_count), err_string_("") {
			if (byte_count_ > 0) {
				cudaError_t err = cudaMalloc(&data_, byte_count_);
				if (err != cudaSuccess) {
					this->data_ = nullptr;
					byte_count_ = 0;
					err_string_ = cudaGetErrorString(err);
					printf("error: %s", err_string_);
				};
			};
		}
		DeviceBuffer() : DeviceBuffer(0) {};

		// delete copy assignments and constructors
		DeviceBuffer& operator=(const DeviceBuffer& other) = delete;
		DeviceBuffer(const DeviceBuffer& other) = delete;

		DeviceBuffer& operator=(DeviceBuffer&& other) noexcept {
			if (this == &other) return *this;
			cudaFree(this->data_);
			this->data_ = other.data_; 
			this->byte_count_ = other.byte_count_;
			this->err_string_ = other.err_string_; 
			other.data_ = nullptr;
			other.byte_count_ = 0;
			other.err_string_ = "";
			return *this;
		};

		DeviceBuffer(DeviceBuffer&& other) noexcept
			: data_(other.data_), byte_count_(other.byte_count_), err_string_(other.err_string_) {
			other.data_ = nullptr;
			other.byte_count_ = 0;
			other.err_string_ = "";
		};

		~DeviceBuffer() {
			cudaFree(data_);
		};

		auto data_mut() noexcept -> View {
			return data_;
		};

		auto data() const noexcept -> ConstView {
			return data_;
		};

		auto size() const noexcept -> std::size_t {
			return byte_count_;
		};

		auto is_empty() const noexcept -> bool { return byte_count_ == 0; }
		
		auto error_state() const noexcept -> const char* { 
			return this->err_string_;
		}
};
} // namespace bmb
