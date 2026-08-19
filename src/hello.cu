#include <stdio.h>
#include <cuda_runtime.h>

__global__ void helloFromGpu() {
	printf("HELLO GPU WORLD!\n");
}

int cuda_main() {
	helloFromGpu<<<1,1>>>();
	const cudaError_t error = cudaDeviceSynchronize();
	if (error != cudaSuccess) {
		fprintf(stderr, "CUDA error: %s\n", cudaGetErrorString(error));
		return 1;
	}
	return 0;
};
