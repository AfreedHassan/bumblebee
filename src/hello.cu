#include <stdio.h>
#include <cuda_runtime.h>
#include "bumblebee/memory/host_buffer.h"

void print_vector(float* arr, size_t n) {
	printf("{");
	for (int i=0; i < n; ++i) {
		printf("%f ", arr[i]);
	};
	printf("}");
}

__global__ void helloFromGpu() {
	printf("HELLO GPU WORLD!\n");
}

__global__ void VecAdd(float* A, float* B, float* C) {
	int i = threadIdx.x;
	C[i] = A[i] + B[i];
}

int hello() {
	const cudaError_t error = cudaDeviceSynchronize();
	helloFromGpu<<<1,1>>>();
	int n = 3;
	size_t size = n * sizeof(float);

	float* arrA = (float *)malloc(size);
	float* arrB = (float *)malloc(size);
	float* arrC = (float *)malloc(size);

	for (int i=0; i < n; ++i) {
		arrA[i] = 1.0f;
		arrB[i] = 2.0f;
	}

	float* d_A = nullptr;
	float* d_B = nullptr;
	float* d_C = nullptr;
	cudaMalloc((void **)&d_A, size);
	cudaMalloc((void **)&d_B, size);
	cudaMalloc((void **)&d_C, size);

	cudaMemcpy(d_A, arrA, size, cudaMemcpyHostToDevice);
	cudaMemcpy(d_B, arrB, size, cudaMemcpyHostToDevice);

	int threadsPerBlock = 8;
	int blocksPerGrid = (n + threadsPerBlock - 1) / threadsPerBlock;

	VecAdd<<<blocksPerGrid, threadsPerBlock>>>(d_A, d_B, d_C);

	cudaMemcpy(arrC, d_C, size, cudaMemcpyDeviceToHost);
	print_vector(arrC, 3);

	cudaFree(d_A);
	cudaFree(d_B);
	cudaFree(d_C);
	free(arrA);
	free(arrB);
	free(arrC);

	if (error != cudaSuccess) {
		fprintf(stderr, "CUDA error: %s\n", cudaGetErrorString(error));
		return 1;
	}
	return 0;
};
