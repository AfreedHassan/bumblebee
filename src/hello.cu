#include <stdio.h>
#include <cuda_runtime.h>

__global__ void helloFromGpu() {
	printf("HELLO GPU WORLD!\n");
}

int cuda_main() {
	helloFromGpu<<<1,1>>>();
	return 0;
};

