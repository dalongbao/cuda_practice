#include <iostream>
#include "common/book.h"

#define N (1024 * 1024)
#define FULL_DATA_SIZE (N*20)

// overlaps mean executing cuda kernels while perfoming a copy between device and host memory
// point of streams is to enable overlap of computation and data transfer

__global__ void kernel(int *a, int *b, int *c) {
	int idx = threadIdx.x + blockIdx.x * blockDim.x;

	if (idx < N) {
		int idx1 = (idx + 1) % 256;
		int idx2 = (idx + 2) % 256;

		float as = (a[idx] + a[idx1] + a[idx2]) / 3.0f;
		float bs = (b[idx] + b[idx1] + b[idx2]) / 3.0f;

		c[idx] = (as + bs) / 2;
	}
}

int main() {
	cudaDeviceProp prop;
	int whichDevice;
	HANDLE_ERROR(cudaGetDevice(&whichDevice));
	HANDLE_ERROR(cudaGetDeviceProperties(&prop, whichDevice));
	if (!prop.deviceOverlap) {
		printf("no overlaps means speedup from streams are gone :(");
		return 0;
	}

	cudaEvent_t start, stop;
	float elapsedTime;

	HANDLE_ERROR(cudaEventCreate(&start));
	HANDLE_ERROR(cudaEventCreate(&stop));
	HANDLE_ERROR(cudaEventRecord(start, 0));

	cudaStream_t stream;
	HANDLE_ERROR(cudaStreamCreate(&stream)); // these always take the address because it allows cuda to directly operate on the memory location instead of taking a copy of the variable

	int *host_a, *host_b, *host_c;
	int *dev_a, *dev_b, *dev_c;

	HANDLE_ERROR(cudaMalloc((void**)&dev_a, N*sizeof(int)));
	HANDLE_ERROR(cudaMalloc((void**)&dev_b, N*sizeof(int)));
	HANDLE_ERROR(cudaMalloc((void**)&dev_c, N*sizeof(int)));
	
	HANDLE_ERROR(cudaHostAlloc((void**)&host_a, FULL_DATA_SIZE * sizeof(int), cudaHostAllocDefault)); // PINNED MEMORY!
	HANDLE_ERROR(cudaHostAlloc((void**)&host_b, FULL_DATA_SIZE * sizeof(int), cudaHostAllocDefault)); // PINNING DOWN THE MEMORY??? freaky 
	HANDLE_ERROR(cudaHostAlloc((void**)&host_c, FULL_DATA_SIZE * sizeof(int), cudaHostAllocDefault));

	for (int i=0; i < FULL_DATA_SIZE; i++) {
		host_a[i] = rand();
		host_b[i] = rand();
	}

	for (int i = 0; i < FULL_DATA_SIZE; i += N) { // copy memory to device
		HANDLE_ERROR(cudaMemcpyAsync(dev_a, host_a+i, N*sizeof(int), cudaMemcpyHostToDevice, stream));
		HANDLE_ERROR(cudaMemcpyAsync(dev_b, host_b+i, N*sizeof(int), cudaMemcpyHostToDevice, stream));
		kernel<<<N/256, 256, 0, stream>>>(dev_a, dev_b, dev_c);
		HANDLE_ERROR(cudaMemcpyAsync(host_c + i, dev_c, N*sizeof(int), cudaMemcpyDeviceToHost, stream));
	}
		
	HANDLE_ERROR(cudaStreamSynchronize(stream));
	HANDLE_ERROR(cudaEventRecord(stop, 0));
	HANDLE_ERROR(cudaEventSynchronize(stop));
	HANDLE_ERROR(cudaEventElapsedTime(&elapsedTime, start, stop));
	printf("time taken %3.1f ms\n", elapsedTime);

	HANDLE_ERROR(cudaFreeHost(host_a));
	HANDLE_ERROR(cudaFreeHost(host_b));
	HANDLE_ERROR(cudaFreeHost(host_c));

	HANDLE_ERROR(cudaFree(dev_a));
	HANDLE_ERROR(cudaFree(dev_b));
	HANDLE_ERROR(cudaFree(dev_c));
	HANDLE_ERROR(cudaStreamDestroy(stream));
	return 0;
}
