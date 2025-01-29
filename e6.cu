#include <iostream> 
#include "book.h"

#define N 10

void c_add(int *a, int *b, int *c) {
	int tid = 0;
	while (tid < N) {
		c[tid] = a[tid] + b[tid];
		tid += 1;
	}
}

__global__ void add (int *a, int *b, int *c) {
	int tid = blockIdx.x;
	if (tid < N) {
		c[tid] = a[tid] + b[tid];
	}
}

int main() {
	int a[N], b[N], c[N];
	int *dev_a, *dev_b, *dev_c;

	HANDLE_ERROR(cudaMalloc((void**)&dev_a, N *sizeof(int)));
	HANDLE_ERROR(cudaMalloc((void**)&dev_b, N *sizeof(int)));
	HANDLE_ERROR(cudaMalloc((void**)&dev_c, N *sizeof(int)));

	for (int i = 0; i < N; i++) {
		a[i] = -i;
		b[i] = i * i;
	}

	HANDLE_ERROR(cudaMemcpy(dev_a, a, N * sizeof(int), cudaMemcpyHostToDevice));
	HANDLE_ERROR(cudaMemcpy(dev_b, b, N * sizeof(int), cudaMemcpyHostToDevice));

	// creating N copies of the code, then run them all in parallel
	// then at each copy, execute the process corresponding to that block_id
	add<<<N, 1>>>(dev_a, dev_b, dev_c); 

	HANDLE_ERROR(cudaMemcpy(c, dev_c, N * sizeof(int), cudaMemcpyHostToHost));

	for (int i = 0; i < N; i++) {
		printf( "%d + %d = %d\n", a[i], b[i], c[i] );
	}

	cudaFree(dev_a);
	cudaFree(dev_b);
	cudaFree(dev_c);

	return 0;
}
