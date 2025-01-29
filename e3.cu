#include <iostream>
#include "book.h"

__global__ void add(int a, int b, int *c) {
	*c = a + b;
}

int main() {
	int c;
	int *dev_c;

	HANDLE_ERROR(cudaMalloc( (void**)&dev_c, sizeof(int))); // allocates memory to the pointer dev_c

	add<<<1, 1>>>(1, 2, dev_c);
	HANDLE_ERROR(cudaMemcpy(&c, dev_c, sizeof(int), cudaMemcpyDeviceToHost)); // copy contents of c to dev_c
	printf("%d\n", c);
	cudaFree(dev_c);

	return 0;
}
