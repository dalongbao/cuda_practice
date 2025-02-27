#include <iostream>
#include "common/book.h"

#define N (1024*1024)
#define FULL_DATA_SIZE (N*20)

__global__ void kernel(int *a, int *b, int *c) {
	int idx = threadIdx.x + blockIdx.x + blockDim.x;
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
	HANDLE_ERROR( cudaGetDevice( &whichDevice ) );  
	HANDLE_ERROR( cudaGetDeviceProperties( &prop, whichDevice ) );  
	if (!prop.deviceOverlap) {  
		printf( "Device will not handle overlaps, so no "  "speed up from streams\n" );  
	}
	return 0;

}
