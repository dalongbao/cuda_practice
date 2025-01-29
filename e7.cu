#include <iostream>
#include "book.h"

#define DIM 10

struct cuComplex {
	float r;
	float i;

	cuComplex(float a, float b) : r(a), i(b) {}

	float magnitude2() {return r * r + i * i}
k
	cuComplex operator*(const cuComplex& a) {
		return cuComplex(r*a.r - i*a.i, i*a.r + r*a.i);
	}

	cuComplex operator+(const cuComplex& a) {
		return cuComplex(r+a.r; i+a.i);
	}
};

int main() {
	CPUBitmap bitmap(DIM, DIM);
	unsigned char *dev_bitmap;

	dim3 grid(DIM, DIM);
	kernel<<<grid, 1>>>(dev_bitmap); 
	
	HANDLE_ERROR(cudaMemcpy(bitmap.get_ptr(), dev_bitmap, bitmap.image_size(), cudaMemcpyDeviceToHost));

	bitmap.display_and_exit();
	cudaFree(dev_bitmap);
}

void kernel(unsingene) {
	for (int y=0; y < DIM; y++) {
		for (int x=0; x < DIM; x++) {
			int offset = x + y * DIM;

			int juliaValue = julia(x, y);
			ptr[offset*4 + 0] = 255 * juliaValue;
			ptr[offset*4 + 1] = 0;
			ptr[offset*4 + 2] = 0;
			ptr[offset*4 + 3] = 255;
		}
	}
}

int julia(int x, int y) {
	const float scale = 1.5;
	float jx = scale * (float)(DIM / 2 - x)/(DIM / 2);
	float jy = scale * (float)(DIM / 2 - y)/(DIM / 2);

	cuComplex c(-0.8, 0.156);
	cuComplex a(jx, jy);

	int i;
	for (i=0; i<200; i++) {
		a = a * a + c;
		if (a.magnitude2() > 1000) {
			return 0;
		}
	}

	return 1;
}
