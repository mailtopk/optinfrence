#include <cuda_runtime.h>
#include <iostream>

__global__ void hello_world(){
	int tid = threadIdx.x + blockIdx.x * blockDim.x;
	printf("Hello, world! Thread %d\n", tid);
}
int main(){
hello_world<<<1, 10>>>();
cudaDeviceSynchronize();
return 0;
}
