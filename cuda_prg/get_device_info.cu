#include <cuda_runtime.h>
#include <iostream>
int main(){
	cudaDeviceProp deviceProp;
	int dev = 0;
	cudaGetDeviceProperties(&deviceProp, dev);
	std::cout << " Device " << dev << " : "<<deviceProp.name << std::endl;
	std::cout << " CUDA Capability Major/Minor version Number " << deviceProp.major << " . " << deviceProp.minor << std::endl;
	std::cout << " Total Number of shared memory per block: " << deviceProp.sharedMemPerBlock << " bytes" <<std::endl;
	std::cout << " Maximum number of threads per block: " << deviceProp.maxThreadsPerBlock << std::endl;
	
	return 0;

}
