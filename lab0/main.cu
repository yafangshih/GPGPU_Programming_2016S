#include <cstdio>
#include <cstdlib>
#include "SyncedMemory.h"

/**
* compile: nvcc main.cu -std=c++11 
**/

#define CHECK {\
	auto e = cudaDeviceSynchronize();\
	if (e != cudaSuccess) {\
		printf("At " __FILE__ ":%d, %s\n", __LINE__, cudaGetErrorString(e));\
		abort();\
	}\
}
/*
__device__ char mytoupper(char input){
	if('a' <= input and input <= 'z'){ return input-('a'-'A');}
	else{ return input;}
}*/

__global__ void ToCapital(char *input_gpu, int fsize) {
	int idx = blockIdx.x * blockDim.x + threadIdx.x;

	if('a' <= input_gpu[idx] and input_gpu[idx] <= 'z'){ //transform lower cases only
		input_gpu[idx] = input_gpu[idx] - ('a'-'A');
	}
	__syncthreads(); //sync before print

}

int main(int argc, char **argv)
{
	// init, and check
	if (argc != 2) {
		printf("Usage %s <input text file>\n", argv[0]);
		abort();
	}
	FILE *fp = fopen(argv[1], "r");
	if (not fp) {
		printf("Cannot open %s", argv[1]);
		abort();
	}
	// get file size
	fseek(fp, 0, SEEK_END);
	size_t fsize = ftell(fp);
	fseek(fp, 0, SEEK_SET);

	// read files
	MemoryBuffer<char> text(fsize+1);
	auto text_smem = text.CreateSync(fsize);
	CHECK;
	fread(text_smem.get_cpu_wo(), 1, fsize, fp);
	text_smem.get_cpu_wo()[fsize] = '\0';
	fclose(fp);

	// TODO: do your transform here
	char *input_gpu = text_smem.get_gpu_rw();
	//transform all characters to capitals
	ToCapital<<<(fsize/32)+1, 32>>>(input_gpu, fsize);
	
	puts(text_smem.get_cpu_ro());
	return 0;
}