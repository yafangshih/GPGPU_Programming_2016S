#include <cstdio>
#include <cstdlib>
#include <cctype>
#include <iostream>
#include "SyncedMemory.h"

#define CHECK {\
	auto e = cudaDeviceSynchronize();\
	if (e != cudaSuccess) {\
		printf("At " __FILE__ ":%d, %s\n", __LINE__, cudaGetErrorString(e));\
		abort();\
	}\
}
__device__ char mytoupper(char input){
	if('a' <= input and input <= 'z'){ return input-('a'-'A');}
	else{ return input;}
}

__global__ void ToCapital(char *input_gpu, int fsize) {
	int idx = blockIdx.x * blockDim.x + threadIdx.x;
	if (idx < fsize and input_gpu[idx] != '\n') {
		input_gpu[idx] = mytoupper(input_gpu[idx]);
	}
}

__global__ void SwitchText(char *input_gpu, int fsize) {
	//int blockRow = blockIdx.y;
    int blockCol = blockIdx.x;

   // int row = threadIdx.y;
    int col = threadIdx.x;

	int idx = blockIdx.x * blockDim.x + threadIdx.x;

//	Matrix Asub = GetSubMatrix(A, blockRow, m);
	__shared__ char As[2];
	As[col] = input_gpu[idx];

	if (idx < fsize and As[0]!='\n' and As[1]!='\n' and As[0]!=' ' and As[1]!=' ') {
		input_gpu[idx] = As[(col+1)%2];
	}
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
	// An example: transform the first 64 characters to '!'
	// Don't transform over the tail
	// And don't transform the line breaks
	printf("Two transformation function implemented:\n");
	printf("0) convert all characters to capitals\n");
	printf("1) switch all pairs of characters\n");
	printf("type the number to choose the function to demo: ");
	int op = 0;
	scanf("%d", &op);

	if(op){
		SwitchText<<<(fsize/2)+1, 2>>>(input_gpu, fsize);
	}
	else{	
		ToCapital<<<(fsize/32)+1, 32>>>(input_gpu, fsize);
	}
	

	puts(text_smem.get_cpu_ro());
	return 0;
}