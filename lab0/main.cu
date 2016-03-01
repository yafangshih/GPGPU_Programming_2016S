#include <cstdio>
#include <cstdlib>
#include <cctype>
#include <iostream>
#include <cstring>
#include "SyncedMemory.h"

/**
*
* nvcc --version: V7.0.27
* compile: nvcc main.cu -std=c++11 
*
**/

#define CHECK {\
	auto e = cudaDeviceSynchronize();\
	if (e != cudaSuccess) {\
		printf("At " __FILE__ ":%d, %s\n", __LINE__, cudaGetErrorString(e));\
		abort();\
	}\
}
__global__ void ToCapital(char *input_gpu, int fsize) {
	int idx = blockIdx.x * blockDim.x + threadIdx.x;
	if('a' <= input_gpu[idx] and input_gpu[idx] <= 'z'){ //transform lower cases only
		input_gpu[idx] = input_gpu[idx] - ('a'-'A');
	}
	__syncthreads(); //sync before print

}
__device__ int CheckisText(char input){ // return 1 for english chars
	if('a'<=input and input<='z'){ return 1; }
	if('A'<=input and input<='Z'){ return 1; }
	return 0;
}

__global__ void FindnonText(char *input_gpu, int *nontxtList_gpu, int fsize){
	// find the idx of special characters
	int idx = blockIdx.x * blockDim.x + threadIdx.x;
	if(CheckisText(input_gpu[idx])){ 
		 nontxtList_gpu[idx]=fsize;
	}
	else{ // special chars
		nontxtList_gpu[idx]=idx;
	}
	__syncthreads();
}

__global__ void SwitchText(char *temptext_gpu, int fsize) {
	// swap each pairs

	// int blockRow = blockIdx.y;
	// int blockCol = blockIdx.x;
	// int row = threadIdx.y;
    int col = threadIdx.x;
	int idx = blockIdx.x * blockDim.x + threadIdx.x;

	// store the pair to swap
	__shared__ char As[2];
	// each thread move 1 data into shared memory
	As[col] = temptext_gpu[idx]; 
	__syncthreads();

	if (CheckisText(As[0]) and CheckisText(As[1])) {
		temptext_gpu[idx] = As[(col+1)%2]; // swap
	}
	__syncthreads();
}

int compare (const void * a, const void * b)
{
  return ( *(int*)a - *(int*)b );
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
	
	printf("Two transformation implemented:\n");
	printf("0) convert all text to capitals\n");
	printf("1) swap all pairs of characters in all words\n");
	printf("Please enter the # of the transformation to demo: ");
	int op = 0;
	scanf("%d", &op);

	if(!op){ // op==0, convert all text to capitals
		ToCapital<<<(fsize/32)+1, 32>>>(input_gpu, fsize);
	}
	else{ // op==1, swap all pairs of characters in all words

		// find special characters (those aren't english)
		// store their indexs (nontxtList)
		// (parallel with GPU)
		MemoryBuffer<int> nontxtList(fsize+1);
		auto nontxtList_smem = nontxtList.CreateSync(fsize);
		CHECK;
		int *nontxtList_gpu = nontxtList_smem.get_gpu_rw();
		FindnonText<<<(fsize/32)+1, 32>>>(input_gpu, nontxtList_gpu, fsize);

		// sort the indexs in nontxtList
		int *Lptr = (int*)nontxtList_smem.get_cpu_ro();
		std::qsort(Lptr, fsize, sizeof(int), compare);	
		Lptr = (int*)nontxtList_smem.get_cpu_ro();

		// the chars between each two indexs (temptext)
		// are those we want to swap
		MemoryBuffer<char> temptext(fsize+1);
		auto temptext_smem = temptext.CreateSync(fsize);
		CHECK;
		char *temptext_gpu = temptext_smem.get_gpu_rw();
		
		int len = *Lptr;
		int *nextLptr;
		char *inputptr = text_smem.get_cpu_wo();
		
		// input a word to SwitchText() each time
		// SwitchText swap each pair with shared memory of size 2
		while(*Lptr!=fsize){
				strncpy(temptext_smem.get_cpu_wo(), inputptr, len);
				temptext_smem.get_cpu_wo()[len] = '\0';
				temptext_gpu = temptext_smem.get_gpu_rw();
				SwitchText<<<(len/2)+1, 2>>>(temptext_gpu, len);

				// copy the swaped word 
				strncpy(inputptr, temptext_smem.get_cpu_ro(), len);
				inputptr = inputptr+len+1;

				nextLptr = Lptr+1;
				len = *nextLptr - *Lptr - 1;
				Lptr++;
		}
	}
	puts(text_smem.get_cpu_ro());
	return 0;
}