#include "counting.h"
#include <cstdio>
#include <cassert>
#include <thrust/scan.h>
#include <thrust/transform.h>
#include <thrust/functional.h>
#include <thrust/device_ptr.h>
#include <thrust/execution_policy.h>

#include "SyncedMemory.h"

/*
nvcc -std=c++11 -arch=sm_30 -O2 -c counting.cu -o counting.o
nvcc -std=c++11 -arch=sm_30 -O2 main.cu counting.o -o $(EXE)
*/

__device__ __host__ int CeilDiv(int a, int b) { return (a-1)/b + 1; }
__device__ __host__ int CeilAlign(int a, int b) { return CeilDiv(a, b) * b; }

__device__ int CheckisText(char input){ // return 1 for english chars
	if('a'<=input and input<='z'){ return 1; }
	if('A'<=input and input<='Z'){ return 1; }
	return 0;
}

__global__ void FindnonText(const char *input_gpu, int *nontxtList_gpu, int fsize){
	// find the idx of special characters
	int idx = blockIdx.x * blockDim.x + threadIdx.x;
	if(CheckisText(input_gpu[idx])){ // english words
		 nontxtList_gpu[idx]=1;
	}
	else{ // special chars
		nontxtList_gpu[idx]=0;
	}

	__syncthreads();
}

__device__ int power2(int pow){
	int ans = 1;
	for(int i=0; i<pow; i++){
		ans = ans * 2;
	}
	return ans;
}

__global__ void indexSum(int *indexList, int len, int donetxt){
	int idx = blockIdx.x * blockDim.x + threadIdx.x;

	int val = indexList[donetxt + idx]; // range[threadIdx.x];
	for(int i=0; i<9; i++){
		if(((int)threadIdx.x - power2(i)) >= 0){
			val = val + indexList[donetxt + idx - power2(i)];
			__syncthreads();
			indexList[donetxt + idx] = val;
		}
	}
}

void CountPosition(const char *text, int *pos, int text_size)
{
	/*
		SyncedMemory<char> text_sync(text.data(), text_gpu, n);
		text_sync.get_cpu_wo();
		int *pos_yours_gpu = pos_yours_sync.get_gpu_wo();
		CountPosition(text_sync.get_gpu_ro(), pos_yours_gpu, n);
	*/

	FindnonText<<<(text_size/32)+1, 32>>>(text, pos, text_size);
	int cpuList[text_size];
	cudaMemcpy(cpuList, pos, text_size*sizeof(int), cudaMemcpyDeviceToHost);

	int donetxt = 0, len = 0;
	int *startptr = cpuList;
	int *endptr = startptr;
	while(donetxt < text_size){

		len = 0;
		while(*startptr != 1){startptr++; donetxt++;}
		endptr = startptr;
		while(*endptr == 1){endptr++; len++;}

		indexSum<<<1, len>>>(pos, len, donetxt);
		startptr = endptr;
		donetxt = donetxt + len;
	}
/**
	int count=0;
	for(int i=0;i<text_size;i++){
		if(cpuList[i]==0){
			count=0;
		}
		else{
			count++;
			cpuList[i]=count;
		}
*/


}

int ExtractHead(const int *pos, int *head, int text_size)
{
	int *buffer;
	int nhead;
	cudaMalloc(&buffer, sizeof(int)*text_size*2); // this is enough
	thrust::device_ptr<const int> pos_d(pos);
	thrust::device_ptr<int> head_d(head), flag_d(buffer), cumsum_d(buffer+text_size);

	// TODO

	cudaFree(buffer);
	return nhead;
}

void Part3(char *text, int *pos, int *head, int text_size, int n_head)
{
}
