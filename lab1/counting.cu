#include "counting.h"
#include <cstdio>
#include <cassert>
#include <thrust/scan.h>
#include <thrust/transform.h>
#include <thrust/functional.h>
#include <thrust/device_ptr.h>
#include <thrust/execution_policy.h>

#include <thrust/replace.h>
#include <thrust/remove.h>
#include <thrust/reduce.h>
/*
nvcc -std=c++11 -arch=sm_30 -O2 -c counting.cu -o counting.o
nvcc -std=c++11 -arch=sm_30 -O2 main.cu counting.o -o main
*/

__device__ __host__ int CeilDiv(int a, int b) { return (a-1)/b + 1; }
__device__ __host__ int CeilAlign(int a, int b) { return CeilDiv(a, b) * b; }

__device__ int tree[10][50000000];

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

	tree[0][idx] = nontxtList_gpu[idx];

}

__device__ int power2(int pow){
	int ans = 1;
	for(int i=0; i<pow; i++){
		ans = ans * 2;
	}
	return ans;
}



__global__ void indextreeInit(int *indexList, int text_size){
	int idx = blockIdx.x * blockDim.x + threadIdx.x;
	tree[0][idx] = indexList[idx];
}

__global__ void indextreeSum(int *indexList, int text_size, int base){
	int idx = blockIdx.x * blockDim.x + threadIdx.x;
		
	if(tree[base][idx] != 0 && idx - power2(base) >= 0){
		tree[base+1][idx] = tree[base][idx] & tree[base][idx - power2(base)];
	}
}


__global__ void treeDown(int *indexList, int base){
	int idx = blockIdx.x * blockDim.x + threadIdx.x;
	int index = idx;
	indexList[idx] = 0;
	
	while(base >= 0 and index >= 0){
		if(tree[base][index] == 0){
			base--;
		}
		else{
			indexList[idx] += power2(base);
			index = index - power2(base);
		}
	}
}

/*
__global__ void indextreeSum_notSafe(int *indexList, int text_size){

	int idx = blockIdx.x * blockDim.x + threadIdx.x;
	
	tree[0][idx] = indexList[idx];
	indexList[idx] = 0;
	
	int base = 0;
	while(tree[base][idx] != 0 && idx - power2(base) >= 0){
		tree[base+1][idx] = tree[base][idx] & tree[base][idx - power2(base)];
		base++;
	}
	int index = idx;
	while(base >= 0 and index >= 0){
		if(tree[base][index] == 0){
			base--;
		}
		else{
			indexList[idx] += power2(base);
			index = index - power2(base);
		}
	}
}
*/

void CountPosition(const char *text, int *pos, int text_size)
{

	FindnonText<<<(text_size/32)+1, 32>>>(text, pos, text_size);
	
	indextreeInit<<<(text_size/32)+1, 32>>>(pos, text_size);
	
	for(int i=0;i<10;i++){
		indextreeSum<<<(text_size/32)+1, 32>>>(pos, text_size, i);
	}
	
	treeDown<<<(text_size/32)+1, 32>>>(pos, 9);
	cudaDeviceSynchronize();

}

struct isnt_one
{
  __host__ __device__ bool operator()(int x){
    
    if(x!=1){
    	return true;
    }
    else return false;
}    
};

int ExtractHead(const int *pos, int *head, int text_size)
{
	int *buffer;
	int nhead;
	cudaMalloc(&buffer, sizeof(int)*text_size*2); // this is enough
	
	thrust::device_ptr<const int> pos_d(pos);
	thrust::device_ptr<int> head_d(head), flag_d(buffer), cumsum_d(buffer+text_size);

//	int cpumem[2*text_size];

	// TODO
	cudaMemset((void*)buffer, 0, 2*text_size*sizeof(int));

	// pos_d: {0, 1, 2, 0}, flag_d: {0, 0, 0, 0}
/*	printf("buffer before:\n");
	cudaMemcpy(cpumem, buffer, text_size*2*sizeof(int), cudaMemcpyDeviceToHost);
	for(int i=0;i<2*text_size;i++){
		printf("%d ", cpumem[i]);
	}
	printf("\n");

	isnt_one opIsntOne;

//	thrust::replace_copy_if(pos_d, pos_d + text_size, flag_d, opIsntOne, 0);
*/  
	// neg the numbers that != 1
	thrust::negate<int> opNeg; 
	thrust::transform_if(thrust::device, pos_d, pos_d + text_size, flag_d, opNeg, isnt_one());

	// build a filter
	thrust::plus<int> opPlus;
	thrust::transform(thrust::device, pos_d, pos_d + text_size, flag_d, flag_d, opPlus);

	// reduce the filter to a number, which is the # of heads
	nhead = thrust::reduce(flag_d, flag_d + text_size);
	
	// filter out the position of the 1s
	thrust::sequence(thrust::device, flag_d + text_size, flag_d + 2*text_size);
	thrust::replace_if(flag_d + text_size, flag_d + 2*text_size, flag_d, isnt_one(), -1);
	
	// copy yto head
	thrust::remove_copy(flag_d + text_size, flag_d + 2*text_size, head_d, -1);

	cudaFree(buffer);
	return nhead;
}

void Part3(char *text, int *pos, int *head, int text_size, int n_head)
{
}
