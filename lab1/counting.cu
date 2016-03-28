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

__device__ bool CheckisText(char input){ // return 1 for english chars
	if('a'<=input and input<='z'){ return true; }
	if('A'<=input and input<='Z'){ return true; }
	return false;
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
	
	// copy to head
	thrust::remove_copy(flag_d + text_size, flag_d + 2*text_size, head_d, -1);

	cudaFree(buffer);
	return nhead;
}

//
__global__ void charcmp(const char *text, int *countbuf, char y){
	int idx = blockIdx.x * blockDim.x + threadIdx.x;
	if(text[idx] == y){
		countbuf[idx] = 1;
	}
	else{
		countbuf[idx] = 0;
	}
}


__global__ void SwitchText(char *text, char *textswap, int start) {
	// swap the char with the neighbor in the block

	int idx = start + blockIdx.x * blockDim.x + threadIdx.x;
	__shared__ char nbor[2];
	
	// a thread is responsible for loading 1 char 
	nbor[threadIdx.x] = text[idx];
	__syncthreads();
	
	if (CheckisText(nbor[0]) and CheckisText(nbor[1])) {
		// swap with the char loaded by neighbor
		textswap[idx] = nbor[(threadIdx.x+1)%2];
	}
	else{
		textswap[idx] = nbor[threadIdx.x];	
	}
}

void Part3(char *text, int *pos, int *head, int text_size, int n_head)
{
	/*
	Feature 1:
		count the occurence of each eng char
	*/
//	printf("Characters count:\n");
	
	int *cnt;
	cudaMalloc(&cnt, sizeof(int)*text_size);
	thrust::device_ptr<int> countbuf_d(cnt);
	int count = 0;

	for(int i='A', j=1; i<='z'; i++, j++){
		// fills 1 where the eng char appears
		charcmp<<<(text_size/32)+1, 32>>>(text, cnt, i);

		// reduce to a number, which is the # of occurence
		count = thrust::reduce(countbuf_d, countbuf_d + text_size);

//		uncomment this line to print the result on the screen
//		printf("%c:%d ", i, count);
//		if(j % 13 == 0){printf("\n");}
		if(i == 'Z'){i += 'a'-'Z'-1;}
	}	
	cudaFree(cnt);
	
	/*
	Feature 2:
		swap every pair of 2 english characters in a word
		the text after swap is stored back to *text (and will not be printed)
	*/
	if(n_head > 0){
		char *textswap;
		cudaMalloc(&textswap, sizeof(char)*text_size);

		int start = 0, end = text_size;
		int nhead = n_head;	

		// get the text between every 2 heads
		cudaMemcpy(&start, head, sizeof(int), cudaMemcpyDeviceToHost);
		
		while(nhead > 1){
			head++;
			cudaMemcpy(&end, head, sizeof(int), cudaMemcpyDeviceToHost);
			// swap the text
			SwitchText<<<((end - start)/2)+1, 2>>>(text, textswap, start);
			// find the next interval
			start = end;
			nhead--;
		}
		// swap the last interval
		end = text_size;
		SwitchText<<<((end - start)/2)+1, 2>>>(text, textswap, start);

		cudaMemcpy(text, textswap, sizeof(char)*text_size, cudaMemcpyDeviceToDevice);

		cudaFree(textswap);
	}
	else{ // # of word == 0
	//	printf("No need for swap.\n");
	}
	

}
