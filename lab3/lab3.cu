#include "lab3.h"
#include <cstdio>
#include <iostream>

#include <thrust/device_vector.h>
#include <thrust/reduce.h>
#include <thrust/execution_policy.h>

#define PRECISION 0.0001

__device__ __host__ int CeilDiv(int a, int b) { return (a-1)/b + 1; }
__device__ __host__ int CeilAlign(int a, int b) { return CeilDiv(a, b) * b; }

__global__ void SimpleClone(
	const float *background,
	const float *target,
	const float *mask,
	float *output,
	const int wb, const int hb, const int wt, const int ht,
	const int oy, const int ox
)
{
	const int yt = blockIdx.y * blockDim.y + threadIdx.y;
	const int xt = blockIdx.x * blockDim.x + threadIdx.x;
	const int curt = wt*yt+xt;
	if ((yt < ht) and (xt < wt) and (mask[curt] > 127.0f) ) {
		const int yb = oy+yt, xb = ox+xt;
		const int curb = wb*yb+xb;
		if ((0 <= yb) and (yb < hb) and (0 <= xb) and (xb < wb) ) {
			output[curb*3+0] = target[curt*3+0];
			output[curb*3+1] = target[curt*3+1];
			output[curb*3+2] = target[curt*3+2];
		}
	}
}

__global__ void initialAxb(const float* mask, const float* background, const float* target, float* A, float* b, float* x, const int ht, const int wt, const int oy, const int ox, const int wb, const int c, int *f){

	const int yt = blockIdx.y * blockDim.y + threadIdx.y;
	const int xt = blockIdx.x * blockDim.x + threadIdx.x;
	const int curt = wt*yt+xt;

	int yoffset, xoffset, yneibor, xneibor;
	int count = 0;
	int pos = 0;

	if ((0 <= yt) and (yt < ht) and (0 <= xt) and (xt < wt) ) {

		if(mask[curt] > 127){
			x[curt] = target[curt*3+c];

			for(int i=0;i<4;i++){
				if(i==0) {yoffset = -1; xoffset = 0; }
				else if(i==1) {yoffset = 0; xoffset = -1; }
				else if(i==2) {yoffset = 0; xoffset = 1; }
				else {yoffset = 1; xoffset = 0; }

				yneibor = yt+yoffset;
				xneibor = xt+xoffset;
			
				if( ! ((yneibor >= 0) && (yneibor < ht) && (xneibor >= 0) && (xneibor < wt))){ 
					continue; 
				}
				else{
					int neibor = yneibor*wt + xneibor;
					count ++;
					pos = curt*5 + i + 1;
					A[pos] = -1;
					b[curt] += (target[curt*3 + c] - target[(neibor)*3 + c]);
				}
				A[curt*5] = count;
			}
		}
		else{
			x[curt] = background[((yt+oy)*wb + (xt+ox))*3 + c];
//			A[curt*5] = 1;
			for(int i=0; i<5; i++) { A[curt*5 + i] = 0; }
			b[curt] = background[((yt+oy)*wb + (xt+ox))*3 + c]; //0;
		}
	}


}


__global__ void jacobiRow(float* x, float* tmpx, const float* A, const float* b, int ht, int wt){
	int idx = blockIdx.x * blockDim.x + threadIdx.x;
	float sum = 0;

	int yt = idx/wt, xt = idx%wt;
	int yoffset, xoffset, yneibor, xneibor, neibor;


	if(idx < ht*wt){
//		sum = sum+ A[idx*5] * x[idx];

		for(int i=0;i<4;i++){
			if(i==0) {yoffset = -1; xoffset = 0; }
			else if(i==1) {yoffset = 0; xoffset = -1; }
			else if(i==2) {yoffset = 0; xoffset = 1; }
			else {yoffset = 1; xoffset = 0; }

			yneibor = yt + yoffset;
			xneibor = xt + xoffset;

			if( ! ((yneibor >= 0) && (yneibor < ht) && (xneibor >= 0) && (xneibor < wt))){ 
				continue; 
			}

			neibor = yneibor*wt + xneibor;
			sum = sum+ A[idx*5 +i +1] * x[neibor];
		}
		tmpx[idx] = sum;

	}
}

__device__ float myabs(float x){
	if(x < 0){ return -x; }
	return x;
}

__global__ void copy2x(float* x, float* tmpx, float *A, float *b, const float * mask, int ht, int wt, int * f){
	int idx = blockIdx.x * blockDim.x + threadIdx.x;

	float tmp = 0;

	if(idx < ht*wt){
		if(mask[idx] > 127){
		//x[idx] = b[idx] - tmpx[idx];
			tmp = (b[idx] - tmpx[idx])/A[idx*5];
			if(myabs(x[idx] - tmp) < PRECISION){f[idx] = 0;}
			x[idx] = tmp;
		}
		else{
			f[idx] = 0;
		}
	}
}

__global__ void paste(float* output, float* x, const float* background, int wt, int ht, const int oy, const int ox, int wb, int hb, int c){
	const int yt = blockIdx.y * blockDim.y + threadIdx.y;
	const int xt = blockIdx.x * blockDim.x + threadIdx.x;
	const int curt = wt*yt+xt;

	if ((yt < ht) and (xt < wt) ) {
		const int yb = oy+yt, xb = ox+xt;
		const int curb = wb*yb+xb;
		if ((0 <= yb) and (yb < hb) and (0 <= xb) and (xb < wb) ) {
			output[curb*3+c] = x[curt];
		}
	}
}

int checkdone(int* fcpu, int ht, int wt){
	for(int i=0; i<wt*ht; i++){
		if(fcpu[i] == 0){return 1;}
	}
	return 0;
}

__global__ void set2one(int* ptr, int sz){
	int idx = blockIdx.x * blockDim.x + threadIdx.x;
	if(idx < sz){ptr[idx] = 1;}
}

void addpad(float* padmask, const float* mask, int wt, int ht, float* zero){
	
	cudaMemcpy(padmask, zero, sizeof(float), cudaMemcpyDeviceToDevice);
	cudaMemcpy(padmask+1, zero, wt*sizeof(float), cudaMemcpyDeviceToDevice);
	cudaMemcpy(padmask+wt+1, zero, sizeof(float), cudaMemcpyDeviceToDevice);

	for(int i=1, j=0; i<ht+1; i++, j++){
		cudaMemcpy(padmask + i*(wt+2), zero, sizeof(float), cudaMemcpyDeviceToDevice);
		cudaMemcpy(padmask + i*(wt+2) +1, mask + j*wt, wt*sizeof(float), cudaMemcpyDeviceToDevice);
		cudaMemcpy(padmask + (i+1)*(wt+2) -1, zero, sizeof(float), cudaMemcpyDeviceToDevice);
	}
	cudaMemcpy(padmask + (ht+1) *(wt+2), zero, sizeof(float), cudaMemcpyDeviceToDevice);
	cudaMemcpy(padmask + (ht+1)*(wt+2) +1, zero, wt*sizeof(float), cudaMemcpyDeviceToDevice);
	cudaMemcpy(padmask + (ht+1+1)*(wt+2) -1, zero, sizeof(float), cudaMemcpyDeviceToDevice);
}

void addpadtarget(float* padmask, const float* mask, int wt, int ht){

	cudaMemcpy(padmask, mask, 3*sizeof(float), cudaMemcpyDeviceToDevice);
	cudaMemcpy(padmask+3, mask, 3*wt*sizeof(float), cudaMemcpyDeviceToDevice);
	cudaMemcpy(padmask+3*wt+3, mask+3*wt-3, 3*sizeof(float), cudaMemcpyDeviceToDevice);

	for(int i=1, j=0; i<ht+1; i++, j++){
		cudaMemcpy(padmask + 3*i*(wt+2), mask + 3*j*wt, 3*sizeof(float), cudaMemcpyDeviceToDevice);
		cudaMemcpy(padmask + 3*i*(wt+2) + 3, mask + 3*j*wt, 3*wt*sizeof(float), cudaMemcpyDeviceToDevice);
		cudaMemcpy(padmask + 3*(i+1)*(wt+2) - 3, mask + 3*(j+1)*wt - 3, 3*sizeof(float), cudaMemcpyDeviceToDevice);
	}
	cudaMemcpy(padmask + 3*(ht+1)*(wt+2), mask + 3*(ht-1)*wt, 3*sizeof(float), cudaMemcpyDeviceToDevice);
	cudaMemcpy(padmask + 3*(ht+1)*(wt+2) +3*1, mask + 3*(ht-1)*wt, 3*wt*sizeof(float), cudaMemcpyDeviceToDevice);
	cudaMemcpy(padmask + 3*(ht+1+1)*(wt+2) -3*1, mask + 3*(ht-1+1)*wt - 3*1, 3*sizeof(float), cudaMemcpyDeviceToDevice);
}


void PoissonImageCloning(
	const float *background,
	const float *target,
	const float *mask,
	float *output,
	const int wb, const int hb, const int wt0, const int ht0,
	const int oy0, const int ox0
)
{
	int wt=wt0+2, ht=ht0+2;
	int oy=oy0-1, ox=ox0-1;

	cudaMemcpy(output, background, wb*hb*sizeof(float)*3, cudaMemcpyDeviceToDevice);

	float *A;
	cudaMalloc((void **) &A, 5*(wt*ht)*sizeof(float));
	float *b;
	cudaMalloc((void **) &b, wt*ht*sizeof(float));
	float *x;
	cudaMalloc((void **) &x, wt*ht*sizeof(float));
	float *tmpx;
	cudaMalloc((void **) &tmpx, wt*ht*sizeof(float));
    int *f;
	cudaMalloc((void **) &f, wt*ht*sizeof(int));	


	float* padmask;
	cudaMalloc((void **) &padmask, (wt*ht)*sizeof(float));
	float *zero;
	cudaMalloc((void **) &zero, (wt*ht)*sizeof(float));
	cudaMemset((void*)zero, 0, wt*ht*sizeof(float));

	addpad(padmask, mask, wt0, ht0, zero);


	float* padtarget;
	cudaMalloc((void **) &padtarget, 3*(wt*ht)*sizeof(float));
	addpadtarget(padtarget, target, wt0, ht0);

	for(int c=0; c<3; c++){
		cudaMemset((void*)A, 0, 5*(wt*ht)*sizeof(float));
		cudaMemset((void*)b, 0, wt*ht*sizeof(float));
		cudaMemset((void*)x, 0, wt*ht*sizeof(float));
		cudaMemset((void*)tmpx, 0, wt*ht*sizeof(float));
		set2one<<<((ht*wt)/32)+1, 32>>>(f, wt*ht);

		int notyet = wt*ht;

		initialAxb<<<dim3(CeilDiv(wt,32), CeilDiv(ht,16)), dim3(32,16)>>>(padmask, background, padtarget, A, b, x, ht, wt, oy, ox, wb, c, f);
	
//		int iter =0;
		while(notyet != 0){
			jacobiRow<<<((ht*wt)/32)+1, 32>>>(x, tmpx, A, b, ht, wt);
			copy2x<<<((ht*wt)/32)+1, 32>>>(x, tmpx, A, b, padmask, ht, wt, f);

			thrust::device_vector<int> flag_d(f, f + wt*ht);
			notyet = thrust::reduce(thrust::device, flag_d.begin(), flag_d.end());
//			iter++;
		}
//		printf("%d %d\n", c, iter);
		paste<<<dim3(CeilDiv(wt,32), CeilDiv(ht,16)), dim3(32,16)>>>(output, x, background, wt, ht, oy, ox, wb, hb, c);
	}
	
	

}