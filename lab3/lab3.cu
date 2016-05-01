#include "lab3.h"
#include <cstdio>
#include <iostream>

#include <thrust/device_vector.h>
#include <thrust/reduce.h>
#include <thrust/execution_policy.h>

#define PRECISION 0.00001

__device__ __host__ int CeilDiv(int a, int b) { return (a-1)/b + 1; }
__device__ __host__ int CeilAlign(int a, int b) { return CeilDiv(a, b) * b; }


__global__ void initialAxb(const float* mask, const float* background, const float* target, float* A, float* b, float* x, const int ht, const int wt, const int oy, const int ox, const int wb, const int hb, const int c, int *f){

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

				int ycurb = yt+oy+yoffset;
				int xcurb = xt+ox+xoffset;
				
				if( ! ((ycurb >= 0) && (ycurb < hb) && (xcurb >= 0) && (xcurb < wb))){ 
					continue;
				}
				else{
					if( ! ((yneibor >= 0) && (yneibor < ht) && (xneibor >= 0) && (xneibor < wt))){ 
						count++;
						pos = curt*5 + i + 1;
						A[pos] = -1;
					}
					else{
						int neibor = yneibor*wt + xneibor;
						count ++;
						pos = curt*5 + i + 1;
						A[pos] = -1;
						b[curt] += (target[curt*3 + c] - target[(neibor)*3 + c]);
					}
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


__global__ void jacobiRow(float* x, float* tmpx, const float* A, const float* background, const float* b, int ht, int wt, int hb, int wb, int oy, int ox, int c){
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
			neibor = yneibor*wt + xneibor;

			int ycurb = yt+oy+yoffset;
			int xcurb = xt+ox+xoffset;

			if(! ((ycurb >= 0) && (ycurb < hb) && (xcurb >= 0) && (xcurb < wb))){
				continue;
			}
			else{
				if( ! ((yneibor >= 0) && (yneibor < ht) && (xneibor >= 0) && (xneibor < wt))){ 
					sum = sum+ A[idx*5 +i +1] * background[((yt+oy+yoffset)*wb+(xt+ox+xoffset))*3 + c];
				}
				else{
					sum = sum+ A[idx*5 +i +1] * x[neibor];
				}
			}
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


__global__ void set2one(int* ptr, int sz){
	int idx = blockIdx.x * blockDim.x + threadIdx.x;
	if(idx < sz){ptr[idx] = 1;}
}

void PoissonImageCloning(
	const float *background,
	const float *target,
	const float *mask,
	float *output,
	const int wb, const int hb, const int wt, const int ht,
	const int oy, const int ox
)
{

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

	for(int c=0; c<3; c++){
		cudaMemset((void*)A, 0, 5*(wt*ht)*sizeof(float));
		cudaMemset((void*)b, 0, wt*ht*sizeof(float));
		cudaMemset((void*)x, 0, wt*ht*sizeof(float));
		cudaMemset((void*)tmpx, 0, wt*ht*sizeof(float));
		set2one<<<((ht*wt)/32)+1, 32>>>(f, wt*ht);

		int notyet = wt*ht;

		initialAxb<<<dim3(CeilDiv(wt,32), CeilDiv(ht,16)), dim3(32,16)>>>(mask, background, target, A, b, x, ht, wt, oy, ox, wb, hb, c, f);
	
		int iter =0;
		while(notyet != 0){
			jacobiRow<<<((ht*wt)/32)+1, 32>>>(x, tmpx, A, background, b, ht, wt, hb, wb, oy, ox, c);
			copy2x<<<((ht*wt)/32)+1, 32>>>(x, tmpx, A, b, mask, ht, wt, f);

			thrust::device_vector<int> flag_d(f, f + wt*ht);
			notyet = thrust::reduce(thrust::device, flag_d.begin(), flag_d.end());
			iter++;
		}
		printf("%d %d\n", c, iter);
		paste<<<dim3(CeilDiv(wt,32), CeilDiv(ht,16)), dim3(32,16)>>>(output, x, background, wt, ht, oy, ox, wb, hb, c);
	}
	
	

}