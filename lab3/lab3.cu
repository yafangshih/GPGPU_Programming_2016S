#include "lab3.h"
#include <cstdio>

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

__global__ void findEdge(float* edge, float* padmask, int wt, int ht, const int oy, const int ox, int wb, int hb){
	const int yt = blockIdx.y * blockDim.y + threadIdx.y;
	const int xt = blockIdx.x * blockDim.x + threadIdx.x;
	const int curt = wt*yt+xt;

	if ((0 < yt) and (yt < ht-1) and (0 < xt) and (xt < wt-1) ) {
		const int cure = (wt-2)*(yt-1)+(xt-1);
		float ans = 0;
		
		float grad = padmask[curt]*4 + (-1)*(padmask[wt*(yt-1)+xt] + padmask[wt*yt+(xt-1)] + padmask[wt*(yt+1)+xt] + padmask[wt*yt+xt+1]);
		if(grad > 0 || grad < 0){
			ans = 127.0;
		}
		else if(padmask[curt] < 127){ ans = 0; }
		else if(padmask[curt] > 127){ ans = 255.0;}
		edge[cure] = ans;
	}

}

__global__ void toEdgepad(float* padmask, float* edge, int wt, int ht){
	const int yt = blockIdx.y * blockDim.y + threadIdx.y;
	const int xt = blockIdx.x * blockDim.x + threadIdx.x;
	const int curt = wt*yt+xt;

	if ((0 < yt) and (yt < ht-1) and (0 < xt) and (xt < wt-1)) {
		const int cure = (wt-2)*(yt-1)+(xt-1);
		padmask[curt] = edge[cure];
	}	
	else if((yt < ht) and (xt < wt)){
		padmask[curt] = -1.0;
	}

}

__global__ void computeAb(const float* mask, const float* background, float* A, float* b, const int ht, const int wt, const int c){
	const int yt = blockIdx.y * blockDim.y + threadIdx.y;
	const int xt = blockIdx.x * blockDim.x + threadIdx.x;
	const int curt = wt*yt+xt;

	const int yp = yt+1, xp = xt+1;
	const int hp = ht+2, wp = wt+2;
	int yoffset, xoffset, cyp, cxp;
	int count = 0;

	for(int i=0;i<4;i++){
		if(i==0) {yoffset = -1; xoffset = 0; }
		else if(i==1) {yoffset = 0; xoffset = -1; }
		else if(i==2) {yoffset = 0; xoffset = 1; }
		else {yoffset = 1; xoffset = 0; }
		cyp = yp+yoffset;
		cxp = xp+xoffset;

		if (!((0 < cyp) && (cyp < hp-1) && (0 < cxp) && (cxp < wp-1))) { 
			continue;
		}

//		printf("%f ", mask[cyp*wp + cxp]);

		if(mask[cyp*wp + cxp] < 64.0){ printf("back "); } //black, background
		else if ((191.0 > mask[cyp*wp + cxp]) && (mask[cyp*wp + cxp] > 64.0) ){ 
			printf("edge ");
			b[curt] += background[(((yt+yoffset)*wt)+(xt+xoffset))*3 + c]; 
			count++;
		} //edge
		else if (mask[cyp*wp + cxp] > 191.0){ 
			printf("fore ");
			A[(curt)*(wt*ht) + ((yt+yoffset)*wt)+(xt+xoffset)] = -1; 
			count++;
		} //white
	}
	A[(curt)*(wt*ht) + curt] = count;

/*
	if (0 < yt and yt < ht-1 and 0 < xt and xt < wt-1) {
		if(mask[(yt-1)*wt + xt] < 64){  } //black, background
		else if ((191 > mask[(yt-1)*wt + xt]) and mask[(yt-1)*wt + xt] > 64 ){ b[] } //edge
		else if (mask[(yt-1)*wt + xt] > 191){ A[((yt-1)*(wt-2) + xt-1)*(wt-2)*(ht-2) + (yt-2)*(wt-2) + xt-1] = -1; } //white
	}	
*/
}
__global__ void computeb(float* b, const float* target, const float* mask, int ht, int wt, int c){
	const int yt = blockIdx.y * blockDim.y + threadIdx.y;
	const int xt = blockIdx.x * blockDim.x + threadIdx.x;
	const int curt = wt*yt+xt;
	int yoffset, xoffset, cyt, cxt;

	for(int i=0;i<4;i++){
		if(i==0) {yoffset = -1; xoffset = 0; }
		else if(i==1) {yoffset = 0; xoffset = -1; }
		else if(i==2) {yoffset = 0; xoffset = 1; }
		else {yoffset = 1; xoffset = 0; }

		cyt = yt + yoffset;
		cxt = xt + xoffset;
		if(!((0 <= cyt) and (cyt < ht) and (0 <= cxt) and (cxt < wt))){ continue; }
		if(mask[(yt+1+yoffset)*(wt+2) + xt+1+xoffset] > 64){ // not background
			b[curt] += target[curt*3 + c] - target[(cyt*wt + cxt)*3 + c];
		}
	}
}

__global__ void jacobiRow(float* x, const float* A, const float* b, int ht, int wt){
	int idx = blockIdx.x;
	float sum = 0;
	for(int i=0;i<ht*wt; i++){
		sum += A[idx*(ht*wt) + i] * x[i];
	}
	__syncthreads();
	x[idx] = b[idx] - sum;
}

__global__ void paste(float* output, float* x, int wt, int ht, const int oy, const int ox, int wb, int hb, int c){
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
/*
__global__ void show(float* output, float* padmask, int wt, int ht, const int oy, const int ox, int wb, int hb){
	const int yt = blockIdx.y * blockDim.y + threadIdx.y;
	const int xt = blockIdx.x * blockDim.x + threadIdx.x;
	const int curt = wt*yt+xt;

	if (yt < ht and xt < wt) {
		const int yb = oy+yt, xb = ox+xt;
		const int curb = wb*yb+xb;
		if (0 <= yb and yb < hb and 0 <= xb and xb < wb) {
			output[curb*3+0] = padmask[curt];
			output[curb*3+1] = padmask[curt];
			output[curb*3+2] = padmask[curt];
		}
	}
}
*/

void addpad(float* padmask, const float* mask, int wt, int ht){
	cudaMemcpy(padmask, mask, sizeof(float), cudaMemcpyDeviceToDevice);
	cudaMemcpy(padmask+1, mask, wt*sizeof(float), cudaMemcpyDeviceToDevice);
	cudaMemcpy(padmask+wt+1, mask+wt-1, sizeof(float), cudaMemcpyDeviceToDevice);
	for(int i=1, j=0; i<ht+1; i++, j++){
		cudaMemcpy(padmask + i*(wt+2), mask + j*wt, sizeof(float), cudaMemcpyDeviceToDevice);
		cudaMemcpy(padmask + i*(wt+2) +1, mask + j*wt, wt*sizeof(float), cudaMemcpyDeviceToDevice);
		cudaMemcpy(padmask + (i+1)*(wt+2) -1, mask + (j+1)*wt - 1, sizeof(float), cudaMemcpyDeviceToDevice);
	}
	cudaMemcpy(padmask + (ht+1) *(wt+2), mask + (ht-1)*wt, sizeof(float), cudaMemcpyDeviceToDevice);
	cudaMemcpy(padmask + (ht+1)*(wt+2) +1, mask + (ht-1)*wt, wt*sizeof(float), cudaMemcpyDeviceToDevice);
	cudaMemcpy(padmask + (ht+1+1)*(wt+2) -1, mask + (ht-1+1)*wt - 1, sizeof(float), cudaMemcpyDeviceToDevice);
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
/*	SimpleClone<<<dim3(CeilDiv(wt,32), CeilDiv(ht,16)), dim3(32,16)>>>(
		background, target, mask, output,
		wb, hb, wt, ht, oy, ox
	);
*/

	// padding
	float *padmask;
	cudaMalloc((void **) &padmask, (wt+2)*(ht+2)*sizeof(float));
	addpad(padmask, mask, wt, ht);
	// find gradient edges
	float *edge;
	cudaMalloc((void **) &edge, wt*ht*sizeof(float));
	findEdge<<<dim3(CeilDiv(wt+2,32), CeilDiv(ht+2,16)), dim3(32,16)>>>(edge, padmask, wt+2, ht+2, oy, ox, wb, hb);
	toEdgepad<<<dim3(CeilDiv(wt+2,32), CeilDiv(ht+2,16)), dim3(32,16)>>>(padmask, edge, wt+2, ht+2);
/*
	for(int c=0;c<3;c++){
		paste<<<dim3(CeilDiv(wt+2,32), CeilDiv(ht+2,16)), dim3(32,16)>>>(output, padmask, wt+2, ht+2, oy, ox, wb, hb, c);
	}
*/
//	cudaFree(edge);

	float *A;
	cudaMalloc((void **) &A, (wt*ht)*(wt*ht)*sizeof(float));
	float *b;
	cudaMalloc((void **) &b, wt*ht*sizeof(float));
	float *x;
	cudaMalloc((void **) &x, wt*ht*sizeof(float));

	float *bpnt = (float *)malloc(wt*ht*sizeof(float));

	for(int c=0; c<3; c++){
		cudaMemset((void*)b, 0, wt*ht*sizeof(float));
		cudaMemset((void*)x, 0, wt*ht*sizeof(float));

		computeAb<<<dim3(CeilDiv(wt,32), CeilDiv(ht,16)), dim3(32,16)>>>(padmask, background, A, b, ht, wt, c);
		computeb<<<dim3(CeilDiv(wt,32), CeilDiv(ht,16)), dim3(32,16)>>>(b, target, padmask, ht, wt, c);
		//cudaMemcpy(bpnt, b, wt*ht*sizeof(float), cudaMemcpyDeviceToHost);
		//for(int j=0;j<wt*ht; j++){if(bpnt[j] > 0.0) printf("%f ", bpnt[j]);}
		//printf("\n");
		
		for(int i=0;i<5;i++){
			jacobiRow<<<1, ht*wt>>>(x, A, b, ht, wt);
		}


		paste<<<dim3(CeilDiv(wt,32), CeilDiv(ht,16)), dim3(32,16)>>>(output, x, wt, ht, oy, ox, wb, hb, c);
	}
	
	//show<<<dim3(CeilDiv(wt,32), CeilDiv(ht,16)), dim3(32,16)>>>(output, edge, wt, ht, oy, ox, wb, hb);



}
