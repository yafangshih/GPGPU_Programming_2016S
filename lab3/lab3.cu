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

__global__ void findEdge(float* edge, const float* mask, int wt, int ht, const int oy, const int ox, int wb, int hb){
	const int yt = blockIdx.y * blockDim.y + threadIdx.y;
	const int xt = blockIdx.x * blockDim.x + threadIdx.x;
	const int curt = wt*yt+xt;
	int xoffset, yoffset;
	int xneibor, yneibor;

	if ((0 <= yt) and (yt < ht) and (0 <= xt) and (xt < wt) ) {
		int ans = 0;
		int grad = 0;
		for(int i=0;i<4;i++){
			if(i==0) {yoffset = -1; xoffset = 0; }
			else if(i==1) {yoffset = 0; xoffset = -1; }
			else if(i==2) {yoffset = 0; xoffset = 1; }
			else {yoffset = 1; xoffset = 0; }

			yneibor = yt+yoffset;
			xneibor = xt+xoffset;
			
			if( ! ((yneibor >= 0) && (yneibor < ht) && (xneibor >= 0) && (xneibor < wt))){ continue; }
			else{
				grad += mask[curt] + (-1)*(mask[yneibor*wt + xneibor]);
			}
		}
		if(grad > 0 || grad < 0){
			ans = 127;
		}
		else if(mask[curt] < 127){ 
			ans = 0; 
		}
		else if(mask[curt] > 127){ ans = 255;}
		edge[curt] = ans;
	}

}
/*
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
*/
__global__ void computeAb(const float* edge, const float* background, float* A, float* b, float* x, const int ht, const int wt, const int c, int *cntgpu, int *f){
//(edge, background, A, b, ht, wt, c, cntgpu)

	const int yt = blockIdx.y * blockDim.y + threadIdx.y;
	const int xt = blockIdx.x * blockDim.x + threadIdx.x;
	const int curt = wt*yt+xt;

	int yoffset, xoffset, yneibor, xneibor;
	int count = 0;
	int pos = 0;

	if ((0 <= yt) and (yt < ht) and (0 <= xt) and (xt < wt) ) {

	//	atomicAdd(cntgpu, 1);

		for(int i=0;i<4;i++){
			if(i==0) {yoffset = -1; xoffset = 0; }
			else if(i==1) {yoffset = 0; xoffset = -1; }
			else if(i==2) {yoffset = 0; xoffset = 1; }
			else {yoffset = 1; xoffset = 0; }

			yneibor = yt+yoffset;
			xneibor = xt+xoffset;
		
			if( ! ((yneibor >= 0) && (yneibor < ht) && (xneibor >= 0) && (xneibor < wt))){ 
				atomicAdd(cntgpu, 1);
				continue; 
			}
			else{
	
				int neibor = yneibor*wt + xneibor;

				if(edge[neibor] < 64){ 

				} //black, background
				else if ((191 > edge[neibor]) && (edge[neibor] > 64) ){ 
					b[curt] += background[(neibor)*3 + c]; 
					count+=1;
				} 
				else if (edge[neibor] > 191){ 
					pos = curt*5 + i + 1;
					A[pos] = -1;
					count = count + 1;
				} 
			}
	
		}
//		pos = (curt)*(wt*ht) + curt;
		A[curt*5] = count;
//		printf("%f %f %f %f %f\n", A[curt*5], A[curt*5+1], A[curt*5+2], A[curt*5+3], A[curt*5+4]);
		
	}


/*
	if (0 < yt and yt < ht-1 and 0 < xt and xt < wt-1) {
		if(mask[(yt-1)*wt + xt] < 64){  } //black, background
		else if ((191 > mask[(yt-1)*wt + xt]) and mask[(yt-1)*wt + xt] > 64 ){ b[] } //edge
		else if (mask[(yt-1)*wt + xt] > 191){ A[((yt-1)*(wt-2) + xt-1)*(wt-2)*(ht-2) + (yt-2)*(wt-2) + xt-1] = -1; } //white
	}	
*/
}

__global__ void computeb(float* b, const float* target, const float* edge, const float* background, float* x, int ht, int wt, int c, int *cntgpu){
	//(b, target, edge, ht, wt, c)

	const int yt = blockIdx.y * blockDim.y + threadIdx.y;
	const int xt = blockIdx.x * blockDim.x + threadIdx.x;
	const int curt = wt*yt+xt;
	int yoffset, xoffset, yneibor, xneibor;


	if ((0 <= yt) and (yt < ht) and (0 <= xt) and (xt < wt) ) {

		if(edge[curt] < 191){x[curt] = background[curt*3 +c];}

		for(int i=0;i<4;i++){
			if(i==0) {yoffset = -1; xoffset = 0; }
			else if(i==1) {yoffset = 0; xoffset = -1; }
			else if(i==2) {yoffset = 0; xoffset = 1; }
			else {yoffset = 1; xoffset = 0; }

			yneibor = yt + yoffset;
			xneibor = xt + xoffset;

			
			if( ! ((yneibor >= 0) && (yneibor < ht) && (xneibor >= 0) && (xneibor < wt))){ 
				atomicAdd(cntgpu, 1);
				continue; 
			}

			int neibor = yneibor*wt + xneibor;
			if(edge[neibor] > 64){ // not background
				b[curt] += (target[curt*3 + c] - target[(neibor)*3 + c]);
			}
		}
	}
}


__global__ void jacobiRow(float* x, float* tmpx, const float* A, const float* b, int ht, int wt){
	int idx = blockIdx.x * blockDim.x + threadIdx.x;
	float sum = 0;

	int yt = idx/wt, xt = idx%wt;
//	printf("%d %d\n", yt, xt);

	int yoffset, xoffset, yneibor, xneibor, neibor;

	sum = sum+ A[idx*5] * x[idx];

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

__global__ void copy2x(float* x, float* tmpx, float *b, int* prnt){
	int idx = blockIdx.x * blockDim.x + threadIdx.x;
	prnt[idx] = idx;
	x[idx] = b[idx] - tmpx[idx];

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
/*
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
*/
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

/*	// padding
	float *padmask;
	cudaMalloc((void **) &padmask, (wt+2)*(ht+2)*sizeof(float));
	addpad(padmask, mask, wt, ht);
*/

	int *f;
	cudaMalloc((void **) &f, sizeof(int));
	cudaMemset((void*)f, 0, sizeof(int));
	int *fcpu = (int *)malloc(sizeof(int));

	// find gradient edges
	float *edge;
	cudaMalloc((void **) &edge, wt*ht*sizeof(float));

	float *tmpx;
	cudaMalloc((void **) &tmpx, wt*ht*sizeof(float));
	cudaMemset((void*)tmpx, 0, wt*ht*sizeof(float));

	findEdge<<<dim3(CeilDiv(wt,32), CeilDiv(ht,16)), dim3(32,16)>>>(edge, mask, wt, ht, oy, ox, wb, hb);

/*
	cudaMemcpy(fcpu, f, sizeof(int), cudaMemcpyDeviceToHost);
	printf("f = %d \n",*fcpu);
	cudaMemset((void*)f, 0, sizeof(int));
*/
	/*
	for(int c=0;c<3;c++)
		paste<<<dim3(CeilDiv(wt,32), CeilDiv(ht,16)), dim3(32,16)>>>(output, edge, wt, ht, oy, ox, wb, hb, c);
*/
//	toEdgepad<<<dim3(CeilDiv(wt+2,32), CeilDiv(ht+2,16)), dim3(32,16)>>>(padmask, edge, wt+2, ht+2);
/*
	for(int c=0;c<3;c++){
		paste<<<dim3(CeilDiv(wt,32), CeilDiv(ht,16)), dim3(32,16)>>>(output, edge, wt, ht, oy, ox, wb, hb, c);
	}
*/


	float *A;
//	int size = ((wt*ht)/4)*(wt*ht);
//	printf("size = %d\n", size);
	cudaMalloc((void **) &A, 5*(wt*ht)*sizeof(float));
	cudaMemset((void*)A, 0, 5*(wt*ht)*sizeof(float));

	float *b;
	cudaMalloc((void **) &b, wt*ht*sizeof(float));
	float *x;
	cudaMalloc((void **) &x, wt*ht*sizeof(float));



	int *cntgpu;
	cudaMalloc((void **) &cntgpu, sizeof(int));
	cudaMemset((void*)cntgpu, 0, sizeof(int));
	int *cnt = (int *)malloc(sizeof(int));
/*
	int *f;
	cudaMalloc((void **) &f, sizeof(int));
	cudaMemset((void*)f, 0, sizeof(int));
	int *fcpu = (int *)malloc(sizeof(int));
*/

	int *prntcpu = (int *)malloc((wt*ht)*sizeof(int));
	int *prnt;
	cudaMalloc((void **) &prnt, (wt*ht)*sizeof(int));

	for(int c=0; c<3; c++){
		cudaMemset((void*)b, 0, wt*ht*sizeof(float));
		cudaMemset((void*)x, 0, wt*ht*sizeof(float));

	//	cudaMemset((void*)cntgpu, 0, sizeof(int));

		computeAb<<<dim3(CeilDiv(wt,32), CeilDiv(ht,16)), dim3(32,16)>>>(edge, background, A, b, tmpx, ht, wt, c, cntgpu, f);
		
		cudaMemcpy(cnt, cntgpu, sizeof(int), cudaMemcpyDeviceToHost);
		printf("cnt = %d \n",*cnt);
		cudaMemset((void*)cntgpu, 0, sizeof(int));
/*
		cudaMemcpy(fcpu, f, sizeof(int), cudaMemcpyDeviceToHost);
		printf("f = %d \n",*fcpu);
		cudaMemset((void*)f, 0, sizeof(int));
*/
		computeb<<<dim3(CeilDiv(wt,32), CeilDiv(ht,16)), dim3(32,16)>>>(b, target, edge, background, tmpx, ht, wt, c, cntgpu);
	//	cudaMemcpy(prnt, b, (wt*ht)*sizeof(float), cudaMemcpyDeviceToHost);
	//	for(int j=0;j<wt*ht; j++){ printf("%f ", prnt[j]);}


	//	cudaMemcpy(cnt, cntgpu, sizeof(int), cudaMemcpyDeviceToHost);
		//for(int j=0;j<wt*ht; j++){if(bpnt[j] > 0.0) printf("%f ", bpnt[j]);}
	//	printf("cnt = %d \n",*cnt);

	
		for(int i=0;i<3;i++){
//			cudaMemset((void*)tmpx, 0, wt*ht*sizeof(float));
			jacobiRow<<<((ht*wt)/32)+1, 32>>>(x, tmpx, A, b, ht, wt);
/*
			if(i == 9){
				cudaMemcpy(prnt, tmpx, (wt*ht)*sizeof(float), cudaMemcpyDeviceToHost);
				for(int j=0;j<wt*ht; j++){ printf("%f ", prnt[j]);}
			}
*/
			copy2x<<<((ht*wt)/32)+1, 32>>>(x, tmpx, b, prnt);
//			cudaMemcpy(prntcpu, prnt, (wt*ht)*sizeof(int), cudaMemcpyDeviceToHost);
//			for(int j=0;j<wt*ht; j++){ printf("%d ", prntcpu[j]);}
		}

		paste<<<dim3(CeilDiv(wt,32), CeilDiv(ht,16)), dim3(32,16)>>>(output, x, wt, ht, oy, ox, wb, hb, c);
		
//		cudaMemcpy(prnt, x, (wt*ht)*sizeof(float), cudaMemcpyDeviceToHost);
//		for(int j=0;j<wt*ht; j++){ printf("%f ", prnt[j]);}
	}
	
	//show<<<dim3(CeilDiv(wt,32), CeilDiv(ht,16)), dim3(32,16)>>>(output, edge, wt, ht, oy, ox, wb, hb);



}
