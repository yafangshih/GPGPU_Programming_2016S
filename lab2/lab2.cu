#include "lab2.h"
#include <math.h>
#include <cstdio>
#include <cstdint>
#include <cstdlib>

#include <opencv2/opencv.hpp>

/*frame settings*/
static const unsigned W = 800;
static const unsigned H = 450;
static const unsigned NFRAME = 960;
static const int ANGLE = 1;
static const int octs = 5;

static const double freq = (double)1/(double)96;

// two colors to interpolate
static const double Y0 = 119;
static const double U0 = 95;
static const double V0 = 225;
static const double Y1 = 239;
static const double U1 = 117;
static const double V1 = 139;

#define PI 3.14159265
#define E 2.71828182


 __device__ double dirs[256][2]; 

// taken from Ken Perlin 
// http://mrl.nyu.edu/~perlin/noise/
 __device__ int perm[256] = { 151,160,137,91,90,15, 
		131,13,201,95,96,53,194,233,7,225,140,36,103,30,69,142,8,99,37,240,21,10,23, 
		190, 6,148,247,120,234,75,0,26,197,62,94,252,219,203,117,35,11,32,57,177,33, 
		88,237,149,56,87,174,20,125,136,171,168, 68,175,74,165,71,134,139,48,27,166, 
		77,146,158,231,83,111,229,122,60,211,133,230,220,105,92,41,55,46,245,40,244, 
		102,143,54, 65,25,63,161, 1,216,80,73,209,76,132,187,208, 89,18,169,200,196, 
		135,130,116,188,159,86,164,100,109,198,173,186, 3,64,52,217,226,250,124,123, 
		5,202,38,147,118,126,255,82,85,212,207,206,59,227,47,16,58,17,182,189,28,42, 
		223,183,170,213,119,248,152, 2,44,154,163, 70,221,153,101,155,167, 43,172,9, 
		129,22,39,253, 19,98,108,110,79,113,224,232,178,185, 112,104,218,246,97,228, 
		251,34,242,193,238,210,144,12,191,179,162,241, 81,51,145,235,249,14,239,107, 
		49,192,214, 31,181,199,106,157,184, 84,204,176,115,121,50,45,127, 4,150,254, 
		138,236,205,93,222,114,67,29,24,72,243,141,128,195,78,66,215,61,156,180 };

__device__ double power(double x, int y){
	double ans = 1;
	for(int i=0; i<y; i++){ ans = ans * x; }
	return ans;
}

__device__ double dblAbs(double x){
	if(x < 0){ return -x; }
	return x;
}
// much idea taken from http://gamedev.stackexchange.com/questions/23625/how-do-you-generate-tileable-perlin-noise

// find the weights of 4 corners
__device__ double Corners4(double x, double y, int perX, int perY, int c, int f){
	
	int gridX = (int)x + c%2, gridY = (int)y + c/2;
	int hashed = perm[ (perm[ (gridX%perX)%256 ] + gridY%perY)%256];
	double grad = (x-gridX) * dirs[(hashed + ANGLE*f) % 256][0] + (y-gridY) * dirs[(hashed + ANGLE*f) % 256][1];

	double distX = dblAbs((double)x-gridX), distY = dblAbs((double)y-gridY);	
	double polyX = 1 - 6*power(distX, 5) + 15*power(distX, 4) - 10*power(distX, 3);
	double polyY = 1 - 6*power(distY, 5) + 15*power(distY, 4) - 10*power(distY, 3);

	return polyX * polyY * grad;
}

__device__ double perlin(double x, double y, int perX, int perY, int f){
	double ans = 0;
	for(int i=0;i<4;i++){ //4 corners
		ans += Corners4(x, y, perX, perY, i, f);
	}
	return ans;
}

__global__ void pixelRate(int f, double *douimgptr){

	int perX = (int)((double)W*freq), perY = (int)((double)H*freq);
	int idx = blockIdx.x * blockDim.x + threadIdx.x;
	int xint = idx%W, yint = idx/W;
	double x = xint*freq, y = yint*freq;

	double ans = 0;
	for(int i=0;i<octs;i++){
		ans += power(0.5, i) * perlin(x*power(2, i), y*power(2, i), perX*power(2, i), perY*power(2, i), f);
	}
	
//	img[yint][xint] = ans; // between -1~1
//	img[yint][xint] = ((255.0 - (255.0+Yb)/2))*img[yint][xint] + ((255.0+Yb)/2); // between Ybound-255
//	intimgptr[yint*W + xint] = (uint8_t)img[yint][xint];

	douimgptr[yint*W + xint] = (1/2.0)*ans + (1/2.0); //0-1
}

__global__ void initdirs(){
	int idx = blockIdx.x * blockDim.x + threadIdx.x;
	dirs[idx][0] = cos((idx * 2.0 * PI)/256.0);
	dirs[idx][1] = sin((idx * 2.0 * PI)/256.0);
}
// interpolate two colors
__global__ void linearInter(double *douimgptr, double C0, double C1, uint8_t *intimgptr, int r){
	int idx = blockIdx.x * blockDim.x + threadIdx.x;
	int xint = idx%W, yint = idx/W;

	if(yint%r == 0 and xint%r == 0){
		int th = (yint/r)*(W/r) + (xint/r);
		intimgptr[th] = (uint8_t)(C0 + (C1-C0) * douimgptr[idx]);
	}
}
// paint N colors, the color painted is decided by the rate between 0-1
__global__ void NColor(double *douimgptr, double *colorListGPU, uint8_t *intimgptr, int r, int cstart, int cN){
	int idx = blockIdx.x * blockDim.x + threadIdx.x;
	int xint = idx%W, yint = idx/W;

	if(yint%r == 0 and xint%r == 0){
		int th = (yint/r)*(W/r) + (xint/r);
		intimgptr[th] = (uint8_t) (colorListGPU[ cstart*cN + (int)(floor(douimgptr[idx] * cN)) ]);
	}
}

struct Lab2VideoGenerator::Impl {
	int t = 1;
	int f = 1;
};

Lab2VideoGenerator::Lab2VideoGenerator(): impl(new Impl) {
	initdirs<<<1, 256>>>();
}

Lab2VideoGenerator::~Lab2VideoGenerator() {}

void Lab2VideoGenerator::get_info(Lab2VideoInfo &info) {
	info.w = W;
	info.h = H;
	info.n_frame = NFRAME;
	info.fps_n = 96;
	info.fps_d = 1;
};

void Lab2VideoGenerator::Generate(uint8_t *yuv) {

	double *douimgptr;
	cudaMalloc((void **) &douimgptr, H*W*sizeof(double));

	pixelRate<<<((H*W)/32)+1, 32>>>((impl->f), douimgptr);
	cudaDeviceSynchronize();
/*
	// Ncolor version
	int cN = 10;
	double *colorList = (double *)malloc(3*cN*sizeof(double));
	for(int i=0; i<3*cN; i+=3){
		double R = rand()%256;
		double G = rand()%256;
		double B = rand()%256;
		colorList[i]   =  0.299*R +   0.587 *G +   0.114 *B;
		colorList[i+1] = -0.169*R + (-0.331)*G +   0.500 *B + 128;
		colorList[i+2] =  0.500*R + (-0.419)*G + (-0.081)*B + 128;
	}
	double *colorListGPU;
	cudaMalloc((void **) &colorListGPU, 3*cN*sizeof(double));
	cudaMemcpy(colorListGPU, colorList, 3*cN*sizeof(double), cudaMemcpyHostToDevice);
*/
	uint8_t *intimgptr;
	cudaMalloc((void **) &intimgptr, H*W*sizeof(uint8_t));

/*	
	// save as an image
	uint8_t *hostimg = (uint8_t *)malloc(H*W*sizeof(uint8_t));
	cudaMemcpy(hostimg, intimgptr, H*W*sizeof(uint8_t), cudaMemcpyDeviceToHost);
	cv::imwrite("Result.png", cv::Mat(H, W, CV_8UC1, hostimg));
*/
/* 
	1) Ncolor: paint with cN random colors (should uncomment sections about Ncolor)
	2) linearInter: linear interpolate between two colors
	
*/

//	NColor<<<((H*W)/32)+1, 32>>>(douimgptr, colorListGPU, intimgptr, 1, 0, cN);
	linearInter<<<((H*W)/32)+1, 32>>>(douimgptr, Y0, Y1, intimgptr, 1);
	cudaMemcpy(yuv, intimgptr, H*W, cudaMemcpyDeviceToDevice); 
	cudaDeviceSynchronize();

//	NColor<<<((H*W)/32)+1, 32>>>(douimgptr, colorListGPU, intimgptr, 2, 1, cN);
	linearInter<<<((H*W)/32)+1, 32>>>(douimgptr, U0, U1, intimgptr, 2);
	cudaMemcpy(yuv+(H*W), intimgptr, H*W/4, cudaMemcpyDeviceToDevice);
	cudaDeviceSynchronize();

//	NColor<<<((H*W)/32)+1, 32>>>(douimgptr, colorListGPU, intimgptr, 2, 2, cN);
	linearInter<<<((H*W)/32)+1, 32>>>(douimgptr, V0, V1, intimgptr, 2);
	cudaMemcpy(yuv+(H*W)+(H*W)/4, intimgptr, H*W/4, cudaMemcpyDeviceToDevice);
	cudaDeviceSynchronize();

//	if((impl->t % 24 == 0)){ ++(impl->f); }
	++(impl->t);
	++(impl->f);
	
}
