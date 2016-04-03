#include <stdio.h>
#include <stdlib.h>
#include <math.h>       /* cos */
#include <opencv2/opencv.hpp>

#define PI 3.14159265

static const unsigned W = 640;
static const unsigned H = 480;

const double freq = (double)1/(double)32;

double img[H][W];

int perm[] = { 151,160,137,91,90,15, 
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
		138,236,205,93,222,114,67,29,24,72,243,141,128,195,78,66,215,61,156,180, 
		151,160,137,91,90,15, 
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

double dirs[256][2];

double power(double x, int y){
	double ans = 1;
	for(int i=0; i<y; i++){ ans = ans * x; }
	return ans;
}

double myabs(double x){
	if(x < 0){ return -x; }
	else{return x;}
}

double surflet(double x, double y, int perX, int perY, int c){
	
	int gridX = (int)x + c%2, gridY = (int)y + c/2;

	double distX = myabs((double)x-gridX), distY = myabs((double)y-gridY);
	
	double polyX = 1 - 6*power(distX, 5) + 15*power(distX, 4) - 10*power(distX, 3);
	double polyY = 1 - 6*power(distY, 5) + 15*power(distY, 4) - 10*power(distY, 3);

	int hashed = perm[ perm[ gridX%perX ] + gridY%perY];

	double grad = (x-gridX) * dirs[hashed][0] + (y-gridY) * dirs[hashed][1];

	return polyX * polyY * grad;
}

double perlin(double x, double y, int perX, int perY){

	return (surflet(x, y, perX, perY, 0) + surflet(x, y, perX, perY, 1) + surflet(x, y, perX, perY, 2) + surflet(x, y, perX, perY, 3));
}

double fBm(double x, double y, int perX, int perY, int octs){
	double ans = 0;

	for(int i=0;i<octs;i++){
		ans += power(0.5, i) * perlin(x*power(2, i), y*power(2, i), perX*power(2, i), perY*power(2, i));
	}
	return ans;
}

int main(){
	// init dirs
	for(int i=0; i<256; i++){
		dirs[i][0] = cos((i * 2.0 * PI)/256);
		dirs[i][1] = sin((i * 2.0 * PI)/256);
	}
	int octs = 5;
	double perX = (double)W*freq, perY = (double)H*freq;

	for(int y=0; y<H; y++){
		for(int x=0; x<W; x++){
			img[y][x] = fBm(x*freq, y*freq, (int)perX, (int)perY, octs); //perlin(x*freq, y*freq, (int)perX, (int)perY);
			img[y][x] = (255.0/2)*img[y][x] + (255.0/2);
		}
	}
	cv::imwrite("result.png", cv::Mat(H, W, CV_64FC1, img));
	return 0;

}
