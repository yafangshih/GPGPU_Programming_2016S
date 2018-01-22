#include <iostream>
#include <cstdlib>
#include <cmath>       /* cos */
#include <opencv2/opencv.hpp>

#define PI 3.14159265
#define E 2.71828182

using namespace std;
using namespace cv;

static const unsigned W = 640;
static const unsigned H = 480;
static const int NFRAME = 1;
static const int ANGLE = 10;

static const int octs = 5;

//const double freq = (double)1/(double)32;
const double gridW = 128;
const double gridH = 128;

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
		138,236,205,93,222,114,67,29,24,72,243,141,128,195,78,66,215,61,156,180 };

double dirs[256][2];

double surflet(double x, double y, int perX, int perY, int c, int f){
	
	int gridX = (int)x + c%2, gridY = (int)y + c/2;
	int hashed = perm[ (perm[ (gridX%perX)%256 ] + gridY%perY)%256];
	double grad = (x-gridX) * dirs[(hashed + ANGLE*f) % 256][0] + (y-gridY) * dirs[(hashed + ANGLE*f) % 256][1];

	double distX = abs((double)x-gridX), distY = abs((double)y-gridY);	
	double polyX = 1 - 6*pow(distX, 5) + 15*pow(distX, 4) - 10*pow(distX, 3);
	double polyY = 1 - 6*pow(distY, 5) + 15*pow(distY, 4) - 10*pow(distY, 3);

	return polyX * polyY * grad;
}

double perlin(double x, double y, int perX, int perY, int f){

	return (surflet(x, y, perX, perY, 0, f) + surflet(x, y, perX, perY, 1, f) + surflet(x, y, perX, perY, 2, f) + surflet(x, y, perX, perY, 3, f));
}

double fBm(double x, double y, int perX, int perY, int f){
	double ans = 0;

	for(int i=0;i<octs;i++){
		ans += pow(0.5, i) * perlin(x*pow(2, i), y*pow(2, i), perX*pow(2, i), perY*pow(2, i), f);
	}
	return ans;
}

int main(){

	Mat image(H, W, CV_64FC1);

	for(int i=0; i<256; i++){
		dirs[i][0] = cos((i * 2. * PI)/255);
		dirs[i][1] = sin((i * 2. * PI)/255);
	}
	
	double perX = W / gridW, perY = H / gridH;
		
	for(int f=0; f<NFRAME; f++){
		for(int y=0; y<H; y++){
			for(int x=0; x<W; x++){
				double tmp = fBm(x / gridW, y / gridH, perX, perY, f);
				tmp = (255.0/2) * tmp + (255.0/2);
				image.at<double>(y, x) = tmp;
			}
		}
		cv::imwrite("test.png", image);
	}
	return 0;

}
