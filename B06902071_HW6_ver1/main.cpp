#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <algorithm>
#include <vector>

#include<opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>

using namespace cv;
using namespace std;

bool isvalid(int cRow, int cCol, int Row, int Col) {
	if (Row < 0 || Row >= cRow) {
		return false;
	}
	if (Col < 0 || Col >= cCol) {
		return false;
	}
	return true;
}

void img_binary128(Mat &img){
	for (int i = 0; i < img.rows; i++)
		for (int j = 0; j < img.cols; j++) {
			if (img.at<uchar>(i, j) >= 128) {
				img.at<uchar>(i, j) = 255;
			}
			else {
				img.at<uchar>(i, j) = 0;
			}
		}
}

void downSample(Mat &img, Mat &output, int num) {
	for (int i = 0; i < output.rows; i++)
		for (int j = 0; j < output.cols; j++) {
			output.at<uchar>(i, j) = img.at<uchar>(i*num, j*num);
		}
}

char yokoi_h(int b, int c, int d, int e) {
	if (b == c && (d != b || e != b)) {
		return 'q';
	}
	if (b == c && (d == b && e == b)) {
		return 'r';
	}
	return 's';
}

int yokoi_f(char a1, char a2, char a3, char a4) {
	if (a1 == a2 && a2 == a3 && a3 == a4 && a4 == 'r') {
		return 5;
	}
	
	int total = 0;
	if (a1 == 'q') {
		total += 1;
	}
	if (a2 == 'q') {
		total += 1;
	}
	if (a3 == 'q') {
		total += 1;
	}
	if (a4 == 'q') {
		total += 1;
	}

	return total;
}

int main()
{
	// read image 
	Mat img = imread("lena.bmp", CV_8UC1);
	//imshow("lena.bmp", img);
	//imwrite("lena.jpg", img);
	
	// sample code : Lomo Effect
	int center_i = img.rows / 2;
	int center_j = img.cols / 2;
	double max_dist = sqrt(center_i*center_i + center_j*center_j);

	//original
	//imshow("img", img);
	//imwrite("img.jpg", img);
	

	//binarize
	Mat img_bin;
	img.copyTo(img_bin);
	img_binary128(img_bin);
	//imshow("binarize at 128", img_bin);
	//imwrite("binarize at 128.jpg", img_bin);


	//downSample
	Mat img_binx64(64, 64, CV_8UC1);
	downSample(img_bin, img_binx64, 8);
	//imshow("downSample as 64x64", img_binx64);
	//imwrite("downSample as 64x64.jpg", img_binx64);


	//yokoi connectivity
	for (int i = 0; i < img_binx64.rows; i++) {
		for (int j = 0; j < img_binx64.cols; j++) {
			if (img_binx64.at<uchar>(i, j) == 255) {
				int x[9] = {0};
				if (isvalid(img_binx64.rows, img_binx64.cols, i, j)) // x0
					x[0] = (int)img_binx64.at<uchar>(i, j);
				if (isvalid(img_binx64.rows, img_binx64.cols, i, j+1)) // x1
					x[1] = (int)img_binx64.at<uchar>(i, j+1);
				if (isvalid(img_binx64.rows, img_binx64.cols, i-1, j)) // x2
					x[2] = (int)img_binx64.at<uchar>(i-1, j);
				if (isvalid(img_binx64.rows, img_binx64.cols, i, j-1)) // x3
					x[3] = (int)img_binx64.at<uchar>(i, j-1);
				if (isvalid(img_binx64.rows, img_binx64.cols, i+1, j)) // x4
					x[4] = (int)img_binx64.at<uchar>(i+1, j);
				if (isvalid(img_binx64.rows, img_binx64.cols, i+1, j+1)) // x5
					x[5] = (int)img_binx64.at<uchar>(i+1, j+1);
				if (isvalid(img_binx64.rows, img_binx64.cols, i-1, j+1)) // x6
					x[6] = (int)img_binx64.at<uchar>(i-1, j+1);
				if (isvalid(img_binx64.rows, img_binx64.cols, i-1, j-1)) // x7
					x[7] = (int)img_binx64.at<uchar>(i-1, j-1);
				if (isvalid(img_binx64.rows, img_binx64.cols, i+1, j-1)) // x8
					x[8] = (int)img_binx64.at<uchar>(i+1, j-1);

				int result = yokoi_f(yokoi_h(x[0], x[1], x[6], x[2]),
									yokoi_h(x[0], x[2], x[7], x[3]),
									yokoi_h(x[0], x[3], x[8], x[4]),
									yokoi_h(x[0], x[4], x[5], x[1]));
				if(result != 0)
					printf("%d", result);
				else
					printf(" ");
			}
			else {
				printf(" ");
			}
			
		}
		printf("\n");
	}
		
	imshow("downSample as 64x64", img_binx64);
	waitKey(0);
	return 0;
}