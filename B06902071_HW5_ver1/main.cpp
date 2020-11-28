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

int octo_kernel_pattern[5][5] = {
	{0, 1, 1, 1, 0},
	{1, 1, 1, 1, 1},
	{1, 1, 1, 1, 1},
	{1, 1, 1, 1, 1},
	{0, 1, 1, 1, 0}
};

int L_shape_kernel_pattern[3][3] = {
	{0, 0, 0},
	{1, 1, 0},
	{0, 1, 0}
};

int L_shape_kernel_pattern_complement[3][3] = {
	{0, 1, 1},
	{0, 0, 1},
	{0, 0, 0}
};

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

void Dilation(Mat &img, Mat src) {
	for (int i = 0; i < img.rows; i++)
		for (int j = 0; j < img.cols; j++) {
			int localmax = 0;
			for (int i2 = 1; i2 <= 5; i2++) {
				for (int j2 = 1; j2 <= 5; j2++) {
					if (isvalid(img.rows, img.cols, i2 + i - 3, j2 + j - 3)) {
						if (localmax < src.at<uchar>(i2 + i - 3, j2 + j - 3)) {
							localmax = src.at<uchar>(i2 + i - 3, j2 + j - 3);
						}
					}
				}
			}
			img.at<uchar>(i, j) = localmax;
		}
}

void Erosion(Mat &img, Mat src) {
	for (int i = 0; i < img.rows; i++)
		for (int j = 0; j < img.cols; j++) {
			int flag = 1;
			int localmin = 255;
			for (int i2 = 1; i2 <= 5 && flag == 1; i2++) {
				for (int j2 = 1; j2 <= 5 && flag == 1; j2++) {
					if (isvalid(img.rows, img.cols, i2 + i - 3, j2 + j - 3)) {
						if (octo_kernel_pattern[i2 - 1][j2 - 1] == 1 && src.at<uchar>(i2 + i - 3, j2 + j - 3) == 0) {
							flag = 0;
						}
						else {
							if (localmin > src.at<uchar>(i2 + i - 3, j2 + j - 3)) {
								localmin = src.at<uchar>(i2 + i - 3, j2 + j - 3);
							}
						}
					}
				}
			}
			if (flag == 1) {
				img.at<uchar>(i, j) = localmin;
			}
		}
}

void HitandMiss(Mat &img) {
	Mat img_complement;
	img.copyTo(img_complement);
	for (int i = 0; i < img.rows; i++)
		for (int j = 0; j < img.cols; j++) {
			if (img_complement.at<uchar>(i, j) >= 128) {
				img_complement.at<uchar>(i, j) = 0;
			}
			else {
				img_complement.at<uchar>(i, j) = 255;
			}
		}

	Mat src_img;
	img.copyTo(src_img);
	for (int i = 0; i < img.rows; i++)
		for (int j = 0; j < img.cols; j++) {
			int flag = 1;
			for (int i2 = 0; i2 < 3 && flag == 1; i2++) {
				for (int j2 = 0; j2 < 3 && flag == 1; j2++) {
					if (isvalid(img.rows, img.cols, i2 + i - 1, j2 + j - 1)) {
						if (L_shape_kernel_pattern[i2][j2] == 1 && src_img.at<uchar>(i2 + i - 1, j2 + j - 1) == 0) {
							flag = 0;
						}
					}
				}
			}
			if (flag == 0) {
				img.at<uchar>(i, j) = 0;
			}
			else {
				img.at<uchar>(i, j) = 255;
			}
		}

	Mat src_img_complement;
	img_complement.copyTo(src_img_complement);
	for (int i = 0; i < img_complement.rows; i++)
		for (int j = 0; j < img_complement.cols; j++) {
			int flag = 1;
			for (int i2 = 0; i2 < 3 && flag == 1; i2++) {
				for (int j2 = 0; j2 < 3 && flag == 1; j2++) {
					if (isvalid(img_complement.rows, img_complement.cols, i2 + i - 1, j2 + j - 1)) {
						if (L_shape_kernel_pattern_complement[i2][j2] == 1 && src_img_complement.at<uchar>(i2 + i - 1, j2 + j - 1) == 0) {
							flag = 0;
						}
					}
				}
			}
			if (flag == 0) {
				img_complement.at<uchar>(i, j) = 0;
			}
			else {
				img_complement.at<uchar>(i, j) = 255;
			}
		}
	for (int i = 0; i < img.rows; i++)
		for (int j = 0; j < img.cols; j++) {
			if (img.at<uchar>(i, j) >= 128 && img_complement.at<uchar>(i, j) >= 128) {
				img.at<uchar>(i, j) = 255;
			}
			else {
				img.at<uchar>(i, j) = 0;
			}
		}
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
	//img_binary128(img_bin);
	//imshow("binarize at 128", img_bin);
	//imwrite("binarize at 128.jpg", img_bin);

	//dilation
	Mat img_dilation;
	img_bin.copyTo(img_dilation);
	Dilation(img_dilation, img_bin);
	imshow("Dilation", img_dilation);
	imwrite("Dilation.jpg", img_dilation);
	
	//erosion
	Mat img_erosion;
	img_bin.copyTo(img_erosion);
	Erosion(img_erosion, img_bin);
	imshow("Erosion", img_erosion);
	imwrite("Erosion.jpg", img_erosion);

	//opening
	Mat img_opening;
	img_erosion.copyTo(img_opening);
	Dilation(img_opening, img_erosion);
	imshow("Opening", img_opening);
	imwrite("Opening.jpg", img_opening);

	//closing
	Mat img_closing;
	img_dilation.copyTo(img_closing);
	Erosion(img_closing, img_dilation);
	imshow("Closing", img_closing);
	imwrite("Closing.jpg", img_closing);

	//hitandmiss
	//Mat img_hitandmiss;
	//img_bin.copyTo(img_hitandmiss);
	//HitandMiss(img_hitandmiss);
	//imshow("hitandmiss", img_hitandmiss);
	//imwrite("hitandmiss.jpg", img_hitandmiss);

	waitKey(0);
	return 0;
}