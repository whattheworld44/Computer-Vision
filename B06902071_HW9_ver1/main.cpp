#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <algorithm>
#include <vector>
#include <random>

#include<opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>

using namespace cv;
using namespace std;

bool isvalid(int constRow, int constCol, int Row, int Col) {
	if (Row < 0 || Row >= constRow) {
		return false;
	}
	if (Col < 0 || Col >= constCol) {
		return false;
	}
	return true;
}

void Roberts_operator(Mat &img, Mat src, int threshold) {
	for (int i = 0; i < img.rows; i++) {
		for (int j = 0; j < img.cols; j++) {
			int temp[2][2];
			for (int k = 0; k < 2; k++) {
				for (int l = 0; l < 2; l++) {
					if (isvalid(img.rows, img.cols, i + k, j + l)) {
						temp[k][l] = src.at<uchar>(i + k, j + l);
					}
					else {
						temp[k][l] = 0;
					}
				}
			}
			int result = sqrt((-temp[0][0]+temp[1][1])*(-temp[0][0] + temp[1][1]) + 
				(-temp[0][1] + temp[1][0])*(-temp[0][1] + temp[1][0]));

			if (result >= threshold) {
				img.at<uchar>(i, j) = 0;
			}
			else {
				img.at<uchar>(i, j) = 255;
			}
		}
	}
}

void Prewitt_operator(Mat &img, Mat src, int threshold) {
	for (int i = 0; i < img.rows; i++) {
		for (int j = 0; j < img.cols; j++) {
			int temp[3][3];
			for (int k = 0; k < 3; k++) {
				for (int l = 0; l < 3; l++) {
					if (isvalid(img.rows, img.cols, i + k - 1, j + l - 1)) {
						temp[k][l] = src.at<uchar>(i + k - 1, j + l - 1);
					}
					else {
						int tr = i + k - 1;
						int tc = j + l - 1;
						if (isvalid(img.rows, img.cols, tr, tc + 1)) {
							temp[k][l] = src.at<uchar>(tr, tc + 1);
						}
						else if (isvalid(img.rows, img.cols, tr, tc - 1)) {
							temp[k][l] = src.at<uchar>(tr, tc - 1);
						}
						else if (isvalid(img.rows, img.cols, tr + 1, tc)) {
							temp[k][l] = src.at<uchar>(tr + 1, tc);
						}
						else if (isvalid(img.rows, img.cols, tr - 1, tc)) {
							temp[k][l] = src.at<uchar>(tr - 1, tc);
						}
						else if (isvalid(img.rows, img.cols, tr + 1, tc + 1)) {
							temp[k][l] = src.at<uchar>(tr + 1, tc + 1);
						}
						else if (isvalid(img.rows, img.cols, tr - 1, tc - 1)) {
							temp[k][l] = src.at<uchar>(tr - 1, tc - 1);
						}
						else if (isvalid(img.rows, img.cols, tr + 1, tc - 1)) {
							temp[k][l] = src.at<uchar>(tr + 1, tc - 1);
						}
						else if (isvalid(img.rows, img.cols, tr - 1, tc + 1)) {
							temp[k][l] = src.at<uchar>(tr - 1, tc + 1);
						}
					}
				}
			}
			int result = sqrt(
				(temp[2][0] + temp[2][1] + temp[2][2] - temp[0][0] - temp[0][1] - temp[0][2])*
				(temp[2][0] + temp[2][1] + temp[2][2] - temp[0][0] - temp[0][1] - temp[0][2]) +
				(temp[0][2] + temp[1][2] + temp[2][2] - temp[0][0] - temp[1][0] - temp[2][0])*
				(temp[0][2] + temp[1][2] + temp[2][2] - temp[0][0] - temp[1][0] - temp[2][0])
			);

			if (result >= threshold) {
				img.at<uchar>(i, j) = 0;
			}
			else {
				img.at<uchar>(i, j) = 255;
			}
		}
	}
}

void Sobel_operator(Mat &img, Mat src, int threshold) {
	for (int i = 0; i < img.rows; i++) {
		for (int j = 0; j < img.cols; j++) {
			int temp[3][3];
			for (int k = 0; k < 3; k++) {
				for (int l = 0; l < 3; l++) {
					if (isvalid(img.rows, img.cols, i + k - 1, j + l - 1)) {
						temp[k][l] = src.at<uchar>(i + k - 1, j + l - 1);
					}
					else {
						int tr = i + k - 1;
						int tc = j + l - 1;
						if (isvalid(img.rows, img.cols, tr, tc + 1)) {
							temp[k][l] = src.at<uchar>(tr, tc + 1);
						}
						else if (isvalid(img.rows, img.cols, tr, tc - 1)) {
							temp[k][l] = src.at<uchar>(tr, tc - 1);
						}
						else if (isvalid(img.rows, img.cols, tr + 1, tc)) {
							temp[k][l] = src.at<uchar>(tr + 1, tc);
						}
						else if (isvalid(img.rows, img.cols, tr - 1, tc)) {
							temp[k][l] = src.at<uchar>(tr - 1, tc);
						}
						else if (isvalid(img.rows, img.cols, tr + 1, tc + 1)) {
							temp[k][l] = src.at<uchar>(tr + 1, tc + 1);
						}
						else if (isvalid(img.rows, img.cols, tr - 1, tc - 1)) {
							temp[k][l] = src.at<uchar>(tr - 1, tc - 1);
						}
						else if (isvalid(img.rows, img.cols, tr + 1, tc - 1)) {
							temp[k][l] = src.at<uchar>(tr + 1, tc - 1);
						}
						else if (isvalid(img.rows, img.cols, tr - 1, tc + 1)) {
							temp[k][l] = src.at<uchar>(tr - 1, tc + 1);
						}
					}
				}
			}
			int result = sqrt(
				(temp[2][0] + 2*temp[2][1] + temp[2][2] - temp[0][0] - 2*temp[0][1] - temp[0][2])*
				(temp[2][0] + 2*temp[2][1] + temp[2][2] - temp[0][0] - 2*temp[0][1] - temp[0][2]) +
				(temp[0][2] + 2*temp[1][2] + temp[2][2] - temp[0][0] - 2*temp[1][0] - temp[2][0])*
				(temp[0][2] + 2*temp[1][2] + temp[2][2] - temp[0][0] - 2*temp[1][0] - temp[2][0])
			);

			if (result >= threshold) {
				img.at<uchar>(i, j) = 0;
			}
			else {
				img.at<uchar>(i, j) = 255;
			}
		}
	}
}

void FreiChen_operator(Mat &img, Mat src, int threshold) {
	for (int i = 0; i < img.rows; i++) {
		for (int j = 0; j < img.cols; j++) {
			int temp[3][3];
			for (int k = 0; k < 3; k++) {
				for (int l = 0; l < 3; l++) {
					if (isvalid(img.rows, img.cols, i + k - 1, j + l - 1)) {
						temp[k][l] = src.at<uchar>(i + k - 1, j + l - 1);
					}
					else {
						int tr = i + k - 1;
						int tc = j + l - 1;
						if (isvalid(img.rows, img.cols, tr, tc + 1)) {
							temp[k][l] = src.at<uchar>(tr, tc + 1);
						}
						else if (isvalid(img.rows, img.cols, tr, tc - 1)) {
							temp[k][l] = src.at<uchar>(tr, tc - 1);
						}
						else if (isvalid(img.rows, img.cols, tr + 1, tc)) {
							temp[k][l] = src.at<uchar>(tr + 1, tc);
						}
						else if (isvalid(img.rows, img.cols, tr - 1, tc)) {
							temp[k][l] = src.at<uchar>(tr - 1, tc);
						}
						else if (isvalid(img.rows, img.cols, tr + 1, tc + 1)) {
							temp[k][l] = src.at<uchar>(tr + 1, tc + 1);
						}
						else if (isvalid(img.rows, img.cols, tr - 1, tc - 1)) {
							temp[k][l] = src.at<uchar>(tr - 1, tc - 1);
						}
						else if (isvalid(img.rows, img.cols, tr + 1, tc - 1)) {
							temp[k][l] = src.at<uchar>(tr + 1, tc - 1);
						}
						else if (isvalid(img.rows, img.cols, tr - 1, tc + 1)) {
							temp[k][l] = src.at<uchar>(tr - 1, tc + 1);
						}
					}
				}
			}
			int result = sqrt(
				(temp[2][0] + sqrt(2) * temp[2][1] + temp[2][2] - temp[0][0] - sqrt(2) * temp[0][1] - temp[0][2])*
				(temp[2][0] + sqrt(2) * temp[2][1] + temp[2][2] - temp[0][0] - sqrt(2) * temp[0][1] - temp[0][2]) +
				(temp[0][2] + sqrt(2) * temp[1][2] + temp[2][2] - temp[0][0] - sqrt(2) * temp[1][0] - temp[2][0])*
				(temp[0][2] + sqrt(2) * temp[1][2] + temp[2][2] - temp[0][0] - sqrt(2) * temp[1][0] - temp[2][0])
			);

			if (result >= threshold) {
				img.at<uchar>(i, j) = 0;
			}
			else {
				img.at<uchar>(i, j) = 255;
			}
		}
	}
}

void Kirsch_operator(Mat &img, Mat src, int threshold) {
	for (int i = 0; i < img.rows; i++) {
		for (int j = 0; j < img.cols; j++) {
			int temp[3][3];
			for (int k = 0; k < 3; k++) {
				for (int l = 0; l < 3; l++) {
					if (isvalid(img.rows, img.cols, i + k - 1, j + l - 1)) {
						temp[k][l] = src.at<uchar>(i + k - 1, j + l - 1);
					}
					else {
						int tr = i + k - 1;
						int tc = j + l - 1;
						if (isvalid(img.rows, img.cols, tr, tc + 1)) {
							temp[k][l] = src.at<uchar>(tr, tc + 1);
						}
						else if (isvalid(img.rows, img.cols, tr, tc - 1)) {
							temp[k][l] = src.at<uchar>(tr, tc - 1);
						}
						else if (isvalid(img.rows, img.cols, tr + 1, tc)) {
							temp[k][l] = src.at<uchar>(tr + 1, tc);
						}
						else if (isvalid(img.rows, img.cols, tr - 1, tc)) {
							temp[k][l] = src.at<uchar>(tr - 1, tc);
						}
						else if (isvalid(img.rows, img.cols, tr + 1, tc + 1)) {
							temp[k][l] = src.at<uchar>(tr + 1, tc + 1);
						}
						else if (isvalid(img.rows, img.cols, tr - 1, tc - 1)) {
							temp[k][l] = src.at<uchar>(tr - 1, tc - 1);
						}
						else if (isvalid(img.rows, img.cols, tr + 1, tc - 1)) {
							temp[k][l] = src.at<uchar>(tr + 1, tc - 1);
						}
						else if (isvalid(img.rows, img.cols, tr - 1, tc + 1)) {
							temp[k][l] = src.at<uchar>(tr - 1, tc + 1);
						}
					}
				}
			}

			int result = 0;
			result = max(result, 5*temp[0][0] + 5*temp[0][1] + 5*temp[0][2]
				- 3* temp[1][0] - 3 * temp[2][0] - 3 * temp[2][1] - 3 * temp[2][2] - 3 * temp[1][2]
				);//up
			result = max(result, 5 * temp[2][0] + 5 * temp[2][1] + 5 * temp[2][2]
				- 3 * temp[1][0] - 3 * temp[0][0] - 3 * temp[0][1] - 3 * temp[0][2] - 3 * temp[1][2]
			);//down
			result = max(result, 5 * temp[0][0] + 5 * temp[1][0] + 5 * temp[2][0]
				- 3 * temp[0][1] - 3 * temp[0][2] - 3 * temp[1][2] - 3 * temp[2][2] - 3 * temp[2][1]
			);//left
			result = max(result, 5 * temp[0][2] + 5 * temp[1][2] + 5 * temp[2][2]
				- 3 * temp[2][1] - 3 * temp[2][0] - 3 * temp[1][0] - 3 * temp[0][0] - 3 * temp[0][1]
			);//right
			result = max(result, 5 * temp[1][0] + 5 * temp[0][0] + 5 * temp[0][1]
				- 3 * temp[0][2] - 3 * temp[1][2] - 3 * temp[2][2] - 3 * temp[2][1] - 3 * temp[2][0]
			);//up-left
			result = max(result, 5 * temp[1][0] + 5 * temp[2][0] + 5 * temp[2][1]
				- 3 * temp[2][2] - 3 * temp[1][2] - 3 * temp[0][2] - 3 * temp[0][1] - 3 * temp[0][0]
			);//down-left
			result = max(result, 5 * temp[0][1] + 5 * temp[0][2] + 5 * temp[1][2]
				- 3 * temp[2][2] - 3 * temp[2][1] - 3 * temp[2][0] - 3 * temp[1][0] - 3 * temp[0][0]
			);//up-right
			result = max(result, 5 * temp[1][2] + 5 * temp[2][2] + 5 * temp[2][1]
				- 3 * temp[2][0] - 3 * temp[1][0] - 3 * temp[0][0] - 3 * temp[0][1] - 3 * temp[0][2]
			);//down-right

			if (result >= threshold) {
				img.at<uchar>(i, j) = 0;
			}
			else {
				img.at<uchar>(i, j) = 255;
			}
		}
	}
}

void Robinson_operator(Mat &img, Mat src, int threshold) {
	for (int i = 0; i < img.rows; i++) {
		for (int j = 0; j < img.cols; j++) {
			int temp[3][3];
			for (int k = 0; k < 3; k++) {
				for (int l = 0; l < 3; l++) {
					if (isvalid(img.rows, img.cols, i + k - 1, j + l - 1)) {
						temp[k][l] = src.at<uchar>(i + k - 1, j + l - 1);
					}
					else {
						int tr = i + k - 1;
						int tc = j + l - 1;
						if (isvalid(img.rows, img.cols, tr, tc + 1)) {
							temp[k][l] = src.at<uchar>(tr, tc + 1);
						}
						else if (isvalid(img.rows, img.cols, tr, tc - 1)) {
							temp[k][l] = src.at<uchar>(tr, tc - 1);
						}
						else if (isvalid(img.rows, img.cols, tr + 1, tc)) {
							temp[k][l] = src.at<uchar>(tr + 1, tc);
						}
						else if (isvalid(img.rows, img.cols, tr - 1, tc)) {
							temp[k][l] = src.at<uchar>(tr - 1, tc);
						}
						else if (isvalid(img.rows, img.cols, tr + 1, tc + 1)) {
							temp[k][l] = src.at<uchar>(tr + 1, tc + 1);
						}
						else if (isvalid(img.rows, img.cols, tr - 1, tc - 1)) {
							temp[k][l] = src.at<uchar>(tr - 1, tc - 1);
						}
						else if (isvalid(img.rows, img.cols, tr + 1, tc - 1)) {
							temp[k][l] = src.at<uchar>(tr + 1, tc - 1);
						}
						else if (isvalid(img.rows, img.cols, tr - 1, tc + 1)) {
							temp[k][l] = src.at<uchar>(tr - 1, tc + 1);
						}
					}
				}
			}

			int result = 0;
			int r[8] = { 0, 0, 0, 1, 2, 2, 2, 1 };
			int c[8] = { 0, 1, 2, 2, 2, 1, 0, 0 };
			for (int m = 0; m < 8; m++) {
				result = max(result,
					1 * temp[r[(0+m)%8]][c[(0+m)%8]] + 2 *temp[r[(1+m)%8]][c[(1+m)%8]] + 1 * temp[r[(2+m)%8]][c[(2+m)%8]]
					- 1 * temp[r[(4+m)%8]][c[(4+m)%8]] - 2 * temp[r[(5+m)%8]][c[(5+m)%8]] - 1 * temp[r[(6+m)%8]][c[(6+m)%8]]
				);
			}

			if (result >= threshold) {
				img.at<uchar>(i, j) = 0;
			}
			else {
				img.at<uchar>(i, j) = 255;
			}
		}
	}
}

void NevatiaBabu_operator(Mat &img, Mat src, int threshold) {
	for (int i = 0; i < img.rows; i++) {
		for (int j = 0; j < img.cols; j++) {
			int temp[5][5];
			for (int k = 0; k < 5; k++) {
				for (int l = 0; l < 5; l++) {
					if (isvalid(img.rows, img.cols, i + k - 2, j + l - 2)) {
						temp[k][l] = src.at<uchar>(i + k - 2, j + l - 2);
					}
					else {
						int tr = i + k - 2;
						int tc = j + l - 2;
						if (isvalid(img.rows, img.cols, tr, tc + 1)) {
							temp[k][l] = src.at<uchar>(tr, tc + 1);
						}
						else if (isvalid(img.rows, img.cols, tr, tc + 2)) {
							temp[k][l] = src.at<uchar>(tr, tc + 2);
						}
						else if (isvalid(img.rows, img.cols, tr, tc - 1)) {
							temp[k][l] = src.at<uchar>(tr, tc - 1);
						}
						else if (isvalid(img.rows, img.cols, tr, tc - 2)) {
							temp[k][l] = src.at<uchar>(tr, tc - 2);
						}
						else if (isvalid(img.rows, img.cols, tr + 1, tc)) {
							temp[k][l] = src.at<uchar>(tr + 1, tc);
						}
						else if (isvalid(img.rows, img.cols, tr + 2, tc)) {
							temp[k][l] = src.at<uchar>(tr + 2, tc);
						}
						else if (isvalid(img.rows, img.cols, tr - 1, tc)) {
							temp[k][l] = src.at<uchar>(tr - 1, tc);
						}
						else if (isvalid(img.rows, img.cols, tr - 2, tc)) {
							temp[k][l] = src.at<uchar>(tr - 2, tc);
						}
						else if (isvalid(img.rows, img.cols, tr + 1, tc + 1)) {
							temp[k][l] = src.at<uchar>(tr + 1, tc + 1);
						}
						else if (isvalid(img.rows, img.cols, tr + 2, tc + 2)) {
							if (isvalid(img.rows, img.cols, tr + 2, tc + 2 - 1)) {
								temp[k][l] = src.at<uchar>(tr + 2, tc + 2 - 1);
							}
							else if (isvalid(img.rows, img.cols, tr + 2 - 1, tc + 2)) {
								temp[k][l] = src.at<uchar>(tr + 2 - 1, tc + 2);
							}
							else {
								temp[k][l] = src.at<uchar>(tr + 2, tc + 2);
							}
							
						}
						else if (isvalid(img.rows, img.cols, tr - 1, tc - 1)) {
							temp[k][l] = src.at<uchar>(tr - 1, tc - 1);
						}
						else if (isvalid(img.rows, img.cols, tr - 2, tc - 2)) {
							if (isvalid(img.rows, img.cols, tr - 2 + 1, tc - 2)) {
								temp[k][l] = src.at<uchar>(tr - 2 + 1, tc - 2);
							}
							else if (isvalid(img.rows, img.cols, tr - 2, tc - 2 + 1)) {
								temp[k][l] = src.at<uchar>(tr - 2, tc - 2 + 1);
							}
							else {
								temp[k][l] = src.at<uchar>(tr - 2, tc - 2);
							}
							
						}
						else if (isvalid(img.rows, img.cols, tr + 1, tc - 1)) {
							temp[k][l] = src.at<uchar>(tr + 1, tc - 1);
						}
						else if (isvalid(img.rows, img.cols, tr + 2, tc - 2)) {
							if (isvalid(img.rows, img.cols, tr + 2, tc - 2 + 1)) {
								temp[k][l] = src.at<uchar>(tr + 2, tc - 2 + 1);
							}
							else if (isvalid(img.rows, img.cols, tr + 2 - 1, tc - 2)) {
								temp[k][l] = src.at<uchar>(tr + 2 - 1, tc - 2);
							}
							else {
								temp[k][l] = src.at<uchar>(tr + 2, tc - 2);
							}
							
						}
						else if (isvalid(img.rows, img.cols, tr - 1, tc + 1)) {
							temp[k][l] = src.at<uchar>(tr - 1, tc + 1);
						}
						else if (isvalid(img.rows, img.cols, tr - 2, tc + 2)) {
							if (isvalid(img.rows, img.cols, tr - 2, tc + 2 - 1)) {
								temp[k][l] = src.at<uchar>(tr - 2 , tc + 2 - 1);
							}
							else if (isvalid(img.rows, img.cols, tr - 2 + 1, tc + 2)) {
								temp[k][l] = src.at<uchar>(tr - 2 + 1, tc + 2);
							}
							else {
								temp[k][l] = src.at<uchar>(tr - 2, tc + 2);
							}
							
						}
					}
				}
			}

			int result = 0;
			int kernel[6][5][5] = {
				{	{100, 100, 100, 100, 100},
					{100, 100, 100, 100, 100},
					{0, 0, 0, 0, 0},
					{-100, -100, -100, -100, -100},
					{-100, -100, -100, -100, -100}
				},
				{	{100, 100, 100, 100, 100},
					{100, 100, 100, 78, -32},
					{100, 92, 0, -92, -100},
					{32, -78, -100, -100, -100},
					{-100, -100, -100, -100, -100}
				},
				{	{100, 100, 100, 32, -100},
					{100, 100, 92, -78, -100},
					{100, 100, 0, -100, -100},
					{100, 78, -92, -100, -100},
					{100, -32, -100, -100, -100}
				},
				{	{-100, -100, 0, 100, 100},
					{-100, -100, 0, 100, 100},
					{-100, -100, 0, 100, 100},
					{-100, -100, 0, 100, 100},
					{-100, -100, 0, 100, 100}
				},
				{	{-100, 32, 100, 100, 100},
					{-100, -78, 92, 100, 100},
					{-100, -100, 0, 100, 100},
					{-100, -100, -92, 78, 100},
					{-100, -100, -100, -32, 100}
				},
				{	{100, 100, 100, 100, 100},
					{-32, 78, 100, 100, 100},
					{-100, -92, 0, 92, 100},
					{-100, -100, -100, -78, -32},
					{-100, -100, -100, -100, -100}
				}
			};
			for (int k = 0; k < 6; k++) {
				int temp_result = 0;
				for (int l = 0; l < 5; l++) {
					for (int m = 0; m < 5; m++) {
						temp_result = temp_result + kernel[k][l][m] * temp[l][m];
					}
				}
				result = max(result, temp_result);
			}
			if (result >= threshold) {
				img.at<uchar>(i, j) = 0;
			}
			else {
				img.at<uchar>(i, j) = 255;
			}
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
	
	//random_device rd;
	//mt19937 gen(rd());
	//normal_distribution<> distr(0, 1); // mean stddev
	//uniform_real_distribution<> distr2(0, 1);

	//Robert's Operator: 12
	Mat img_roberts;
	img.copyTo(img_roberts);
	Roberts_operator(img_roberts, img, 12);
	imshow("img_roberts", img_roberts);
	imwrite("img_roberts.jpg", img_roberts);

	//Prewitt's Edge Detector: 24
	Mat img_prewitt;
	img.copyTo(img_prewitt);
	Prewitt_operator(img_prewitt, img, 24);
	imshow("img_prewitt", img_prewitt);
	imwrite("img_prewitt.jpg", img_prewitt);

	//Sobel's Edge Detector: 38
	Mat img_Sobel;
	img.copyTo(img_Sobel);
	Sobel_operator(img_Sobel, img, 38);
	imshow("img_Sobel", img_Sobel);
	imwrite("img_Sobel.jpg", img_Sobel);

	//Frei and Chen's Gradient Operator: 30
	Mat img_FreiChen;
	img.copyTo(img_FreiChen);
	FreiChen_operator(img_FreiChen, img, 30);
	imshow("img_FreiChen", img_FreiChen);
	imwrite("img_FreiChen.jpg", img_FreiChen);

	//Kirsch's Compass Operator: 135
	Mat img_Kirsch;
	img.copyTo(img_Kirsch);
	Kirsch_operator(img_Kirsch, img, 135);
	imshow("img_Kirsch", img_Kirsch);
	imwrite("img_Kirsch.jpg", img_Kirsch);

	//Robinson's Compass Operator: 43
	Mat img_Robinson;
	img.copyTo(img_Robinson);
	Robinson_operator(img_Robinson, img, 43);
	imshow("img_Robinson", img_Robinson);
	imwrite("img_Robinson.jpg", img_Robinson);

	//Nevatia-Babu 5x5 Operator: 12500
	Mat img_NevatiaBabu;
	img.copyTo(img_NevatiaBabu);
	NevatiaBabu_operator(img_NevatiaBabu, img, 12500);
	imshow("img_NevatiaBabu", img_NevatiaBabu);
	imwrite("img_NevatiaBabu.jpg", img_NevatiaBabu);

	waitKey(0);

	return 0;
}