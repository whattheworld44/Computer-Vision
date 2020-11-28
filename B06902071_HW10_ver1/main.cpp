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

int kernel1[3][3] = {
	{0, 1, 0},
	{1, -4, 1},
	{0, 1, 0}
};

int kernel2[3][3] = {
	{1, 1, 1},
	{1, -8, 1},
	{1, 1, 1}
};

int kernel3[3][3] = {
	{2, -1, 2},
	{-1, -4, -1},
	{2, -1, 2}
};

int kernel4[11][11] = {
	{0, 0, 0, -1, -1, -2, -1, -1, 0, 0, 0},
	{0, 0, -2, -4, -8, -9, -8, -4, -2, 0, 0},
	{0, -2, -7, -15, -22, -23, -22, -15, -7, -2, 0},
	{-1, -4, -15, -24, -14, -1, -14, -24, -15, -4, -1},
	{-1, -8, -22, -14, 52, 103, 52, -14, -22, -8, -1},
	{-2, -9, -23, -1, 103, 178, 103, -1, -23, -9, -2},
	{-1, -8, -22, -14, 52, 103, 52, -14, -22, -8, -1},
	{-1, -4, -15, -24, -14, -1, -14, -24, -15, -4, -1},
	{0, -2, -7, -15, -22, -23, -22, -15, -7, -2, 0},
	{0, 0, -2, -4, -8, -9, -8, -4, -2, 0, 0},
	{0, 0, 0, -1, -1, -2, -1, -1, 0, 0, 0}
};

int kernel5[11][11] = {
	{-1, -3, -4, -6, -7, -8, -7, -6, -4, -3, -1},
	{-3, -5, -8, -11, -13, -13, -13, -11, -8, -5, -3},
	{-4, -8, -12, -16, -17, -17, -17, -16, -12, -8, -4},
	{-6, -11, -16, -16, 0, 15, 0, -16, -16, -11, -6},
	{-7, -13, -17, 0, 85, 160, 85, 0, -17, -13, -7},
	{-8, -13, -17, 15, 160, 283, 160, 15, -17, -13, -8},
	{-7, -13, -17, 0, 85, 160, 85, 0, -17, -13, -7},
	{-6, -11, -16, -16, 0, 15, 0, -16, -16, -11, -6},
	{-4, -8, -12, -16, -17, -17, -17, -16, -12, -8, -4},
	{-3, -5, -8, -11, -13, -13, -13, -11, -8, -5, -3},
	{-1, -3, -4, -6, -7, -8, -7, -6, -4, -3, -1}
};

bool isvalid(int constRow, int constCol, int Row, int Col) {
	if (Row < 0 || Row >= constRow) {
		return false;
	}
	if (Col < 0 || Col >= constCol) {
		return false;
	}
	return true;
}

void zco(Mat &img, Mat src) {
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
			for (int k = 0; k < 3; k++) {
				for (int l = 0; l < 3; l++) {
					if (temp[k][l] == 2) {
						result = -1;
					}

				}
			}

			if (result == -1) { //1
				if (img.at<uchar>(i, j) == 1) {
					img.at<uchar>(i, j) = 0;
				}
				else {
					img.at<uchar>(i, j) = 255;
				}
			}
			else {
				img.at<uchar>(i, j) = 255;
			}
		}
	}
}

void Laplacian(Mat &img, Mat src, double threshold, int kernel[3][3], double plus_number) {
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

			double result = 0;
			for (int k = 0; k < 3; k++) {
				for (int l = 0; l < 3; l++) {
					result = result + temp[k][l] * kernel[k][l] * plus_number;
				}
			}

			if (result >= threshold) { //1
				img.at<uchar>(i, j) = 1;
			}
			else if(result <= -threshold) { //-1
				img.at<uchar>(i, j) = 2;
			}
			else { //0
				img.at<uchar>(i, j) = 0;
			}
		}
	}
	Mat img_2;
	img.copyTo(img_2);
	zco(img, img_2);
}



void Laplacian_kernel11(Mat &img, Mat src, double threshold, int kernel[11][11]) {
	for (int i = 0; i < img.rows; i++) {
		for (int j = 0; j < img.cols; j++) {
			int temp[11][11];
			int flag = 0;
			for (int k = 0; k < 11; k++) {
				for (int l = 0; l < 11; l++) {
					if (isvalid(img.rows, img.cols, i + k - 5, j + l - 5)) {
						temp[k][l] = src.at<uchar>(i + k - 5, j + l - 5);
					}
					else {
						temp[k][l] = -1;
						flag = 1;
					}
				}
			}

			if (flag == 1) {
				int cr = 0;
				int cc = 0;
				flag = 0;
				for (int k = 0; k < 11 && flag == 0; k++) {
					for (int l = 0; l < 11 && flag == 0; l++) {
						if (temp[k][l] != -1)  {
							if (isvalid(11, 11, k - 1, l) && isvalid(11, 11, k, l - 1) && 
								temp[k - 1][l] == -1 && temp[k][l - 1] == -1) {
								cr = k;
								cc = l;
								flag = 1;
							}
							else if (isvalid(11, 11, k - 1, l) && isvalid(11, 11, k, l + 1) &&
								temp[k - 1][l] == -1 && temp[k][l + 1] == -1) {
								cr = k;
								cc = l;
								flag = 2;
							}
							else if (isvalid(11, 11, k + 1, l) && isvalid(11, 11, k, l - 1) &&
								temp[k + 1][l] == -1 && temp[k][l - 1] == -1) {
								cr = k;
								cc = l;
								flag = 3;
							}
							else if (isvalid(11, 11, k, l + 1) && isvalid(11, 11, k + 1, l) &&
								temp[k][l + 1] == -1 && temp[k + 1][l] == -1) {
								cr = k;
								cc = l;
								flag = 4;
							}
						}
					}
				}

				if (flag == 1) {
					for (int k = 0; k < cr; k++) {
						for (int l = 0; l < cc; l++) {
							temp[k][l] = temp[cr][cc];
						}
					}

					for (int k = cr; k >= 0; k--) {
						for (int l = cc; l < 11; l++) {
							temp[k][l] = temp[cr][l];
						}
					}

					for (int k = cr; k < 11; k++) {
						for (int l = cc; l >= 0; l--) {
							temp[k][l] = temp[k][cc];
						}
					}
				}
				else if (flag == 2) {
					for (int k = 0; k < cr; k++) {
						for (int l = cc; l < 11; l++) {
							temp[k][l] = temp[cr][cc];
						}
					}

					for (int k = cr; k >= 0; k--) {
						for (int l = cc; l >= 0; l--) {
							temp[k][l] = temp[cr][l];
						}
					}

					for (int k = cr; k < 11; k++) {
						for (int l = cc; l < 11; l++) {
							temp[k][l] = temp[k][cc];
						}
					}
				}
				else if (flag == 3) {
					for (int k = cr; k < 11; k++) {
						for (int l = 0; l < cc; l++) {
							temp[k][l] = temp[cr][cc];
						}
					}

					for (int k = cr; k < 11; k++) {
						for (int l = cc; l < 11; l++) {
							temp[k][l] = temp[cr][l];
						}
					}

					for (int k = cr; k >= 0; k--) {
						for (int l = cc; l >= 0; l--) {
							temp[k][l] = temp[k][cc];
						}
					}
				}
				else if (flag == 4) {
					for (int k = cr ; k < 11; k++) {
						for (int l = cc; l < 11; l++) {
							temp[k][l] = temp[cr][cc];
						}
					}

					
					for (int k = cr; k < 11; k++) {
						for (int l = cc; l >= 0; l--) {
							temp[k][l] = temp[cr][l];
						}
					}

					for (int k = cr; k >= 0; k--) {
						for (int l = cc; l < 11; l++) {
							temp[k][l] = temp[k][cc];
						}
					}
				}
				else {
					for (int k = 0; k < 11; k++) {
						for (int l = 0; l < 11; l++) {
							if (temp[k][l] == -1) {
								if (k == 0 && l == 0) {
									int bound;
									int flag2 = 0;
									for (int m = 0; m < 11 && flag == 0; m++) {
										for (int n = 0; n < 11 && flag == 0; n++) {
											if (temp[k][l] != -1) {
												if (k == 0) {
													bound = l;
													flag2 = 1;
												}
												else if (l == 0){
													bound = k;
													flag2 = 2;
												}
												
											}
										}
									}

									if (flag2 == 1) {
										for (int m = 0; m < 11; m++) {
											for (int n = 0; n < bound; n++) {
												temp[m][n] = temp[m][bound];
											}
										}
									}
									else if (flag2 == 2) {
										for (int m = 0; m < bound; m++) {
											for (int n = 0; n < 11; n++) {
												temp[m][n] = temp[bound][n];
											}
										}
									}
									
								}
								else if (k == 0) {
									for (int m = 0; m < 11; m++) {
										for (int n = l; n < 11; n++) {
											temp[m][n] = temp[m][l - 1];
										}
									}
								}
								else if (l == 0) {
									for (int m = k; m < 11; m++) {
										for (int n = 0; n < 11; n++) {
											temp[m][n] = temp[k - 1][n];
										}
									}
								}
							}
						}
					}
				}
			}
			

			double result = 0;
			for (int k = 0; k < 11; k++) {
				for (int l = 0; l < 11; l++) {
					result = result + temp[k][l] * kernel[k][l];
				}
			}

			if (result >= threshold) { //1
				img.at<uchar>(i, j) = 1;
			}
			else if (result <= -threshold) { //-1
				img.at<uchar>(i, j) = 2;
			}
			else { //0
				img.at<uchar>(i, j) = 0;
			}
		}
	}
	Mat img_2;
	img.copyTo(img_2);
	zco(img, img_2);
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

	//Laplacian kernel 1 threshold 15
	Mat img_Laplacian1;
	img.copyTo(img_Laplacian1);
	Laplacian(img_Laplacian1, img, 15, kernel1, 1);
	imshow("img_Laplacian1", img_Laplacian1);
	imwrite("img_Laplacian1.jpg", img_Laplacian1);

	//Laplacian kernel 2 threshold 15
	Mat img_Laplacian2;
	img.copyTo(img_Laplacian2);
	Laplacian(img_Laplacian2, img, 15, kernel2, (double)1/3);
	imshow("img_Laplacian2", img_Laplacian2);
	imwrite("img_Laplacian2.jpg", img_Laplacian2);

	//Minimum-variance Laplacian kernel 3 threshold 30
	Mat img_Laplacian3;
	img.copyTo(img_Laplacian3);
	Laplacian(img_Laplacian3, img, 30, kernel3, (double)1/3);
	imshow("img_Laplacian3", img_Laplacian3);
	imwrite("img_Laplacian3.jpg", img_Laplacian3);

	//Laplacian of Gaussian kernel 4 threshold 3000
	Mat img_Laplacian4;
	img.copyTo(img_Laplacian4);
	Laplacian_kernel11(img_Laplacian4, img, 3000, kernel4);
	imshow("img_Laplacian4", img_Laplacian4);
	imwrite("img_Laplacian4.jpg", img_Laplacian4);

	//Difference of Gaussian threshold 1
	Mat img_Laplacian5;
	img.copyTo(img_Laplacian5);
	Laplacian_kernel11(img_Laplacian5, img, 1, kernel5);
	imshow("img_Laplacian5", img_Laplacian5);
	imwrite("img_Laplacian5.jpg", img_Laplacian5);

	waitKey(0);

	return 0;
}