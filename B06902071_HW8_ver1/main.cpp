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

bool isvalid(int constRow, int constCol, int Row, int Col) {
	if (Row < 0 || Row >= constRow) {
		return false;
	}
	if (Col < 0 || Col >= constCol) {
		return false;
	}
	return true;
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

void Openning(Mat &img, Mat src) {
	Mat img_erosion;
	src.copyTo(img_erosion);
	Erosion(img_erosion, src);
	img_erosion.copyTo(img);
	Dilation(img, img_erosion);
}

void Closing(Mat &img, Mat src) {
	Mat img_dilation;
	src.copyTo(img_dilation);
	Dilation(img_dilation, src);
	img_dilation.copyTo(img);
	Erosion(img, img_dilation);
}

void  Openning_then_Closing(Mat &img, Mat src) {
	Mat img_temp;
	src.copyTo(img_temp);
	Openning(img_temp, src);
	img_temp.copyTo(img);
	Closing(img, img_temp);
}

void Closing_then_Opening(Mat &img, Mat src) {
	Mat img_temp;
	src.copyTo(img_temp);
	Closing(img_temp, src);
	img_temp.copyTo(img);
	Openning(img, img_temp);
}

void Boxfilter(Mat &img, Mat src, int size) {
	int mid = size / 2;
	for (int i = 0; i < img.rows; i++) {
		for (int j = 0; j < img.cols; j++) {
			int temp = 0;
			int count = 0;
			for (int k = 0; k < size; k++) {
				for (int l = 0; l < size; l++) {
					if (isvalid(img.rows, img.cols, i + k - mid, j + l - mid)) {
						temp = temp + src.at<uchar>(i + k - mid, j + l - mid);
						count++;
					}
				}
			}
			if (count != 0) {
				img.at<uchar>(i, j) = (uchar)(temp / count);
			}	
		}
	}
}

void Medianfilter(Mat &img, Mat src, int size) {
	int mid = size / 2;
	for (int i = 0; i < img.rows; i++) {
		for (int j = 0; j < img.cols; j++) {
			uchar temp[32];
			int count = 0;
			for (int k = 0; k < size; k++) {
				for (int l = 0; l < size; l++) {
					if (isvalid(img.rows, img.cols, i + k - mid, j + l - mid)) {
						temp[count] = src.at<uchar>(i + k - mid, j + l - mid);
						count++;
					}
				}
			}
			
			if (count != 0) {
				sort(temp, temp+count);
				img.at<uchar>(i, j) = temp[count / 2];
			}
		}
	}
}

double SNRcount(Mat img, Mat src) {
	double mean_img = 0.0;
	double mean_src = 0.0;

	for (int i = 0; i < img.rows; i++) {
		for (int j = 0; j < img.cols; j++) {
			mean_img = mean_img + (img.at<uchar>(i, j) - src.at<uchar>(i, j));
			mean_src = mean_src + src.at<uchar>(i, j);
		}
	}
	mean_img = mean_img / (img.rows*img.cols);
	mean_src = mean_src / (img.rows*img.cols);

	double stddev_img = 0.0;
	double stddev_src = 0.0;
	for (int i = 0; i < img.rows; i++) {
		for (int j = 0; j < img.cols; j++) {
			stddev_img = stddev_img + (img.at<uchar>(i, j) - src.at<uchar>(i, j) - mean_img)*(img.at<uchar>(i, j) - src.at<uchar>(i, j) - mean_img);
			stddev_src = stddev_src + (src.at<uchar>(i, j) - mean_src)*(src.at<uchar>(i, j) - mean_src);
		}
	}
	stddev_img = stddev_img / (img.rows*img.cols);
	stddev_src = stddev_src / (img.rows*img.cols);

	double snr = 20 * log10(sqrt(stddev_src) / sqrt(stddev_img));
	return snr;
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
	
	random_device rd;
	mt19937 gen(rd());
	normal_distribution<> distr(0, 1); // mean stddev
	uniform_real_distribution<> distr2(0, 1);

	//Gnoise10
	Mat img_Gnoise10;
	img.copyTo(img_Gnoise10);

	for (int i = 0; i < img.rows; i++) {
		for (int j = 0; j < img.cols; j++) {
			int temp = (int)(img_Gnoise10.at<uchar>(i, j) + 10 * distr(gen));
			img_Gnoise10.at<uchar>(i, j) = max(0, min(255, temp));
		}
	}
	imshow("img_Gnoise10", img_Gnoise10);
	imwrite("img_Gnoise10.jpg", img_Gnoise10);
	printf("img_Gnoise10 SNR = %lf\n", SNRcount(img_Gnoise10, img));

	//box3x3
	Mat img_Gnoise10_box3x3;
	img_Gnoise10.copyTo(img_Gnoise10_box3x3);
	Boxfilter(img_Gnoise10_box3x3, img_Gnoise10, 3);
	imshow("img_Gnoise10_box3x3", img_Gnoise10_box3x3);
	imwrite("img_Gnoise10_box3x3.jpg", img_Gnoise10_box3x3);
	printf("img_Gnoise10_box3x3 SNR = %lf\n", SNRcount(img_Gnoise10_box3x3, img));

	//box5x5
	Mat img_Gnoise10_box5x5;
	img_Gnoise10.copyTo(img_Gnoise10_box5x5);
	Boxfilter(img_Gnoise10_box5x5, img_Gnoise10, 5);
	imshow("img_Gnoise10_box5x5", img_Gnoise10_box5x5);
	imwrite("img_Gnoise10_box5x5.jpg", img_Gnoise10_box5x5);
	printf("img_Gnoise10_box5x5 SNR = %lf\n", SNRcount(img_Gnoise10_box5x5, img));

	//median3x3
	Mat img_Gnoise10_median3x3;
	img_Gnoise10.copyTo(img_Gnoise10_median3x3);
	Medianfilter(img_Gnoise10_median3x3, img_Gnoise10, 3);
	imshow("img_Gnoise10_median3x3", img_Gnoise10_median3x3);
	imwrite("img_Gnoise10_median3x3.jpg", img_Gnoise10_median3x3);
	printf("img_Gnoise10_median3x3 SNR = %lf\n", SNRcount(img_Gnoise10_median3x3, img));

	//median5x5
	Mat img_Gnoise10_median5x5;
	img_Gnoise10.copyTo(img_Gnoise10_median5x5);
	Medianfilter(img_Gnoise10_median5x5, img_Gnoise10, 5);
	imshow("img_Gnoise10_median5x5", img_Gnoise10_median5x5);
	imwrite("img_Gnoise10_median5x5.jpg", img_Gnoise10_median5x5);
	printf("img_Gnoise10_median5x5 SNR = %lf\n", SNRcount(img_Gnoise10_median5x5, img));

	//opening-then-closing
	Mat img_Gnoise10_otclosing;
	img_Gnoise10.copyTo(img_Gnoise10_otclosing);
	Openning_then_Closing(img_Gnoise10_otclosing, img_Gnoise10);
	imshow("img_Gnoise10_otclosing", img_Gnoise10_otclosing);
	imwrite("img_Gnoise10_otclosing.jpg", img_Gnoise10_otclosing);
	printf("img_Gnoise10_otclosing SNR = %lf\n", SNRcount(img_Gnoise10_otclosing, img));


	//closeing-then-opening
	Mat img_Gnoise10_ctopenning;
	img_Gnoise10.copyTo(img_Gnoise10_ctopenning);
	Closing_then_Opening(img_Gnoise10_ctopenning, img_Gnoise10);
	imshow("img_Gnoise10_ctopenning", img_Gnoise10_ctopenning);
	imwrite("img_Gnoise10_ctopenning.jpg", img_Gnoise10_ctopenning);
	printf("img_Gnoise10_ctopenning SNR = %lf\n", SNRcount(img_Gnoise10_ctopenning, img));
	///////////////////////////////////////////////////////////////////////////

	//Gnoise30
	Mat img_Gnoise30;
	img.copyTo(img_Gnoise30);

	for (int i = 0; i < img.rows; i++) {
		for (int j = 0; j < img.cols; j++) {
			int temp = (int)(img_Gnoise30.at<uchar>(i, j) + 30 * distr(gen));
			img_Gnoise30.at<uchar>(i, j) = max(0, min(255, temp));
		}
	}

	imshow("img_Gnoise30", img_Gnoise30);
	imwrite("img_Gnoise30.jpg", img_Gnoise30);
	printf("img_Gnoise30 SNR = %lf\n", SNRcount(img_Gnoise30, img));

	//box3x3
	Mat img_Gnoise30_box3x3;
	img_Gnoise30.copyTo(img_Gnoise30_box3x3);
	Boxfilter(img_Gnoise30_box3x3, img_Gnoise30, 3);
	imshow("img_Gnoise30_box3x3", img_Gnoise30_box3x3);
	imwrite("img_Gnoise30_box3x3.jpg", img_Gnoise30_box3x3);
	printf("img_Gnoise30_box3x3 SNR = %lf\n", SNRcount(img_Gnoise30_box3x3, img));

	//box5x5
	Mat img_Gnoise30_box5x5;
	img_Gnoise30.copyTo(img_Gnoise30_box5x5);
	Boxfilter(img_Gnoise30_box5x5, img_Gnoise30, 5);
	imshow("img_Gnoise30_box5x5", img_Gnoise30_box5x5);
	imwrite("img_Gnoise30_box5x5.jpg", img_Gnoise30_box5x5);
	printf("img_Gnoise30_box5x5 SNR = %lf\n", SNRcount(img_Gnoise30_box5x5, img));

	//median3x3
	Mat img_Gnoise30_median3x3;
	img_Gnoise30.copyTo(img_Gnoise30_median3x3);
	Medianfilter(img_Gnoise30_median3x3, img_Gnoise30, 3);
	imshow("img_Gnoise30_median3x3", img_Gnoise30_median3x3);
	imwrite("img_Gnoise30_median3x3.jpg", img_Gnoise30_median3x3);
	printf("img_Gnoise30_median3x3 SNR = %lf\n", SNRcount(img_Gnoise30_median3x3, img));

	//median5x5
	Mat img_Gnoise30_median5x5;
	img_Gnoise30.copyTo(img_Gnoise30_median5x5);
	Medianfilter(img_Gnoise30_median5x5, img_Gnoise30, 5);
	imshow("img_Gnoise30_median5x5", img_Gnoise30_median5x5);
	imwrite("img_Gnoise30_median5x5.jpg", img_Gnoise30_median5x5);
	printf("img_Gnoise30_median5x5 SNR = %lf\n", SNRcount(img_Gnoise30_median5x5, img));

	//opening-then-closing
	Mat img_Gnoise30_otclosing;
	img_Gnoise30.copyTo(img_Gnoise30_otclosing);
	Openning_then_Closing(img_Gnoise30_otclosing, img_Gnoise30);
	imshow("img_Gnoise30_otclosing", img_Gnoise30_otclosing);
	imwrite("img_Gnoise30_otclosing.jpg", img_Gnoise30_otclosing);
	printf("img_Gnoise30_otclosing SNR = %lf\n", SNRcount(img_Gnoise30_otclosing, img));


	//closeing-then-opening
	Mat img_Gnoise30_ctopenning;
	img_Gnoise30.copyTo(img_Gnoise30_ctopenning);
	Closing_then_Opening(img_Gnoise30_ctopenning, img_Gnoise30);
	imshow("img_Gnoise30_ctopenning", img_Gnoise30_ctopenning);
	imwrite("img_Gnoise30_ctopenning.jpg", img_Gnoise30_ctopenning);
	printf("img_Gnoise30_ctopenning SNR = %lf\n", SNRcount(img_Gnoise30_ctopenning, img));
	///////////////////////////////////////////////////////////////////////////

	//img_Normal_01;
	Mat img_Normal_01;
	img.copyTo(img_Normal_01);

	for (int i = 0; i < img.rows; i++) {
		for (int j = 0; j < img.cols; j++) {
			double temp = distr2(gen);
			
			if (temp < 0.1) {
				img_Normal_01.at<uchar>(i, j) = 0;
			}
			else if (temp > (1-0.1)) {
				img_Normal_01.at<uchar>(i, j) = 255;
			}
			else {
				img_Normal_01.at<uchar>(i, j) = img_Normal_01.at<uchar>(i, j);
			}
		}
	}
	imshow("img_Normal_01", img_Normal_01);
	imwrite("img_Normal_01.jpg", img_Normal_01);
	printf("img_Normal_01 SNR = %lf\n", SNRcount(img_Normal_01, img));

	//box3x3
	Mat img_Normal_01_box3x3;
	img_Normal_01.copyTo(img_Normal_01_box3x3);
	Boxfilter(img_Normal_01_box3x3, img_Normal_01, 3);
	imshow("img_Normal_01_box3x3", img_Normal_01_box3x3);
	imwrite("img_Normal_01_box3x3.jpg", img_Normal_01_box3x3);
	printf("img_Normal_01_box3x3 SNR = %lf\n", SNRcount(img_Normal_01_box3x3, img));

	//box5x5
	Mat img_Normal_01_box5x5;
	img_Normal_01.copyTo(img_Normal_01_box5x5);
	Boxfilter(img_Normal_01_box5x5, img_Normal_01, 5);
	imshow("img_Normal_01_box5x5", img_Normal_01_box5x5);
	imwrite("img_Normal_01_box5x5.jpg", img_Normal_01_box5x5);
	printf("img_Normal_01_box5x5 SNR = %lf\n", SNRcount(img_Normal_01_box5x5, img));

	//median3x3
	Mat img_Normal_01_median3x3;
	img_Normal_01.copyTo(img_Normal_01_median3x3);
	Medianfilter(img_Normal_01_median3x3, img_Normal_01, 3);
	imshow("img_Normal_01_median3x3", img_Normal_01_median3x3);
	imwrite("img_Normal_01_median3x3.jpg", img_Normal_01_median3x3);
	printf("img_Normal_01_median3x3 SNR = %lf\n", SNRcount(img_Normal_01_median3x3, img));

	//median5x5
	Mat img_Normal_01_median5x5;
	img_Normal_01.copyTo(img_Normal_01_median5x5);
	Medianfilter(img_Normal_01_median5x5, img_Normal_01, 5);
	imshow("img_Normal_01_median5x5", img_Normal_01_median5x5);
	imwrite("img_Normal_01_median5x5.jpg", img_Normal_01_median5x5);
	printf("img_Normal_01_median5x5 SNR = %lf\n", SNRcount(img_Normal_01_median5x5, img));

	//opening-then-closing
	Mat img_Normal_01_otclosing;
	img_Normal_01.copyTo(img_Normal_01_otclosing);
	Openning_then_Closing(img_Normal_01_otclosing, img_Normal_01);
	imshow("img_Normal_01_otclosing", img_Normal_01_otclosing);
	imwrite("img_Normal_01_otclosing.jpg", img_Normal_01_otclosing);
	printf("img_Normal_01_otclosing SNR = %lf\n", SNRcount(img_Normal_01_otclosing, img));


	//closeing-then-opening
	Mat img_Normal_01_ctopenning;
	img_Normal_01.copyTo(img_Normal_01_ctopenning);
	Closing_then_Opening(img_Normal_01_ctopenning, img_Normal_01);
	imshow("img_Normal_01_ctopenning", img_Normal_01_ctopenning);
	imwrite("img_Normal_01_ctopenning.jpg", img_Normal_01_ctopenning);
	printf("img_Normal_01_ctopenning SNR = %lf\n", SNRcount(img_Normal_01_ctopenning, img));
	///////////////////////////////////////////////////////////////////////////

	//img_Normal_005;
	Mat img_Normal_005;
	img.copyTo(img_Normal_005);

	for (int i = 0; i < img.rows; i++) {
		for (int j = 0; j < img.cols; j++) {
			double temp = distr2(gen);

			if (temp < 0.05) {
				img_Normal_005.at<uchar>(i, j) = 0;
			}
			else if (temp > (1 - 0.05)) {
				img_Normal_005.at<uchar>(i, j) = 255;
			}
			else {
				img_Normal_005.at<uchar>(i, j) = img_Normal_005.at<uchar>(i, j);
			}
		}
	}
	imshow("img_Normal_005", img_Normal_005);
	imwrite("img_Normal_005.jpg", img_Normal_005);
	printf("img_Normal_005 SNR = %lf\n", SNRcount(img_Normal_005, img));

	//box3x3
	Mat img_Normal_005_box3x3;
	img_Normal_005.copyTo(img_Normal_005_box3x3);
	Boxfilter(img_Normal_005_box3x3, img_Normal_005, 3);
	imshow("img_Normal_005_box3x3", img_Normal_005_box3x3);
	imwrite("img_Normal_005_box3x3.jpg", img_Normal_005_box3x3);
	printf("img_Normal_005_box3x3 SNR = %lf\n", SNRcount(img_Normal_005_box3x3, img));

	//box5x5
	Mat img_Normal_005_box5x5;
	img_Normal_005.copyTo(img_Normal_005_box5x5);
	Boxfilter(img_Normal_005_box5x5, img_Normal_005, 5);
	imshow("img_Normal_005_box5x5", img_Normal_005_box5x5);
	imwrite("img_Normal_005_box5x5.jpg", img_Normal_005_box5x5);
	printf("img_Normal_005_box5x5 SNR = %lf\n", SNRcount(img_Normal_005_box5x5, img));

	//median3x3
	Mat img_Normal_005_median3x3;
	img_Normal_005.copyTo(img_Normal_005_median3x3);
	Medianfilter(img_Normal_005_median3x3, img_Normal_005, 3);
	imshow("img_Normal_005_median3x3", img_Normal_005_median3x3);
	imwrite("img_Normal_005_median3x3.jpg", img_Normal_005_median3x3);
	printf("img_Normal_005_median3x3 SNR = %lf\n", SNRcount(img_Normal_005_median3x3, img));

	//median5x5
	Mat img_Normal_005_median5x5;
	img_Normal_005.copyTo(img_Normal_005_median5x5);
	Medianfilter(img_Normal_005_median5x5, img_Normal_005, 5);
	imshow("img_Normal_005_median5x5", img_Normal_005_median5x5);
	imwrite("img_Normal_005_median5x5.jpg", img_Normal_005_median5x5);
	printf("img_Normal_005_median5x5 SNR = %lf\n", SNRcount(img_Normal_005_median5x5, img));

	//opening-then-closing
	Mat img_Normal_005_otclosing;
	img_Normal_005.copyTo(img_Normal_005_otclosing);
	Openning_then_Closing(img_Normal_005_otclosing, img_Normal_005);
	imshow("img_Normal_005_otclosing", img_Normal_005_otclosing);
	imwrite("img_Normal_005_otclosing.jpg", img_Normal_005_otclosing);
	printf("img_Normal_005_otclosing SNR = %lf\n", SNRcount(img_Normal_005_otclosing, img));


	//closeing-then-opening
	Mat img_Normal_005_ctopenning;
	img_Normal_005.copyTo(img_Normal_005_ctopenning);
	Closing_then_Opening(img_Normal_005_ctopenning, img_Normal_005);
	imshow("img_Normal_005_ctopenning", img_Normal_005_ctopenning);
	imwrite("img_Normal_005_ctopenning.jpg", img_Normal_005_ctopenning);
	printf("img_Normal_005_ctopenning SNR = %lf\n", SNRcount(img_Normal_005_ctopenning, img));
	///////////////////////////////////////////////////////////////////////////

	waitKey(0);

	return 0;
}