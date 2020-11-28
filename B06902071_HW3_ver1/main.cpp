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

int main()
{
	// read image 
	Mat img = imread("lena.bmp", CV_8UC1);

	

	// sample code : Lomo Effect
	int center_i = img.rows / 2;
	int center_j = img.cols / 2;
	double max_dist = sqrt(center_i*center_i + center_j*center_j);

	//imshow("lena.bmp", img);
	//imwrite("lena.jpg", img);

	//original
	imshow("img", img);
	imwrite("img.jpg", img);

	//histogram
	int pixel_array[256] = { 0 };
	int max_parray_v = 1;
	for (int i = 0; i < img.rows; i++)
		for (int j = 0; j < img.cols; j++) {
			pixel_array[img.at<uchar>(i, j)] += 1;
			if (pixel_array[img.at<uchar>(i, j)] > max_parray_v) {
				max_parray_v = pixel_array[img.at<uchar>(i, j)];
			}
		}

	double histo_rate = max_parray_v / 512;
	Mat img_histo(512, 512, CV_8UC1, Scalar(0));
	

	for (int i = 0; i < 512; i++)
		for (int j = 0; j < 512; j++) {
			if (j < pixel_array[i]/ histo_rate) {
				img_histo.at<uchar>(511-j, i) = 0;
			}
			else {
				img_histo.at<uchar>(511-j, i) = 255;
			}
		}
	imshow("img histogram", img_histo);
	imwrite("img histogram.jpg", img_histo);

	//divide by 3 img
	Mat img_bin_d3;
	img.copyTo(img_bin_d3);
	
	for (int i = 0; i < img.rows; i++)
		for (int j = 0; j < img.cols; j++) {
			img_bin_d3.at<uchar>(i, j) = (uchar)(img_bin_d3.at<uchar>(i, j) / 3);
		}
	imshow("img divid by 3", img_bin_d3);
	imwrite("img divid by 3.jpg", img_bin_d3);

	//divide by 3 img histogram
	int pixel_array_d3[256] = { 0 };
	int max_parray_v_d3 = 1;
	for (int i = 0; i < img.rows; i++)
		for (int j = 0; j < img.cols; j++) {
			pixel_array_d3[img_bin_d3.at<uchar>(i, j)] += 1;
			if (pixel_array_d3[img_bin_d3.at<uchar>(i, j)] > max_parray_v_d3) {
				max_parray_v_d3 = pixel_array_d3[img_bin_d3.at<uchar>(i, j)];
			}
		}

	double histo_rate_d3 = max_parray_v_d3 / 512;
	Mat img_histo_d3(512, 512, CV_8UC1, Scalar(0));


	for (int i = 0; i < 512; i++)
		for (int j = 0; j < 512; j++) {
			if (j < pixel_array_d3[i] / histo_rate_d3) {
				img_histo_d3.at<uchar>(511 - j, i) = 0;
			}
			else {
				img_histo_d3.at<uchar>(511 - j, i) = 255;
			}
		}
	imshow("img divid by 3 histogram", img_histo_d3);
	imwrite("img divid by 3 histogram.jpg", img_histo_d3);

	//img equalization
	Mat img_equ_d3;
	img_bin_d3.copyTo(img_equ_d3);

	int CDF[256] = { 0 };
	int min_parray_v_d3 = 2147483647;
	for (int i = 0; i < 256; i++) {
		for (int j = 0; j < i; j++) {
			CDF[i] = CDF[i] + pixel_array_d3[j];
		}
		if (CDF[i] != 0) {
			if (min_parray_v_d3 > CDF[i]) {
				min_parray_v_d3 = CDF[i];
			}
		}
	}

	for (int i = 0; i < 512; i++)
		for (int j = 0; j < 512; j++) {
			img_equ_d3.at<uchar>(i, j) = (uchar)(((double)(CDF[img_equ_d3.at<uchar>(i, j)] - min_parray_v_d3) /
				(double)(CDF[255] - min_parray_v_d3)) * 255);
		}
	imshow("img divid by 3 equalization", img_equ_d3);
	imwrite("img divid by 3 equalization.jpg", img_equ_d3);

	//histogram equalization 
	int pixel_array_d3_equalization[256] = { 0 };
	int max_parray_v_d3_equalization = 1;
	for (int i = 0; i < img.rows; i++)
		for (int j = 0; j < img.cols; j++) {
			pixel_array_d3_equalization[img_equ_d3.at<uchar>(i, j)] += 1;
			if (pixel_array_d3_equalization[img_equ_d3.at<uchar>(i, j)] > max_parray_v_d3_equalization) {
				max_parray_v_d3_equalization = pixel_array_d3_equalization[img_equ_d3.at<uchar>(i, j)];
			}
		}

	double histo_rate_d3_equalization = max_parray_v_d3_equalization / 512;
	Mat img_histo_d3_equalization(512, 512, CV_8UC1, Scalar(0));


	for (int i = 0; i < 512; i++)
		for (int j = 0; j < 512; j++) {
			if (j < pixel_array_d3_equalization[i] / histo_rate_d3_equalization) {
				img_histo_d3_equalization.at<uchar>(511 - j, i) = 0;
			}
			else {
				img_histo_d3_equalization.at<uchar>(511 - j, i) = 255;
			}
		}
	imshow("img divid by 3 equalization histogram", img_histo_d3_equalization);
	imwrite("img divid by 3 equalization histogram.jpg", img_histo_d3_equalization);

	waitKey(0);
	return 0;
}