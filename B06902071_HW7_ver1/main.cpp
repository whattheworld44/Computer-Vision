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

int cso_h(int b, int c, int d, int e) {
	if (b == c && (d != b || e != b)) {
		return 1;
	}
	return 0;
}

char cso_f(int a1, int a2, int a3, int a4) {
	int s = a1 + a2 + a3 + a4;
	if (s == 1) {
		return 'g';
	}
	else {
		return '0';
	}
}

int pro_h(char a, char m) {
	if (a == m) {
		return 1;
	}
	return 0;
}

char pro_f(int a1, int a2, int a3, int a4, int x0) {

	int total = 0;
	if (a1 == 1) {
		total += 1;
	}
	if (a2 == 1) {
		total += 1;
	}
	if (a3 == 1) {
		total += 1;
	}
	if (a4 == 1) {
		total += 1;
	}

	if (total < 1 || x0 != 1) {
		return 'q';
	}
	if (total >= 1 && x0 == 1) {
		return 'p';
	}
}

void printmatrix_x64(char m[64][64]) {
	for (int i = 0; i < 64; i++) {
		for (int j = 0; j < 64; j++) {
			printf("%c", m[i][j]);
		}
		printf("\n");
	}
	printf("\n");
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

	int sss = 0;
	while (sss < 7) {
		//yokoi connectivity
		char yokoi_matrix[64][64];

		for (int i = 0; i < img_binx64.rows; i++) {
			for (int j = 0; j < img_binx64.cols; j++) {
				if (img_binx64.at<uchar>(i, j) == 255) {
					int x[9] = { 0 };
					if (isvalid(img_binx64.rows, img_binx64.cols, i, j)) // x0
						x[0] = (int)img_binx64.at<uchar>(i, j);
					if (isvalid(img_binx64.rows, img_binx64.cols, i, j + 1)) // x1
						x[1] = (int)img_binx64.at<uchar>(i, j + 1);
					if (isvalid(img_binx64.rows, img_binx64.cols, i - 1, j)) // x2
						x[2] = (int)img_binx64.at<uchar>(i - 1, j);
					if (isvalid(img_binx64.rows, img_binx64.cols, i, j - 1)) // x3
						x[3] = (int)img_binx64.at<uchar>(i, j - 1);
					if (isvalid(img_binx64.rows, img_binx64.cols, i + 1, j)) // x4
						x[4] = (int)img_binx64.at<uchar>(i + 1, j);
					if (isvalid(img_binx64.rows, img_binx64.cols, i + 1, j + 1)) // x5
						x[5] = (int)img_binx64.at<uchar>(i + 1, j + 1);
					if (isvalid(img_binx64.rows, img_binx64.cols, i - 1, j + 1)) // x6
						x[6] = (int)img_binx64.at<uchar>(i - 1, j + 1);
					if (isvalid(img_binx64.rows, img_binx64.cols, i - 1, j - 1)) // x7
						x[7] = (int)img_binx64.at<uchar>(i - 1, j - 1);
					if (isvalid(img_binx64.rows, img_binx64.cols, i + 1, j - 1)) // x8
						x[8] = (int)img_binx64.at<uchar>(i + 1, j - 1);

					int result = yokoi_f(yokoi_h(x[0], x[1], x[6], x[2]),
						yokoi_h(x[0], x[2], x[7], x[3]),
						yokoi_h(x[0], x[3], x[8], x[4]),
						yokoi_h(x[0], x[4], x[5], x[1]));
					if (result != 0)
						yokoi_matrix[i][j] = result + '0';
					else
						yokoi_matrix[i][j] = ' ';
				}
				else {
					yokoi_matrix[i][j] = ' ';
				}

			}
		}

		//printmatrix_x64(yokoi_matrix);

		//pro
		char pro_matrix[64][64];
		for (int i = 0; i < 64; i++) {
			for (int j = 0; j < 64; j++) {
				if (yokoi_matrix[i][j] != ' ') {
					char x[5] = { '0' };
					if (isvalid(64, 64, i, j)) // x0
						x[0] = yokoi_matrix[i][j];
					if (isvalid(64, 64, i, j + 1)) // x1
						x[1] = yokoi_matrix[i][j + 1];
					if (isvalid(64, 64, i - 1, j)) // x2
						x[2] = yokoi_matrix[i - 1][j];
					if (isvalid(64, 64, i, j - 1)) // x3
						x[3] = yokoi_matrix[i][j - 1];
					if (isvalid(64, 64, i + 1, j)) // x4
						x[4] = yokoi_matrix[i + 1][j];

					char result = pro_f(pro_h(x[1], '1'),
						pro_h(x[2], '1'),
						pro_h(x[3], '1'),
						pro_h(x[4], '1'),
						pro_h(x[0], '1'));

					pro_matrix[i][j] = result;
				}
				else {
					pro_matrix[i][j] = ' ';
				}
			}
		}

		//printmatrix_x64(pro_matrix);

		//cso
		for (int i = 0; i < 64; i++) {
			for (int j = 0; j < 64; j++) {
				if (pro_matrix[i][j] == 'p') {
					int x[9] = { 0 };
					if (isvalid(img_binx64.rows, img_binx64.cols, i, j)) // x0
						x[0] = (int)img_binx64.at<uchar>(i, j);
					if (isvalid(img_binx64.rows, img_binx64.cols, i, j + 1)) // x1
						x[1] = (int)img_binx64.at<uchar>(i, j + 1);
					if (isvalid(img_binx64.rows, img_binx64.cols, i - 1, j)) // x2
						x[2] = (int)img_binx64.at<uchar>(i - 1, j);
					if (isvalid(img_binx64.rows, img_binx64.cols, i, j - 1)) // x3
						x[3] = (int)img_binx64.at<uchar>(i, j - 1);
					if (isvalid(img_binx64.rows, img_binx64.cols, i + 1, j)) // x4
						x[4] = (int)img_binx64.at<uchar>(i + 1, j);
					if (isvalid(img_binx64.rows, img_binx64.cols, i + 1, j + 1)) // x5
						x[5] = (int)img_binx64.at<uchar>(i + 1, j + 1);
					if (isvalid(img_binx64.rows, img_binx64.cols, i - 1, j + 1)) // x6
						x[6] = (int)img_binx64.at<uchar>(i - 1, j + 1);
					if (isvalid(img_binx64.rows, img_binx64.cols, i - 1, j - 1)) // x7
						x[7] = (int)img_binx64.at<uchar>(i - 1, j - 1);
					if (isvalid(img_binx64.rows, img_binx64.cols, i + 1, j - 1)) // x8
						x[8] = (int)img_binx64.at<uchar>(i + 1, j - 1);

					char result = cso_f(cso_h(x[0], x[1], x[6], x[2]),
						cso_h(x[0], x[2], x[7], x[3]),
						cso_h(x[0], x[3], x[8], x[4]),
						cso_h(x[0], x[4], x[5], x[1]));

					if (result == 'g')
						img_binx64.at<uchar>(i, j) = 0;
				}
			}
		}
		sss++;

		namedWindow("downSample as 64x64", 0);
		imshow("downSample as 64x64", img_binx64);
		waitKey(500);
	}
	
	namedWindow("downSample as 64x64", 0);
	imshow("downSample as 64x64", img_binx64);
	imwrite("downSample as 64x64.jpg", img_binx64);
	waitKey(0);


	return 0;
}