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

	//binarize
	Mat img_bin;
	img.copyTo(img_bin);

	for (int i = 0; i < img.rows; i++)
		for (int j = 0; j < img.cols; j++) {
			if (img.at<uchar>(i, j) >= 128) {
				img_bin.at<uchar>(i, j) = 255;
			}
			else {
				img_bin.at<uchar>(i, j) = 0;
			}

		}
	imshow("binarize at 128", img_bin);
	imwrite("binarize at 128.jpg", img_bin);

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
	imshow("histogram", img_histo);
	imwrite("histogram.jpg", img_histo);


	//connected components
	
	Mat img_bin_clone(img_bin.rows, img_bin.cols, CV_32SC1, Scalar(0));

	int g_lebel = 1;
	vector<bool> table[1024];
	for (int i = 0; i < 1024; i++) {
		table[i].resize(1024);
	}
	for (int i = 0; i < 1024; i++)
		for (int j = 0; j < 1024; j++)
			table[i][j] = false;

	for (int i = 0; i < img_bin.rows; i++)
		for (int j = 0; j < img_bin.cols; j++) {
			if (img_bin.at<uchar>(i, j) != 0) { //not Background
				//left
				int lebel[4] = { 0 };
				if (isvalid(img_bin.rows, img_bin.cols, i , j - 1)) {
					lebel[0] = img_bin_clone.at<int>(i, j - 1);
				}
				//left + up
				if (isvalid(img_bin.rows, img_bin.cols, i - 1, j - 1)) {
					lebel[1] = img_bin_clone.at<int>(i - 1, j - 1);
				}
				// up
				if (isvalid(img_bin.rows, img_bin.cols, i - 1, j)) {
					lebel[2] = img_bin_clone.at<int>(i - 1, j);
				}
				// right + up
				if (isvalid(img_bin.rows, img_bin.cols, i - 1, j + 1)) {
					lebel[3] = img_bin_clone.at<int>(i - 1, j + 1);
				}

				if (lebel[0] == 0 && lebel[1] == 0 && lebel[2] == 0 && lebel[3] == 0) {
					img_bin_clone.at<int>(i, j) = g_lebel;
					table[g_lebel][g_lebel] = true;
					g_lebel++; 
				}
				else {
					if ((lebel[0] == lebel[1]) && (lebel[1] == lebel[2]) && (lebel[2] == lebel[3]) ) {
						img_bin_clone.at<int>(i, j) = lebel[0];
					}
					else {
						sort(lebel, lebel+4);
						for (int temp = 0; temp < 4; temp++) {
							if (lebel[temp] != 0) {
								img_bin_clone.at<int>(i, j) = lebel[temp];
								break;
							}
						}
						for (int temp = 0; temp < 4; temp++) {
							if (lebel[temp] != 0 && lebel[temp] != img_bin_clone.at<int>(i, j) ) {
								table[img_bin_clone.at<int>(i, j)][lebel[temp]] = true;
								table[lebel[temp]][img_bin_clone.at<int>(i, j)] = true;
								break;
							}
						}
						
					}
				}
			}
		}
	
	int count_pixel[1024] = { 0 };
	
	for (int i = 0; i < img_bin.rows; i++) {
		for (int j = 0; j < img_bin.cols; j++) {
			for (int k = 0; k < 1024; k++) {
				if (table[img_bin_clone.at<int>(i, j)][k] == true) {
					if (k == img_bin_clone.at<int>(i, j)) {
						img_bin_clone.at<int>(i, j) = k;
						count_pixel[k]++;
						break;
					}
					else {
						img_bin_clone.at<int>(i, j) = k;
						k = 0;
					}
					
				}
				else if (k > img_bin_clone.at<int>(i, j)) {
					break;
				}
			}
		}
	}
	int index_count = 0;
	int index_to_rec[1024];
	int rec_to_index[1024];
	int fx[1024];
	int	fy[1024];
	int bx[1024];
	int by[1024];

	for (int i = 0; i < 1024; i++) {
		rec_to_index[i] = -1;
	}
	for (int i = 0; i < img_bin.rows; i++) {
		for (int j = 0; j < img_bin.cols; j++) {
			if (img_bin_clone.at<int>(i, j) != 0) {
				if (count_pixel[img_bin_clone.at<int>(i, j)] > 500) {
					if (rec_to_index[img_bin_clone.at<int>(i, j)] == -1) {
						index_to_rec[index_count] = img_bin_clone.at<int>(i, j);
						rec_to_index[img_bin_clone.at<int>(i, j)] = index_count;
						fx[rec_to_index[img_bin_clone.at<int>(i, j)]] = i;
						fy[rec_to_index[img_bin_clone.at<int>(i, j)]] = j;
						bx[rec_to_index[img_bin_clone.at<int>(i, j)]] = i;
						by[rec_to_index[img_bin_clone.at<int>(i, j)]] = j;
						index_count++;
					}
					else {
						if (by[rec_to_index[img_bin_clone.at<int>(i, j)]] < j) {
							by[rec_to_index[img_bin_clone.at<int>(i, j)]] = j;
						}
						bx[rec_to_index[img_bin_clone.at<int>(i, j)]] = i;
						
					}
				}
			}
		}
	}

	for (int i = 0; i < index_count; i++) {
		rectangle(img_bin_clone, 
			Point(fy[rec_to_index[index_to_rec[i]]], fx[rec_to_index[index_to_rec[i]]]),
			Point(by[rec_to_index[index_to_rec[i]]], bx[rec_to_index[index_to_rec[i]]]),
			Scalar(255, 255, 0), 1, 20, 0);

	}
	img_bin_clone.convertTo(img_bin_clone, CV_8U, 255);
	imshow("connected components", img_bin_clone);
	imwrite("connected components.jpg", img_bin_clone);

	waitKey(0);
	return 0;
}