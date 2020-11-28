#include<cstdio>
#include<cstdlib>
#include<cmath>

#include<opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>

using namespace cv;

int main()
{
	// read image 
	Mat img = imread("lena.bmp", CV_8UC1);

	

	// sample code : Lomo Effect
	int center_i = img.rows / 2;
	int center_j = img.cols / 2;
	double max_dist = sqrt(center_i*center_i + center_j*center_j);

	imshow("lena.bmp", img);
	imwrite("lena.jpg", img);


	//upside-down
	Mat img_UD;
	img.copyTo(img_UD);
	for (int i = 0; i < center_i; i++)
		for (int j = 0; j < img.cols; j++) {
			swap(img_UD.at<uchar>(i, j), img_UD.at<uchar>(img_UD.rows - i -1, j));
		}
	imshow("upside down", img_UD);
	imwrite("upside down.jpg", img_UD);


	//right - side - left
	Mat img_RsL;
	img.copyTo(img_RsL);
	for(int i = 0; i < img.rows; i++)
		for (int j = 0; j < center_j; j++) {
			swap(img_RsL.at<uchar>(i, j), img_RsL.at<uchar>(i, img_RsL.cols - j - 1));
		}
	
	imshow("right side left", img_RsL);
	imwrite("right side left.jpg", img_RsL);

	//diagonally flip
	Mat img_df(img.rows, img.cols, CV_8UC1);
	for (int i = 0; i < img.cols; i++)
		for (int j = 0; j < img.rows; j++) {
			img_df.at<uchar>(img.rows - j - 1, img.cols - i - 1) = img.at<uchar>(i, j);
			
		}
	imshow("diagonally flip", img_df);
	imwrite("diagonally flip.jpg", img_df);


	//rotate
	Mat img_rot;
	Point2f pt(img.cols / 2., img.rows / 2.);
	Mat r = getRotationMatrix2D(pt, -45, 1.0);
	warpAffine(img, img_rot, r, Size(img.cols, img.rows));

	imshow("rotate 45", img_rot);
	imwrite("rotate 45.jpg", img_rot);

	//shrink
	Size dsize = Size(img.cols/2, img.rows/2);
	Mat img_shrink = Mat(dsize, CV_8UC1);
	resize(img, img_shrink, dsize);
	imshow("shrink in half", img_shrink);
	imwrite("shrink in half.jpg", img_shrink);

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

	waitKey(0);
	return 0;
}