#include <opencv2/core.hpp>
#include <opencv2/ml.hpp>
#include "opencv2/highgui/highgui.hpp"
//#include "opencv2/imgproc/imgproc.hpp"
#include <iostream>
#include <string>
#include <fstream>

#define NUM_CLUS 1

int check_file(cv::Mat& file, cv::String& filename)
{
	if (file.empty())
	{
		std::cout << "Couldn't find/open image: " << filename << std::endl;
		system("pause");
		return -1;
	}
	return 0;
}

int getMaskedPoints(cv::String& fn, cv::Mat& img, cv::Mat& mask_points)
{
	cv::Mat mask = cv::imread(fn.c_str(), cv::IMREAD_GRAYSCALE);
	if (check_file(mask, fn) < 0)
		return -1;

	// Turn to binary mask
	mask = mask < 255;

	// Get only the seeded pixels
	cv::bitwise_and(img, mask, mask);

	// Get foreground points
	mask_points = cv::Mat::zeros(cv::countNonZero(mask), 1, CV_32SC1);
	int ctr = 0;
	for (int i = 0; i < mask.rows; i++) {
		for (int j = 0; j < mask.cols; j++) {
			if (mask.at<uchar>(i, j) > 0) {
				mask_points.at<int>(ctr) = mask.at<uchar>(i, j);
				ctr++;
			}
		}
	}
	return 0;
}

int cvOutputData(cv::Mat means, cv::Mat weights, std::vector<cv::Mat> cov)
{
	// Mean
	std::cout << "MEANS" << std::endl;
	for (int i = 0; i < NUM_CLUS; i++)
		std::cout << "\t" << means.at<double>(i) << std::endl;

	// Sigma
	std::cout << "\n\nSIGMAS " << std::endl;
	for (int i = 0; i < NUM_CLUS; i++)
		std::cout << "\t" << cov[i].at<double>(0) << std::endl;

	// Weights
	std::cout << "\n\nWEIGHTS " << std::endl;
	for (int i = 0; i < NUM_CLUS; i++)
		std::cout << "\t" << weights.at<double>(i) << std::endl;


	return 0;
}


int main(int argc, char** argv)
{
	// Image name
	cv::String imageName = "../samples/noise_blur_00.png";

	// Open the images
	cv::Mat img = cv::imread(imageName.c_str(), cv::IMREAD_GRAYSCALE);
	if (check_file(img, imageName) < 0)
		return -1;


	cv::String fgName = "../samples/noise_blur_00_fg.bmp";
	cv::Mat fg_points;
	getMaskedPoints(fgName, img, fg_points);

	cv::String bgName = "../samples/noise_blur_00_bg.bmp";
	cv::Mat bg_points;
	getMaskedPoints(bgName, img, bg_points);
	
	
	//cv::namedWindow("Display window", cv::WINDOW_AUTOSIZE); // Create a window for display.
	//cv::imshow("Display window", fg);
	
	//cv::Ptr<cv::ml::EM> mdl = cv::ml::EM::create();
	//mdl->setClustersNumber(NUM_CLUS);
	//mdl->trainEM(fg_points);

	//// Get means
	//cv::Mat means = mdl->getMeans();
	//// Get weights
	//cv::Mat weights = mdl->getWeights();

	//// Get covariance matrix
	//std::vector<cv::Mat> cov;
	//for (int i = 0; i < NUM_CLUS; i++)
	//	cov.push_back(cv::Mat::zeros(1, 1, CV_64FC1));
	//mdl->getCovs(cov);

	//// Print out the data (to stdout and to file)
	//cvOutputData(means, weights, cov);

	//// Display image
	//cv::namedWindow("Display window", cv::WINDOW_AUTOSIZE); // Create a window for display.
	//cv::imshow("Display window", bg_bin);


	cv::waitKey(0);
	return 0;
}