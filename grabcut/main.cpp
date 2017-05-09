#include <iostream>
#include <string>
#include <fstream>

#include <opencv2/core.hpp>
#include <opencv2/ml.hpp>
#include "opencv2/highgui/highgui.hpp"
//#include "opencv2/imgproc/imgproc.hpp"

#include "graph.h"
#define NUM_CLUS 1

int checkFile(cv::Mat& file, cv::String& filename)
{
	if (file.empty())
	{
		std::cout << "Couldn't find/open image: " << filename << std::endl;
		system("pause");
		return -1;
	}
	return 0;
}

int getMaskedPoints(cv::String& fn, cv::Mat& img, cv::Mat& mask, cv::Mat& mask_points)
{
	mask = cv::imread(fn.c_str(), cv::IMREAD_GRAYSCALE);
	if (checkFile(mask, fn) < 0)
		return -1;

	// Turn to binary mask
	mask = mask < 255;

	// Get only the seeded pixels
	cv::bitwise_and(img, mask, mask);

	// Get points
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

void estGMM(cv::Mat& points, cv::Mat& likelihoods)
{
	cv::Ptr<cv::ml::EM> mdl = cv::ml::EM::create();
	mdl->setClustersNumber(NUM_CLUS);
	mdl->trainEM(points, likelihoods);
}

// For checking
int outputData(cv::Mat means, cv::Mat weights, std::vector<cv::Mat> cov)
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
	typedef Graph<int, int, int> GraphType;
	// Image name
	cv::String imageName = "../samples/noise_blur_00.png";

	// Open the images
	cv::Mat img = cv::imread(imageName.c_str(), cv::IMREAD_GRAYSCALE);
	if (checkFile(img, imageName) < 0)
		return -1;
	int nrows = img.rows;
	int ncols = img.cols;

	// Foreground data
	cv::String fgName = "../samples/noise_blur_00_fg.bmp";
	cv::Mat fg_mask, fg_points;
	getMaskedPoints(fgName, img, fg_mask, fg_points);
	// Background data
	cv::String bgName = "../samples/noise_blur_00_bg.bmp";
	cv::Mat bg_mask, bg_points;
	getMaskedPoints(bgName, img, bg_mask, bg_points);
	
	// Foreground GMM
	cv::Mat fg_likelihoods;
	estGMM(fg_points, fg_likelihoods);
	//Backvround GMM
	cv::Mat bg_likelihoods;
	estGMM(bg_points, bg_likelihoods);

	// Create the graph
	int num_nodes = nrows*ncols;
	GraphType *g = new GraphType(/*estimated # of nodes*/ num_nodes, /*estimated # of edges*/ num_nodes *3);
	// Add the nodes
	for (int i = 0; i < num_nodes; i++)
		g->add_node();

	// Add t-links
	for (int i = 0; i < nrows; i++) {
		for (int j = 0; j < ncols; j++) {
			if (fg_mask.at<uchar>(i, j) > 0)
				g->add_tweights(i*ncols + j, 1000, 0);
		}
	}
	for (int i = 0; i < nrows; i++) {
		for (int j = 0; j < ncols; j++) {
			if (bg_mask.at<uchar>(i, j) > 0)
				g->add_tweights(i*ncols + j, 0, 1000);
		}
	}

	// add [n-links = e(-diff)] across cols
	for (int j = 0; j < ncols; j++) {
		for (int i = 1; i < nrows; i++) {
			int weight = exp(-(abs(img.at<uchar>(i-1, j) - img.at<uchar>(i, j))));
			g->add_edge((i-1)*ncols + j, i*ncols + j, weight, weight);
		}
	}
	// add [n-links = e(-diff)] across rows
	for (int i = 0; i < nrows; i++) {
		for (int j = 1; j < ncols; j++) {
			double weight = exp(-abs((int)img.at<uchar>(i, j-1) - (int)img.at<uchar>(i, j)));
			g->add_edge(i*ncols + (j-1), i*ncols + j, weight, weight);
		}
	}

	double flow = g->maxflow();

	cv::Mat finalImg = cv::Mat::zeros(nrows, ncols, CV_8UC1);
	for (int i = 0; i < nrows; i++)
		for (int j = 0; j < ncols; j++)
			finalImg.at<uchar>(i, j) = (g->what_segment(i*ncols + j) == GraphType::SOURCE) ? 255 : 0;		

	// Display image
	cv::namedWindow("Display window", cv::WINDOW_AUTOSIZE); // Create a window for display.
	cv::imshow("Display window", finalImg);


	cv::waitKey(0);
	return 0;
}