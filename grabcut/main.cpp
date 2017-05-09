#include <iostream>
#include <string>
#include <fstream>

#include <opencv2/core.hpp>
#include <opencv2/ml.hpp>
#include "opencv2/highgui/highgui.hpp"
//#include "opencv2/imgproc/imgproc.hpp"

#include "graph.h"
#define NUM_CLUS 1
#define LAMBDA 50

typedef Graph<int, int, int> GraphType;

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

int getMask(cv::String& fn, cv::Mat& img, cv::Mat& mask)
{
	mask = cv::imread(fn.c_str(), cv::IMREAD_GRAYSCALE);
	if (checkFile(mask, fn) < 0)
		return -1;
	
	mask = mask < 255; // Turn to binary mask
	cv::bitwise_and(img, mask, mask); // Get only the seeded pixels

	return 0;
}

void estGMM(cv::Mat& mask, cv::Mat& likelihoods)
{
	// Get points
	cv::Mat points = cv::Mat::zeros(cv::countNonZero(mask), 1, CV_32SC1);
	int ctr = 0;
	for (int i = 0; i < mask.rows; i++) {
		for (int j = 0; j < mask.cols; j++) {
			if (mask.at<uchar>(i, j) > 0) {
				points.at<int>(ctr) = mask.at<uchar>(i, j);
				ctr++;
			}
		}
	}

	cv::Ptr<cv::ml::EM> mdl = cv::ml::EM::create();
	mdl->setClustersNumber(NUM_CLUS);
	mdl->trainEM(points, likelihoods);
}

void createGraph(GraphType *g, cv::Mat& img, cv::Mat& fg_seed, cv::Mat& bg_seed, 
				cv::Mat& fg_mask, cv::Mat& bg_mask, cv::Mat& fg_prob, cv::Mat& bg_prob)
{
	int nrows = img.rows;
	int ncols = img.cols;

	// Create the graph
	// Add the nodes
	for (int i = 0; i < nrows*ncols; i++)
		g->add_node();

	// Add [n-links = e(-diff)] - will be the same for every graph
	for (int j = 0; j < ncols; j++) { // across cols
		for (int i = 1; i < nrows; i++) {
			int diff = abs((int)img.at<uchar>(i - 1, j) - (int)img.at<uchar>(i, j));
			double beta = pow(2 * pow(diff, 2), -1);
			double weight = diff > 0 ? LAMBDA*exp(-beta*(diff)) : 0;
			g->add_edge((i - 1)*ncols + j, i*ncols + j, weight, weight);
		}
	}
	for (int i = 0; i < nrows; i++) { // across rows
		for (int j = 1; j < ncols; j++) {
			int diff = abs((int)img.at<uchar>(i, j - 1) - (int)img.at<uchar>(i, j));
			double beta = pow(2 * pow(diff, 2), -1);
			double weight = diff > 0 ? LAMBDA*exp(-beta*diff) : 0;
			g->add_edge(i*ncols + (j - 1), i*ncols + j, weight, weight);
		}
	}
	// Add t-links - changes depending on the mask
	int pidx = 0;
	for (int i = 0; i < nrows; i++) {
		for (int j = 0; j < ncols; j++) {
			// Make sure seeded values won't be cut
			if (fg_seed.at<uchar>(i, j) > 0)
				g->add_tweights(i*ncols + j, 1000, 0);
			else if (fg_mask.at<uchar>(i, j) > 0) {
				g->add_tweights(i*ncols + j, fg_prob.at<double>(pidx), 0);
			}
			if (fg_mask.at<uchar>(i, j) > 0)
				pidx++;
		}
	}
	pidx = 0;
	for (int i = 0; i < nrows; i++) {
		for (int j = 0; j < ncols; j++) {
			// Make sure seeded values won't be cut
			if (bg_seed.at<uchar>(i, j) > 0)
				g->add_tweights(i*ncols + j, 0, 1000);
			else if (bg_mask.at<uchar>(i, j) > 0)
				g->add_tweights(i*ncols + j, bg_prob.at<double>(pidx), 0);
			if (bg_mask.at<uchar>(i, j) > 0)
				pidx++;
		}
	}

	double flow = g->maxflow();
	std::cout << "Flow is: " << flow << std::endl;
}


int main(int argc, char** argv)
{
	// Image name
	cv::String imageName = "../samples/noise_blur_00.png";

	// Open the images
	cv::Mat img = cv::imread(imageName.c_str(), cv::IMREAD_GRAYSCALE);
	if (checkFile(img, imageName) < 0)
		return -1;
	int nrows = img.rows;
	int ncols = img.cols;
	GraphType *g = new GraphType(/*estimated # of nodes*/ nrows*ncols, /*estimated # of edges*/ nrows*ncols * 3);

	// Foreground data
	cv::String fgName = "../samples/noise_blur_00_fg.bmp";
	cv::Mat fg_mask, fg_likelihoods;
	getMask(fgName, img, fg_mask);

	// Background data
	cv::String bgName = "../samples/noise_blur_00_bg.bmp";
	cv::Mat bg_mask, bg_likelihoods;
	getMask(bgName, img, bg_mask);
	
	int iter = 0;
	cv::Mat new_fg_mask = fg_mask.clone();
	cv::Mat new_bg_mask = bg_mask.clone();

	while (iter <= 1) {
		estGMM(new_fg_mask, fg_likelihoods);
		estGMM(new_bg_mask, bg_likelihoods);

		createGraph(g, img, fg_mask, bg_mask, new_fg_mask, new_bg_mask, fg_likelihoods, bg_likelihoods);

		// Reset the masks
		new_fg_mask = cv::Mat::zeros(nrows, ncols, CV_8UC1);
		new_bg_mask = cv::Mat::zeros(nrows, ncols, CV_8UC1);
		for (int i = 0; i < nrows; i++) {
			for (int j = 0; j < ncols; j++) {
				if (g->what_segment(i*ncols + j) == GraphType::SOURCE)
					new_fg_mask.at<uchar>(i, j) = 1;
				else
					new_bg_mask.at<uchar>(i, j) = 1;
			}
		}

		iter++;
		g->reset(); // WITHIN THE LOOP!!
	}


	// Checking
	cv::Mat finalImg = cv::Mat::zeros(nrows, ncols, CV_8UC1);
	for (int i = 0; i < nrows; i++)
		for (int j = 0; j < ncols; j++)
			finalImg.at<uchar>(i, j) = (g->what_segment(i*ncols + j) == GraphType::SOURCE) ? 255 : 0;		

	// Display image
	cv::namedWindow("Display window", cv::WINDOW_AUTOSIZE); // Create a window for display.
	cv::imshow("Display window", finalImg);


	delete g;
	cv::waitKey(0);
	return 0;
}