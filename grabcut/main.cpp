#include <iostream>
#include <string>
#include <fstream>

#include <opencv2/core.hpp>
#include <opencv2/ml.hpp>
#include "opencv2/highgui/highgui.hpp"
//#include "opencv2/imgproc/imgproc.hpp"

#include "graph.h"
#define NUM_CLUS 2 // Number of clusters for GMM Estimation
#define LAMBDA 50 // Graph weight parameter
#define KEEP 10000000 // Thick edge weight

typedef Graph<int, int, int> GraphType;
double beta;
cv::String fileBase = "../samples/pigfoot";

/*
Calculate beta (graph weight parameter)
beta = 1/(2*avg(sqr(||color[i] - color[j]||)))
*/
void calcBeta(const cv::Mat& img, int nrows, int ncols)
{
	beta = 0;
	int num_edges = 0;

	for (int j = 0; j < ncols; j++) { // across cols
		for (int i = 1; i < nrows; i++) {
			int diff = abs((int)img.at<uchar>(i - 1, j) - (int)img.at<uchar>(i, j));
			beta += diff*diff;
			num_edges++;
		}
	}
	for (int i = 0; i < nrows; i++) { // across rows
		for (int j = 1; j < ncols; j++) {
			int diff = abs((int)img.at<uchar>(i, j - 1) - (int)img.at<uchar>(i, j));
			beta += diff*diff;
			num_edges++;
		}
	}

	beta = 1.f / (2 * beta / num_edges);
}

/*
Function to check if file can be found
*/
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

/*
Function to get the mask given a filename and turn into a binary mask
*/
int getMask(cv::String& fn, cv::Mat& img, cv::Mat& mask)
{
	mask = cv::imread(fn.c_str(), cv::IMREAD_GRAYSCALE);
	if (checkFile(mask, fn) < 0)
		return -1;
	
	mask = mask < 255; // Turn to binary mask
	cv::bitwise_and(img, mask, mask); // Get only the seeded pixels

	return 0;
}

/*
Given the image and mask, change points into a 1D matrix to be passed for GMM training
*/
void estGMM(cv::Mat& mask, cv::Mat& img, cv::Mat& likelihoods)
{
	// Get points
	cv::Mat points = cv::Mat::zeros(cv::countNonZero(mask), 1, CV_32SC1);
	int ctr = 0;
	for (int i = 0; i < mask.rows; i++) {
		for (int j = 0; j < mask.cols; j++) {
			if (mask.at<uchar>(i, j) > 0) {
				points.at<int>(ctr) = img.at<uchar>(i, j);
				ctr++;
			}
		}
	}

	cv::Ptr<cv::ml::EM> mdl = cv::ml::EM::create();
	mdl->setClustersNumber(NUM_CLUS);
	mdl->trainEM(points, likelihoods);
}

/*
Create the graph for foreground/background segmentation and run cut
*/
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
			double weight = diff > 0 ? LAMBDA*exp(-beta*(diff*diff)) : KEEP;
			if (weight < 0)
				std::cout << "Weight: " << weight << std::endl;
			g->add_edge((i - 1)*ncols + j, i*ncols + j, weight, weight);
		}
	}
	for (int i = 0; i < nrows; i++) { // across rows
		for (int j = 1; j < ncols; j++) {
			int diff = abs((int)img.at<uchar>(i, j - 1) - (int)img.at<uchar>(i, j));
			double weight = diff > 0 ? LAMBDA*exp(-beta*(diff*diff)) : KEEP;
			g->add_edge(i*ncols + (j - 1), i*ncols + j, weight, weight);
		}
	}
	// Add t-links - changes depending on the mask
	int pidx = 0;
	for (int i = 0; i < nrows; i++) {
		for (int j = 0; j < ncols; j++) {
			// Make sure seeded values won't be cut
			if (fg_seed.at<uchar>(i, j) > 0)
				g->add_tweights(i*ncols + j, KEEP, 0);
			else if (fg_mask.at<uchar>(i, j) > 0)
				g->add_tweights(i*ncols + j, -fg_prob.at<double>(pidx), 0);
			if (fg_mask.at<uchar>(i, j) > 0)
				pidx++;
		}
	}
	pidx = 0;
	for (int i = 0; i < nrows; i++) {
		for (int j = 0; j < ncols; j++) {
			// Make sure seeded values won't be cut
			if (bg_seed.at<uchar>(i, j) > 0)
				g->add_tweights(i*ncols + j, 0, KEEP);
			else if (bg_mask.at<uchar>(i, j) > 0)
				g->add_tweights(i*ncols + j, 0, -bg_prob.at<double>(pidx));
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
	cv::String imageName = fileBase+".png";

	// Open the images
	cv::Mat img = cv::imread(imageName.c_str(), cv::IMREAD_GRAYSCALE);
	if (checkFile(img, imageName) < 0)
		return -1;
	int nrows = img.rows;
	int ncols = img.cols;
	GraphType *g = new GraphType(/*estimated # of nodes*/ nrows*ncols, /*estimated # of edges*/ nrows*ncols * 3);
	calcBeta(img, nrows, ncols); // Set the beta value

	// Foreground data
	cv::String fgName = fileBase+"_fg.bmp";
	cv::Mat fg_mask, fg_likelihoods;
	getMask(fgName, img, fg_mask);

	// Background data
	cv::String bgName = fileBase+"_bg.bmp";
	cv::Mat bg_mask, bg_likelihoods;
	getMask(bgName, img, bg_mask);
	
	int iter = 0;
	cv::Mat new_fg_mask = fg_mask.clone();
	cv::Mat new_bg_mask = bg_mask.clone();

	while (iter <= 2) {
		std::cout << "RUN #" << iter << "\n=========================\n";
		std::cout << "Estimating GMMs\n";
		estGMM(new_fg_mask, img, fg_likelihoods);
		estGMM(new_bg_mask, img, bg_likelihoods);

		std::cout << "Creating Graph\n";
		createGraph(g, img, fg_mask, bg_mask, new_fg_mask, new_bg_mask, fg_likelihoods, bg_likelihoods);

		// Reset the masks
		std::cout << "Recalculating foreground/background\n\n";
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
		g->reset(); // Reset the graph so don't have to allocate memory again
	}

	// Checking - creating an image to display results
	cv::Mat finalImg = cv::Mat::zeros(nrows, ncols, CV_8UC1);
	for (int i = 0; i < nrows; i++)
		for (int j = 0; j < ncols; j++)
			finalImg.at<uchar>(i, j) = (g->what_segment(i*ncols + j) == GraphType::SOURCE) ? 255 : 0;		

	// Display image
	cv::namedWindow("Display window", cv::WINDOW_AUTOSIZE); // Create a window for display.
	cv::imshow("Display window", finalImg);

	delete g;
	std::cout << "Done!";
	cv::waitKey(0);
	return 0;
}