#include <iostream>
#include <string>
#include <fstream>

#include <opencv2/core.hpp>
#include <opencv2/ml.hpp>
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/imgproc/imgproc.hpp"

#include "graph.h"
#define NUM_CLUS_FG 5 // Number of clusters for fg GMM Estimation
#define NUM_CLUS_BG 3 // Number of clusters for bg GMM Estimation
#define LAMBDA 50 // Graph weight parameter
#define KEEP 10000000 // Thick edge weight
#define MAX_ITER 20
#define EIGHT_CON true

typedef Graph<double, double, double> GraphType;
double beta;
cv::String fileBase = "../samples/noise_blur_12";

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
void estGMM(cv::Mat& mask, cv::Mat& img, cv::Mat& means, std::vector<cv::Mat>& covs, cv::Mat& weights, int num_clus, bool init = false, bool save = false)
{
	// Creating files
	std::ofstream point_file, gmm_file;
	if (save) {
		// Points file
		if (std::ifstream("../gmm_test/fg_points"))
			point_file.open("../gmm_test/bg_points");
		else
			point_file.open("../gmm_test/fg_points");

		// GMM file
		if (std::ifstream("../gmm_test/fg_gmm"))
			gmm_file.open("../gmm_test/bg_gmm");
		else
			gmm_file.open("../gmm_test/fg_gmm");
	}

	// Get points
	cv::Mat sample_points = cv::Mat::zeros(cv::countNonZero(mask), 1, CV_32SC1);
	int ctr = 0;
	for (int i = 0; i < mask.rows; i++) {
		for (int j = 0; j < mask.cols; j++) {
			if (mask.at<double>(i, j) > 0) {
				sample_points.at<int>(ctr) = img.at<uchar>(i, j);
				ctr++;
				if (save) // save points into file
					point_file << (int)img.at<uchar>(i, j) << "\n";
			}
		}
	}
	
	// Put prior GMM information into file
	if (save) {
		for (int i = 0; i < num_clus; i++)
			gmm_file << means.at<double>(i, 0) << std::endl;
		gmm_file << std::endl;
		for (int i = 0; i < num_clus; i++)
			gmm_file << weights.at<double>(0, i) << std::endl;
		gmm_file << std::endl;
		for (int i = 0; i < num_clus; i++)
			gmm_file << covs[i].at<double>(0, 0) << std::endl;
		gmm_file << std::endl << std::endl;
	}

	cv::Mat likelihoods;
	cv::Ptr<cv::ml::EM> mdl = cv::ml::EM::create();
	mdl->setClustersNumber(num_clus);
	// If first run - base training on points. Otherwise, use mean/covs as base
	if (init) 
		mdl->trainEM(sample_points, likelihoods);
	else 
		mdl->trainE(sample_points, means, covs, weights, likelihoods);
	
	means = mdl->getMeans();
	mdl->getCovs(covs);
	weights = mdl->getWeights();

	// Put after GMM information into file
	if (save) {
		for (int i = 0; i < num_clus; i++)
			gmm_file << means.at<double>(i, 0) << std::endl;
		gmm_file << std::endl;
		for (int i = 0; i < num_clus; i++)
			gmm_file << weights.at<double>(0, i) << std::endl;
		gmm_file << std::endl;
		for (int i = 0; i < num_clus; i++)
			gmm_file << covs[i].at<double>(0, 0) << std::endl;
	}

	// Close files
	point_file.close();
	gmm_file.close();

	// Set the likelihoods to the masks
	ctr = 0;
	for (int i = 0; i < mask.rows; i++) {
		for (int j = 0; j < mask.cols; j++) {
			if (mask.at<double>(i, j) > 0) {
				mask.at<double>(i, j) = likelihoods.at<double>(ctr);
				ctr++;
			}
			else { // Predict rest of points (not in mask)
				cv::Mat pt = cv::Mat::zeros(1, 1, CV_32SC1); 
				pt.at<int>(0) = img.at<uchar>(i, j);
				cv::Vec2d point_likelihood;
				point_likelihood = mdl->predict2(pt, cv::noArray());
				mask.at<double>(i, j) = point_likelihood(0);
			}
		}
	}

}


/*
Create the graph for foreground/background segmentation and run cut
*/
void createGraph(GraphType *g, cv::Mat& img, cv::Mat& fg_seed, cv::Mat& bg_seed, 
				cv::Mat& fg_prob, cv::Mat& bg_prob)
{
	int nrows = img.rows;
	int ncols = img.cols;

	// Create the graph
	// Add the nodes
	for (int i = 0; i < nrows*ncols; i++)
		g->add_node();

	// Add [n-links = e(-diff)] - will be the same for every graph
	for (int j = 0; j < ncols; j++) { // across rows
		for (int i = 1; i < nrows; i++) {
			int diff = abs((int)img.at<uchar>(i - 1, j) - (int)img.at<uchar>(i, j));
			double weight = diff > 0 ? LAMBDA*exp(-beta*(diff*diff)) : KEEP;
			g->add_edge((i - 1)*ncols + j, i*ncols + j, weight, weight);

			// if 8-econnected
			if (EIGHT_CON) {
				if (j < (ncols - 1)) { // if it's not the last column, add right-top (starting at i = 1 so don't have to check)
					diff = abs((int)img.at<uchar>(i - 1, j + 1) - (int)img.at<uchar>(i, j));
					weight = diff > 0 ? LAMBDA*exp(-beta*(diff*diff)) : KEEP;
					g->add_edge((i - 1)*ncols + (j + 1), i*ncols + j, weight, weight); // weight different
				}

				if (j > 0) {  // if it's not the first column, add left-top (starting at i=1 so don't have to check)
					diff = abs((int)img.at<uchar>(i - 1, j - 1) - (int)img.at<uchar>(i, j));
					weight = diff > 0 ? LAMBDA*exp(-beta*(diff*diff)) : KEEP;
					g->add_edge((i - 1)*ncols + (j - 1), i*ncols + j, weight, weight); // weight different
				}
			}
		}
	}
	for (int i = 0; i < nrows; i++) { // across cols
		for (int j = 1; j < ncols; j++) {
			int diff = abs((int)img.at<uchar>(i, j - 1) - (int)img.at<uchar>(i, j));
			double weight = diff > 0 ? LAMBDA*exp(-beta*(diff*diff)) : KEEP;
			g->add_edge(i*ncols + (j - 1), i*ncols + j, weight, weight);
		}
	}
	// Add t-links - changes depending on the mask
	for (int i = 0; i < nrows; i++) {
		for (int j = 0; j < ncols; j++) {
			// Make sure seeded values won't be cut
			if (fg_seed.at<uchar>(i, j) > 0) {
				g->add_tweights(i*ncols + j, KEEP, 0); 
			} 
			else if (bg_seed.at<uchar>(i, j) > 0) {
				g->add_tweights(i*ncols + j, 0, KEEP); 
			}
			else {
				g->add_tweights(i*ncols + j, -bg_prob.at<double>(i, j), -fg_prob.at<double>(i, j));
			}
		}
	}

	double flow = g->maxflow();
	std::cout << "Flow is: " << flow << std::endl;
}

double getEnergy(cv::Mat& img, cv::Mat& fg_seed, cv::Mat& fg_mask, cv::Mat& fg_prob, cv::Mat& bg_seed, cv::Mat& bg_mask, cv::Mat& bg_prob)
{
	int nrows = fg_mask.rows;
	int ncols = fg_mask.cols;
	int diff;
	double energy = 0, weight;


	// n-links
	for (int j = 0; j < ncols; j++) { // across rows
		for (int i = 1; i < nrows; i++) {
			// Check if edge was cut
			if (fg_mask.at<double>(i - 1, j) != fg_mask.at<double>(i, j)) {
				diff = abs((int)img.at<uchar>(i - 1, j) - (int)img.at<uchar>(i, j));
				weight = diff > 0 ? LAMBDA*exp(-beta*(diff*diff)) : KEEP;
				energy += weight;
			}
			
			// If 8-connected
			if (EIGHT_CON) {
				// right-top diagonals
				if (j < (ncols - 1) && (fg_mask.at<double>(i - 1, j + 1) != fg_mask.at<double>(i, j))) {
					diff = abs((int)img.at<uchar>(i - 1, j + 1) - (int)img.at<uchar>(i, j));
					weight = diff > 0 ? LAMBDA*exp(-beta*(diff*diff)) : KEEP;
					energy += weight;
				}

				// left-top diagons
				if (j > 0 && (fg_mask.at<double>(i - 1, j - 1) != fg_mask.at<double>(i, j))) {
					diff = abs((int)img.at<uchar>(i - 1, j - 1) - (int)img.at<uchar>(i, j));
					weight = diff > 0 ? LAMBDA*exp(-beta*(diff*diff)) : KEEP;
					energy += weight;
				}
			}
		}
	}
	for (int i = 0; i < nrows; i++) { // across cols
		for (int j = 1; j < ncols; j++) {
			// Check if edge was cut
			if (fg_mask.at<double>(i, j - 1) != fg_mask.at<double>(i, j)) {
				diff = abs((int)img.at<uchar>(i, j - 1) - (int)img.at<uchar>(i, j));
				weight = diff > 0 ? LAMBDA*exp(-beta*(diff*diff)) : KEEP;
				energy += weight;
			}
		}
	}
	double ncut_energy = energy;
	std::cout << "N-Link ENERGY: " << ncut_energy << std::endl;


	// t-links
	for (int i = 0; i < nrows; i++) {
		for (int j = 0; j < ncols; j++) {
			if ((fg_seed.at<uchar>(i, j) > 0) || (bg_seed.at<uchar>(i, j) > 0))
				energy = energy;
			else if (fg_mask.at<double>(i, j) > 0)
				energy += -fg_prob.at<double>(i, j); 
			else 
				energy += -bg_prob.at<double>(i, j);
		}
	}

	std::cout << "T-Link ENERGY: " << (energy-ncut_energy) << std::endl;

	return energy;
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
	cv::Mat fg_means, bg_means; // mat for GMM means
	std::vector<cv::Mat> fg_covs, bg_covs; // mat for GMM covs
	cv::Mat fg_weights, bg_weights; // weights for GMM means

	// Foreground data
	cv::String fgName = fileBase+"_fg.bmp";
	cv::Mat fg_mask;
	getMask(fgName, img, fg_mask);

	// Background data
	cv::String bgName = fileBase+"_bg.bmp";
	cv::Mat bg_mask;
	getMask(bgName, img, bg_mask);
	
	cv::Mat new_fg_mask = fg_mask.clone();
	new_fg_mask.convertTo(new_fg_mask, CV_64FC1);
	cv::Mat new_bg_mask = bg_mask.clone();
	new_bg_mask.convertTo(new_bg_mask, CV_64FC1);

	cv::Mat fg_prob = new_fg_mask.clone();
	cv::Mat bg_prob = new_bg_mask.clone();
	estGMM(fg_prob, img, fg_means, fg_covs, fg_weights, NUM_CLUS_FG, true);
	estGMM(bg_prob, img, bg_means, bg_covs, bg_weights, NUM_CLUS_BG, true);
	getEnergy(img, fg_mask, new_fg_mask, fg_prob, bg_mask, new_bg_mask, bg_prob);

	if (EIGHT_CON)
		std::cout << "Running 8-con Graph\n\n";
	else
		std::cout << "Running 4-con Graph\n\n";

	double prev_energy = 0, energy = -1;
	int iter = 0;
	while (iter < MAX_ITER) {
		std::cout << "RUN #" << iter << "\n=========================\n";
		std::cout << "Creating Graph\n";
		createGraph(g, img, fg_mask, bg_mask, fg_prob, bg_prob);

		// Reset the masks
		std::cout << "Recalculating foreground/background\n";
		new_fg_mask = cv::Mat::zeros(nrows, ncols, CV_64FC1);
		new_bg_mask = cv::Mat::zeros(nrows, ncols, CV_64FC1);
		for (int i = 0; i < nrows; i++) {
			for (int j = 0; j < ncols; j++) {
				if (g->what_segment(i*ncols + j) == GraphType::SOURCE)
					new_fg_mask.at<double>(i, j) = 1;
				else
					new_bg_mask.at<double>(i, j) = 1;
			}
		}
		prev_energy = energy;
		energy = getEnergy(img, fg_mask, new_fg_mask, fg_prob, bg_mask, new_bg_mask, bg_prob);
		std::cout.precision(17);
		std::cout << "Graph Energy is: " << energy << std::endl << std::endl;
		if ((prev_energy >= 0) && (prev_energy < energy)) {
			std::cout << "ERROR: ENERGY INCREASED.";
			//system("pause");
			//return -1;
		}

		// Figuring out increasing energy issue
		if (iter == 3)
		{

		}

		std::cout << "Estimating GMMs\n";
		fg_prob = new_fg_mask.clone();
		bg_prob = new_bg_mask.clone();

		// TEMP: Figuring out increasing energy issue
		if (iter == 3) {
			estGMM(fg_prob, img, fg_means, fg_covs, fg_weights, NUM_CLUS_FG, false, true);
			estGMM(bg_prob, img, bg_means, bg_covs, bg_weights, NUM_CLUS_BG, false, true);
		}
		else {
			estGMM(fg_prob, img, fg_means, fg_covs, fg_weights, NUM_CLUS_FG);
			estGMM(bg_prob, img, bg_means, bg_covs, bg_weights, NUM_CLUS_BG);
		}

		prev_energy = energy;
		energy = getEnergy(img, fg_mask, new_fg_mask, fg_prob, bg_mask, new_bg_mask, bg_prob);
		std::cout << "GMM Energy is: " << energy << std::endl << std::endl;
		if ((prev_energy >= 0) && (prev_energy < energy)) {
			std::cout << "ERROR: ENERGY INCREASED.";
			//system("pause");
			//return -1;
		}

		iter++;
		g->reset(); // Reset the graph so don't have to allocate memory again
		if (abs(prev_energy - energy) < 0.01)
			break;
	}

	// Checking - creating an image to display results

	// Checking - creating an image to display results
	cv::Mat finalMask = cv::Mat::zeros(nrows, ncols, CV_8UC1);
	for (int i = 0; i < nrows; i++)
		for (int j = 0; j < ncols; j++)
			finalMask.at<uchar>(i, j) = new_fg_mask.at<double>(i, j) ? 255 : 0;
	cv::imwrite(fileBase + "_mask.bmp", finalMask);


	// Creating overlayed image
	cv::Mat overlayImg;
	cv::cvtColor(img, overlayImg, cv::COLOR_GRAY2RGB);

	for (int i = 0; i < nrows; i++) {
		for (int j = 0; j < ncols; j++) {
			if (new_fg_mask.at<double>(i, j)) {
				overlayImg.at<cv::Vec3b>(i, j)[0] = 0;
				overlayImg.at<cv::Vec3b>(i, j)[1] = 0;
				overlayImg.at<cv::Vec3b>(i, j)[2] = 255;
			}
		}
	}
	cv::imwrite(fileBase + "_overlay.png", overlayImg);



	// Display image
	cv::namedWindow("Display window", cv::WINDOW_AUTOSIZE); // Create a window for display.
	cv::imshow("Display window", overlayImg);

	delete g;
	std::cout << "Done!";
	cv::waitKey(0);
	return 0;
}
