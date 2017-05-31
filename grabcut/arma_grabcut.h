#pragma once
#include <iostream>
#include <string>
#include <fstream>
#include <armadillo>

#include "opencv2/highgui/highgui.hpp"
#include "opencv2/imgproc/imgproc.hpp"

#include "graph.h"
#include "common.h"

void ocv_to_arma(cv::Mat& cv_img, arma::Mat<short>& arma_img)
{
	arma_img.set_size(cv_img.rows, cv_img.cols);
	for (int i = 0; i < cv_img.rows; i++)
		for (int j = 0; j < cv_img.cols; j++)
			arma_img(i, j) = cv_img.at<uchar>(i, j);
}

void arma_to_ocv(arma::Mat<short>& arma_img, cv::Mat& cv_img)
{
	cv_img = cv::Mat::zeros(arma_img.n_rows, arma_img.n_cols, CV_8UC1);
	for (int i = 0; i < arma_img.n_rows; i++)
		for (int j = 0; j < arma_img.n_cols; j++)
			cv_img.at<uchar>(i, j) = (uchar)arma_img(i, j);
}


/*
Function to get the mask given a filename and turn into a binary mask
*/
int arm_getMask(std::string& fn, arma::Mat<short>& img, arma::Mat<short>& mask)
{
	cv::Mat cv_mask;
	cv_mask = cv::imread(fn.c_str(), cv::IMREAD_GRAYSCALE);
	if (checkFile<cv::Mat>(cv_mask, fn) < 0)
		return -1;

	cv_mask = cv_mask < 255; // Turn to binary mask

	ocv_to_arma(cv_mask, mask);
	mask = img % mask; // Get only the seeded pixels

	return 0;
}

/*
Given the image and mask, change points into a 1D matrix to be passed for GMM training
*/
void arm_estGMM(arma::mat& mask, arma::Mat<short>& img, arma::mat& means, arma::mat& covs, arma::Row<double>& weights, int num_clus, bool init = false, bool save = false)
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
	arma::vec nonzeros = arma::nonzeros(mask);
	arma::mat sample_points = arma::mat(1, arma::size(nonzeros)(0), arma::fill::zeros);
	int ctr = 0;
	for (int i = 0; i < mask.n_rows; i++) {
		for (int j = 0; j < mask.n_cols; j++) {
			if (mask(i, j) > 0) {
				sample_points(ctr) = img(i, j);
				ctr++;
				if (save) // save points into file
					point_file << (int)img(i, j) << "\n";
			}
		}
	}

	// Put prior GMM information into file
	if (save) {
		for (int i = 0; i < num_clus; i++)
			gmm_file << means(i, 0) << std::endl;
		gmm_file << std::endl;
		for (int i = 0; i < num_clus; i++)
			gmm_file << weights(0, i) << std::endl;
		gmm_file << std::endl;
		for (int i = 0; i < num_clus; i++)
			gmm_file << covs(0, i) << std::endl;
		gmm_file << std::endl << std::endl;
	}

	arma::gmm_diag mdl;
	bool status;
	if (init) {
		status = mdl.learn(sample_points, num_clus, arma::maha_dist, arma::random_subset, 10, 5, 1e-10, false);
	} else {

		mdl.set_params(means, covs, weights);
		status = mdl.learn(sample_points, num_clus, arma::maha_dist, arma::keep_existing, 10, 5, 1e-10, false);
	}

	if (status == false)
		std::cout << "\n\n*** LEARNING FAILED ***\n\n" << std::endl;

	means = mdl.means;
	covs = mdl.dcovs;
	// Make sure sum of hefts == 1
	weights = mdl.hefts;
	weights(0, num_clus - 1) = weights(0, num_clus - 1) + (1 - accu(weights));

	// Put after GMM information into file
	if (save) {
		for (int i = 0; i < num_clus; i++)
			gmm_file << means(0, 1) << std::endl;
		gmm_file << std::endl;
		for (int i = 0; i < num_clus; i++)
			gmm_file << weights(0, i) << std::endl;
		gmm_file << std::endl;
		for (int i = 0; i < num_clus; i++)
			gmm_file << covs(0, i) << std::endl;

		// Close files
		point_file.close();
		gmm_file.close();
	}

	// Set the likelihoods to the masks
	ctr = 0;
	for (int i = 0; i < mask.n_rows; i++) {
		for (int j = 0; j < mask.n_cols; j++) {
			arma::vec pt(1);
			pt(0) = img(i, j);
			mask(i, j) = mdl.log_p(pt);
		}
	}

}


int run_arma()
{
	std::string imageName = fileBase + ".png";

	// Open the images
	cv::Mat cv_img = cv::imread(imageName.c_str(), cv::IMREAD_GRAYSCALE);
	arma::Mat<short> img;
	ocv_to_arma(cv_img, img);

	if (checkFile<arma::Mat<short>>(img, imageName) < 0)
		return -1;

	int nrows = numRows<arma::Mat<short>>(img);
	int ncols = numCols<arma::Mat<short>>(img);

	GraphType *g = new GraphType(/*estimated # of nodes*/ nrows*ncols, /*estimated # of edges*/ nrows*ncols * 3);
	calcBeta<arma::Mat<short>>(img, nrows, ncols); // Set the beta value

	arma::mat fg_means, bg_means; // mat for GMM means
	arma::mat fg_covs, bg_covs; // mat for GMM covs
	arma::Row<double> fg_weights, bg_weights; // weights for GMM means

	// Foreground data
	std::string fgName = fileBase + "_fg.bmp";
	arma::Mat<short> fg_mask;
	arm_getMask(fgName, img, fg_mask);

	// Background data
	std::string bgName = fileBase + "_bg.bmp";
	arma::Mat<short> bg_mask;
	arm_getMask(bgName, img, bg_mask);

	arma::mat new_fg_mask = arma::conv_to<arma::mat>::from(fg_mask);
	arma::mat new_bg_mask = arma::conv_to<arma::mat>::from(fg_mask);
	arma::mat fg_prob = new_fg_mask;
	arma::mat bg_prob = new_bg_mask;
	arm_estGMM(fg_prob, img, fg_means, fg_covs, fg_weights, NUM_CLUS_FG, true);
	arm_estGMM(bg_prob, img, bg_means, bg_covs, bg_weights, NUM_CLUS_BG, true);
	getEnergy<arma::Mat<short>, arma::mat>(img, fg_mask, new_fg_mask, fg_prob, bg_mask, new_bg_mask, bg_prob);

	if (EIGHT_CON)
		std::cout << "Running 8-con Graph\n\n";
	else
		std::cout << "Running 4-con Graph\n\n";

	double prev_energy = 0, energy = -1;
	int iter = 0;
	while (iter < MAX_RUNS) {
		std::cout << "RUN #" << iter << "\n=========================\n";
		std::cout << "Creating Graph\n";
		createGraph<arma::Mat<short>, arma::mat>(g, img, fg_mask, bg_mask, fg_prob, bg_prob);

		// Reset the masks
		std::cout << "Recalculating foreground/background\n";
		new_fg_mask = arma::mat(nrows, ncols, arma::fill::zeros);
		new_bg_mask = arma::mat(nrows, ncols, arma::fill::zeros);
		for (int i = 0; i < nrows; i++) {
			for (int j = 0; j < ncols; j++) {
				if (g->what_segment(i*ncols + j) == GraphType::SOURCE)
					new_fg_mask(i, j) = 1;
				else
					new_bg_mask(i, j) = 1;
			}
		}
		prev_energy = energy;
		energy = getEnergy<arma::Mat<short>, arma::mat>(img, fg_mask, new_fg_mask, fg_prob, bg_mask, new_bg_mask, bg_prob);
		std::cout.precision(17);
		std::cout << "Graph Energy is: " << energy << std::endl << std::endl;
		if ((prev_energy >= 0) && (prev_energy < energy)) {
			std::cout << "ERROR: ENERGY INCREASED.";
			//system("pause");
			//return -1;
		}

		std::cout << "Estimating GMMs\n";
		fg_prob = new_fg_mask;
		bg_prob = new_bg_mask;

		// TEMP: Figuring out increasing energy issue
		arm_estGMM(fg_prob, img, fg_means, fg_covs, fg_weights, NUM_CLUS_FG);
		arm_estGMM(bg_prob, img, bg_means, bg_covs, bg_weights, NUM_CLUS_BG);

		prev_energy = energy;
		energy = getEnergy<arma::Mat<short>, arma::mat>(img, fg_mask, new_fg_mask, fg_prob, bg_mask, new_bg_mask, bg_prob);
		std::cout << "GMM Energy is: " << energy << std::endl << std::endl;
		if ((prev_energy >= 0) && (prev_energy < energy)) {
			std::cout << "ERROR: ENERGY INCREASED.";
			//system("pause");
			//return -1;
		}

		iter++;
		g->reset(); // Reset the graph so don't have to allocate memory again
		if (std::abs(prev_energy - energy) < 0.01)
			break;
	}

	// Checking - creating an image to display results
	cv::Mat finalMask = cv::Mat::zeros(nrows, ncols, CV_8UC1);
	for (int i = 0; i < nrows; i++)
		for (int j = 0; j < ncols; j++)
			finalMask.at<uchar>(i, j) = new_fg_mask(i, j) ? 255 : 0;
	cv::imwrite(fileBase + "_mask.bmp", finalMask);


	// Creating overlayed image
	cv::Mat overlayImg;
	cv::cvtColor(cv_img, overlayImg, cv::COLOR_GRAY2RGB);

	for (int i = 0; i < nrows; i++) {
		for (int j = 0; j < ncols; j++) {
			if (new_fg_mask(i, j)) {
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