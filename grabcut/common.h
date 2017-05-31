#pragma once
#include <string>
#include "graph.h"

#define NUM_CLUS_FG 2 // Number of clusters for fg GMM Estimation
#define NUM_CLUS_BG 2 // Number of clusters for bg GMM Estimation
#define LAMBDA 50 // Graph weight parameter
#define KEEP 10000000 // Thick edge weight
#define MAX_RUNS 20
#define EIGHT_CON true
// Define library to use [OCV, ARM]
//#define OCV
#define ARM

// 
typedef Graph<double, double, double> GraphType;
std::string fileBase = "../samples/pigfoot";

/* Depending on library, different way of accessing the matrix */
template<typename datatype, typename paramtype>
paramtype getPix(datatype& data, int i, int j)
{
	#ifdef OCV
		return data.at<paramtype>(i, j);
	#elif defined(ARM)
		return data(i, j);
	#endif
}

template<typename datatype>
int numRows(datatype& data) 
{
	#ifdef OCV
		return data.rows;
	#elif defined(ARM)
		return data.n_rows;
	#endif
}

template<typename datatype>
int numCols(datatype& data)
{
	#ifdef OCV
		return data.cols;
	#elif defined(ARM)
		return data.n_cols;
	#endif
}

/*
Function to check if file can be found
*/
template<typename datatype>
int checkFile(datatype& file, std::string& filename)
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
Calculate beta (graph weight parameter)
beta = 1/(2*avg(sqr(||color[i] - color[j]||)))
*/
template<typename datatype>
double calcBeta(datatype& img, int nrows, int ncols)
{
	double beta = 0;
	int num_edges = 0;

	for (int j = 0; j < ncols; j++) { // across cols
		for (int i = 1; i < nrows; i++) {
			int diff = abs((int)getPix<datatype, uchar>(img, i - 1, j) - (int)getPix<datatype, uchar>(img, i, j));
			beta += diff*diff;
			num_edges++;
		}
	}
	for (int i = 0; i < nrows; i++) { // across rows
		for (int j = 1; j < ncols; j++) {
			int diff = abs((int)getPix<datatype, uchar>(img, i, j - 1) - (int)getPix<datatype, uchar>(img, i, j));
			beta += diff*diff;
			num_edges++;
		}
	}

	beta = 1.f / (2 * beta / num_edges);
	return beta;
}


/*
Create the graph for foreground/background segmentation and run cut
*/
template <typename imgtype, typename datatype>
void createGraph(GraphType *g, imgtype& img, imgtype& fg_seed, imgtype& bg_seed,
	datatype& fg_prob, datatype& bg_prob)
{
	int nrows = numRows<imgtype>(img);
	int ncols = numCols<imgtype>(img);
	double beta = calcBeta<imgtype>(img, nrows, ncols);

	// Create the graph
	// Add the nodes
	for (int i = 0; i < nrows*ncols; i++)
		g->add_node();

	// Add [n-links = e(-diff)] - will be the same for every graph
	for (int j = 0; j < ncols; j++) { // across rows
		for (int i = 1; i < nrows; i++) {
			int diff = abs((int)getPix<imgtype, uchar>(img, i - 1, j) - (int)getPix<imgtype, uchar>(img, i, j));
			double weight = diff > 0 ? LAMBDA*exp(-beta*(diff*diff)) : KEEP;
			g->add_edge((i - 1)*ncols + j, i*ncols + j, weight, weight);

			// if 8-econnected
			if (EIGHT_CON) {
				if (j < (ncols - 1)) { // if it's not the last column, add right-top (starting at i = 1 so don't have to check)
					diff = abs((int)getPix<imgtype, uchar>(img, i - 1, j + 1) - (int)getPix<imgtype, uchar>(img, i, j));
					weight = diff > 0 ? LAMBDA*exp(-beta*(diff*diff)) : KEEP;
					g->add_edge((i - 1)*ncols + (j + 1), i*ncols + j, weight, weight); // weight different
				}

				if (j > 0) {  // if it's not the first column, add left-top (starting at i=1 so don't have to check)
					diff = abs((int)getPix<imgtype, uchar>(img, i - 1, j - 1) - (int)getPix<imgtype, uchar>(img, i, j));
					weight = diff > 0 ? LAMBDA*exp(-beta*(diff*diff)) : KEEP;
					g->add_edge((i - 1)*ncols + (j - 1), i*ncols + j, weight, weight); // weight different
				}
			}
		}
	}
	for (int i = 0; i < nrows; i++) { // across cols
		for (int j = 1; j < ncols; j++) {
			int diff = abs((int)getPix<imgtype, uchar>(img, i, j - 1) - (int)getPix<imgtype, uchar>(img, i, j));
			double weight = diff > 0 ? LAMBDA*exp(-beta*(diff*diff)) : KEEP;
			g->add_edge(i*ncols + (j - 1), i*ncols + j, weight, weight);
		}
	}
	// Add t-links - changes depending on the mask
	for (int i = 0; i < nrows; i++) {
		for (int j = 0; j < ncols; j++) {
			// Make sure seeded values won't be cut
			if (getPix<imgtype, uchar>(fg_seed, i, j) > 0) {
				g->add_tweights(i*ncols + j, KEEP, 0);
			}
			else if (getPix<imgtype, uchar>(bg_seed, i, j) > 0) {
				g->add_tweights(i*ncols + j, 0, KEEP);
			}
			else {
				g->add_tweights(i*ncols + j, -getPix<datatype, double>(bg_prob, i, j), -getPix<datatype, double>(fg_prob, i, j));
			}
		}
	}

	double flow = g->maxflow();
	std::cout << "Flow is: " << flow << std::endl;
}

template <typename imgtype, typename datatype>
double getEnergy(imgtype& img, imgtype& fg_seed, datatype& fg_mask, datatype& fg_prob, imgtype& bg_seed, datatype& bg_mask, datatype& bg_prob)
{
	int nrows = numRows<imgtype>(img);
	int ncols = numCols<imgtype>(img);
	int diff;
	double energy = 0, weight;
	double beta = calcBeta<imgtype>(img, nrows, ncols);


	// n-links
	for (int j = 0; j < ncols; j++) { // across rows
		for (int i = 1; i < nrows; i++) {
			// Check if edge was cut
			if (getPix<datatype, double>(fg_mask, i - 1, j) != getPix<datatype, double>(fg_mask, i, j)) {
				diff = abs((int)getPix<imgtype, uchar>(img, i - 1, j) - (int)getPix<imgtype, uchar>(img, i, j));
				weight = diff > 0 ? LAMBDA*exp(-beta*(diff*diff)) : KEEP;
				energy += weight;
			}

			// If 8-connected
			if (EIGHT_CON) {
				// right-top diagonals
				if (j < (ncols - 1) && (getPix<datatype, double>(fg_mask, i - 1, j + 1) != getPix<datatype, double>(fg_mask, i, j))) {
					diff = abs((int)getPix<imgtype, uchar>(img, i - 1, j + 1) - (int)getPix<imgtype, uchar>(img, i, j));
					weight = diff > 0 ? LAMBDA*exp(-beta*(diff*diff)) : KEEP;
					energy += weight;
				}

				// left-top diagons
				if (j > 0 && (getPix<datatype, double>(fg_mask, i - 1, j - 1) != getPix<datatype, double>(fg_mask, i, j))) {
					diff = abs((int)getPix<imgtype, uchar>(img, i - 1, j - 1) - (int)getPix<imgtype, uchar>(img, i, j));
					weight = diff > 0 ? LAMBDA*exp(-beta*(diff*diff)) : KEEP;
					energy += weight;
				}
			}
		}
	}
	for (int i = 0; i < nrows; i++) { // across cols
		for (int j = 1; j < ncols; j++) {
			// Check if edge was cut
			if (getPix<datatype, double>(fg_mask, i, j - 1) != getPix<datatype, double>(fg_mask, i, j)) {
				diff = abs((int)getPix<imgtype, uchar>(img, i, j - 1) - (int)getPix<imgtype, uchar>(img, i, j));
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
			if ((getPix<imgtype, uchar>(fg_seed, i, j) > 0) || (getPix<imgtype, uchar>(bg_seed, i, j) > 0))
				energy = energy;
			else if (getPix<datatype, double>(fg_mask, i, j) > 0)
				energy += -getPix<datatype, double>(fg_prob, i, j);
			else
				energy += -getPix<datatype, double>(bg_prob, i, j);
		}
	}

	std::cout << "T-Link ENERGY: " << (energy - ncut_energy) << std::endl;

	return energy;
}