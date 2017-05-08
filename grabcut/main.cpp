#include <opencv2/core.hpp>
#include <opencv2/ml.hpp>
#include "opencv2/highgui/highgui.hpp"
#include <iostream>
#include <string>
#include <fstream>

int main(int argc, char** argv)
{
	// Image name
	cv::String imageName = "../samples/noise_blur_00.png";
	if (argc > 1)
		imageName = argv[1];

	// Open the image
	cv::Mat img = cv::imread(imageName.c_str(), cv::IMREAD_GRAYSCALE);
	if (img.empty())
	{
		std::cout << "Couldn't find/open image" << std::endl;
		system("pause");
		return -1;
	}

	cv::namedWindow("Display window", cv::WINDOW_AUTOSIZE); // Create a window for display.
	cv::imshow("Display window", img);


	cv::waitKey(0);
	return 0;
}