#include <opencv2/core.hpp>
#include <opencv2/ml.hpp>
#include "opencv2/highgui/highgui.hpp"
//#include "opencv2/imgproc/imgproc.hpp"
#include <iostream>
#include <string>
#include <fstream>

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

int main(int argc, char** argv)
{
	// Image name
	cv::String imageName = "../samples/noise_blur_00.png";
	cv::String fgName = "../samples/noise_blur_00_fg.bmp";
	cv::String bgName = "../samples/noise_blur_00_bg.bmp";

	// Open the images
	cv::Mat img = cv::imread(imageName.c_str(), cv::IMREAD_GRAYSCALE);
	if (check_file(img, imageName) < 0)
		return -1;

	cv::Mat fg = cv::imread(fgName.c_str());
	if (check_file(fg, fgName) < 0)
		return -1;

	cv::Mat bg = cv::imread(bgName.c_str());
	if (check_file(bg, bgName) < 0)
		return -1;


	// Turn fg/bg to binary
	cv::Mat fg_bin = fg >= 255;
	cv::Mat bg_bin = bg >= 255;


	// Display image
	cv::namedWindow("Display window", cv::WINDOW_AUTOSIZE); // Create a window for display.
	cv::imshow("Display window", bg_bin);


	cv::waitKey(0);
	return 0;
}