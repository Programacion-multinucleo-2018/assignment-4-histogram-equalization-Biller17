#include <iostream>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
using namespace std;


//Adrian Biller A01018940

// Histogram equalization function given an opencv mat and output
void image_histogram_equalizer(const cv::Mat& input, cv::Mat& output)
{
	int histogram[256] = {0};

	//creating histogram
  for (int pixel_tid = 0; pixel_tid < input.cols*input.rows; pixel_tid++) {
    histogram[ (int)input.data[pixel_tid] ] ++;
  }

  // Normalize histogram with cumulative distribution function and set range from 0 to 255
  unsigned long int normalized_histogram[256];
	unsigned long int accumulated = 0;


	for (int i=0; i<256; i++) {
		// #pragma omp atomic
		accumulated += histogram[i];
		normalized_histogram[i] = accumulated * 255 / (input.cols*input.rows);
	}

  //updating image with normalized histogram
	for (int pixel_tid = 0; pixel_tid < input.cols*input.rows; pixel_tid++) {

    output.data[pixel_tid] = normalized_histogram[ (int)input.data[pixel_tid] ];
  }
}





int main(int argc, char *argv[])
{
	string imagePath;

	// checking image path
	if(argc < 2)
		imagePath = "Images/dog2	.jpeg";
	else
		imagePath = argv[1];

	// read color image
	cv::Mat input = cv::imread(imagePath, CV_LOAD_IMAGE_COLOR);

	if (input.empty())
	{
		cout << "Image Not Found!" << std::endl;
		cin.get();
		return -1;
	}

//converting image to grayscale
	cv::Mat grayscale_input;
  cvtColor(input, grayscale_input, cv::COLOR_BGR2GRAY);

  //creating output image
  cv::Mat output(grayscale_input.rows, grayscale_input.cols, grayscale_input.type());

  //changing contrastof output image
	image_histogram_equalizer(grayscale_input, output);

	//Allow the windows to resize
	namedWindow("Input", cv::WINDOW_NORMAL);
	namedWindow("Output", cv::WINDOW_NORMAL);


  //showing initial image vs contrast change
	cv::imshow("Input", grayscale_input);
	cv::imshow("Output", output);

	//Wait for key press
	cv::waitKey();

	return 0;
}
