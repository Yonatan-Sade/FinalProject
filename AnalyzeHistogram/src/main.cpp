#include <atomic>
#include <thread>
#include <mutex>

#include "Spinnaker.h"
#include "opencv2/opencv.hpp"

#include <stdio.h>
#include <NIDAQmx.h>

using namespace Spinnaker;
using namespace cv;
using namespace std;

#define GAUSSIAN_SIZE               55

int main(int, char**)
{
	VideoCapture videoCapture("C:\\Users\\Yonatan sade\\Desktop\\videos\\0.3 mA.avi");

	if (!videoCapture.isOpened())  // check if we succeeded
		return -1;

	const int frames = videoCapture.get(CV_CAP_PROP_FRAME_COUNT);
	std::cout << "Total number of frames: " << frames << std::endl;

	ofstream myfile;
	myfile.open("trajectory.txt");

	Mat frame;
	videoCapture >> frame;

	namedWindow("frame", WINDOW_NORMAL);

	Rect roi = selectROI("frame", frame);

	for (int i = 1; i < frames; i++)
	{
		videoCapture >> frame;
		cvtColor(frame, frame, COLOR_BGR2GRAY);

		// Gaussian blur:
		GaussianBlur(frame, frame, Size(GAUSSIAN_SIZE, GAUSSIAN_SIZE), 0, 0);
		
		double minValue, maxValue;
		Point minIndex, maxIndex;
		minMaxLoc(frame(roi), &minValue, &maxValue, &minIndex, &maxIndex);

		maxIndex.x += roi.x;
		maxIndex.y += roi.y;

		cvtColor(frame, frame, COLOR_GRAY2BGR);

		rectangle(frame, roi, Scalar(0, 255, 0), 3);
		circle(frame, maxIndex, 29, Scalar(0, 0, 255), 3);

		myfile << maxIndex.x << " " << maxIndex.y << "\n";

		imshow("frame", frame);
		if (waitKey(1) >= 0)
			break;

		std::cout << i << " / " << frames << std::endl;
	}

	myfile.close();

	return 0;
}