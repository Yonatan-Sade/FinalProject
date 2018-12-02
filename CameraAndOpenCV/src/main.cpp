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

// DAC/ao0 is for updown. applying positive will lead to down.
// DAC/ao1 is for right-left. applying positive will lead to right.
// xDeltaFromCenter positive means particle is right to the center
// yDeltaFromCenter positive means particle is below to the center

VideoCapture videoCapture;
VideoWriter videoWriter;

Mat sharedFrame;
Mat sharedCircles;

mutex sharedFrameMutex;
mutex sharedCirclesMutex;

void* galvoTaskHandleX = 0;
void* galvoTaskHandleY = 0;

double galvoDataX[1] = { 0 };
double galvoDataY[1] = { 0 };

atomic<bool> applicationDone = false;
atomic<bool> frameIsReady = false;
atomic<bool> circlesIsReady = false;
atomic<bool> trackingIsEnabled = false;
atomic<bool> galvoIsEnabled = false;
atomic<bool> writeToFile = false;


atomic<int> ROIWidth = 100;
atomic<int> ROIHeight = 100;
atomic<int> ROIXPosition = 0;
atomic<int> ROIYPosition = 0;

atomic<int> rightLeftMove = 0;
atomic<int> upDownMovement = 0;

#define GAUSSIAN_SIZE               55

Mat ConvertToCVmat(ImagePtr pImage)
{
	int result = 0;
	ImagePtr convertedImage = pImage->Convert(PixelFormat_Mono8, NEAREST_NEIGHBOR);

	unsigned int XPadding = static_cast<unsigned int>(convertedImage->GetXPadding());
	unsigned int YPadding = static_cast<unsigned int>(convertedImage->GetYPadding());
	unsigned int rowsize = static_cast<unsigned int>(convertedImage->GetWidth());
	unsigned int colsize = static_cast<unsigned int>(convertedImage->GetHeight());

	//image data contains padding. When allocating Mat container size, you need to account for the X,Y image data padding. 
	Mat cvimg = cv::Mat(colsize + YPadding, rowsize + XPadding, CV_8UC1, convertedImage->GetData(), convertedImage->GetStride());
	return cvimg.clone();
}

void measureTime(void)
{
	static int frameNumber = 0;
	frameNumber++;
	if (frameNumber == 24)
	{
		static int64 e2 = 1;
		static int64 e1 = 0;
		e2 = getTickCount();
		double time = (e2 - e1) / getTickFrequency();
		e1 = getTickCount();
		printf("Proccessed %d frames in %f seconds\n", frameNumber, time);
		frameNumber = 0;
	}
}

void acquisition(CameraPtr pCam)
{
	Mat localFrame;

	/* Capture first frame: */
	pCam->TriggerSoftware.Execute();
	ImagePtr pResultImage = pCam->GetNextImage();

	// convert to openCV format
	localFrame = ConvertToCVmat(pResultImage);
	pResultImage->Release();

	sharedFrame = localFrame.clone();

	/* Indicate other threads that first frame is ready: */
	frameIsReady = true;

	while (1)
	{
		pCam->TriggerSoftware.Execute();
		ImagePtr pResultImage = pCam->GetNextImage();

		// convert to openCV format
		localFrame = ConvertToCVmat(pResultImage);
		pResultImage->Release();

		/* Lock the shared memory: */
		sharedFrameMutex.lock();
		sharedFrame = localFrame.clone();
		sharedFrameMutex.unlock();

		/* Exit thread if done: */
		if (applicationDone == true)
		{
			return;
		}
	}
}

void processing()
{
	// open video writer: 
	//videoWriter.open("outputVideo.avi", CV_FOURCC('M', 'J', 'P', 'G'), 24.0, Size(640, 480));

	ofstream myfile;
	myfile.open("trajectory.txt");
	//myfile << "x y\n";

	// configure the galvo handler:
	if (DAQmxCreateTask("", &galvoTaskHandleX) < 0)
		printf("Error in DAQ card\n");
	if (DAQmxCreateAOVoltageChan(galvoTaskHandleX, "DAQCARD/ao1", "", -10.0, 10.0, DAQmx_Val_Volts, "") < 0)
		printf("Error in DAQ card\n");
	if (DAQmxStartTask(galvoTaskHandleX) < 0)
		printf("Error in DAQ card\n");

	if (DAQmxCreateTask("", &galvoTaskHandleY) < 0)
		printf("Error in DAQ card\n");
	if (DAQmxCreateAOVoltageChan(galvoTaskHandleY, "DAQCARD/ao0", "", -10.0, 10.0, DAQmx_Val_Volts, "") < 0)
		printf("Error in DAQ card\n");
	if (DAQmxStartTask(galvoTaskHandleY) < 0)
		printf("Error in DAQ card\n");

	/* Wait until first frame is ready: */
	while (frameIsReady == false);

	Mat localFrame;

	// Get first frame for frame sizes: 
	sharedFrameMutex.lock();
	localFrame = sharedFrame.clone();
	sharedFrameMutex.unlock();
	// Get image size:
	int imageWidth = localFrame.cols;
	int imageHeight = localFrame.rows;

	int xAnchor = imageWidth / 2 - ROIWidth / 2;
	int yAnchor = imageHeight / 2 - ROIHeight / 2;

	int xDeltaFromCenter = 0;
	int yDeltaFromCenter = 0;

	while (1)
	{
		/* Lock the shared memory: */
		sharedFrameMutex.lock();
		localFrame = sharedFrame.clone();
		sharedFrameMutex.unlock();

		// Gaussian blur:
		GaussianBlur(localFrame, localFrame, Size(GAUSSIAN_SIZE, GAUSSIAN_SIZE), 0, 0);

		// Get range of interest: 
		Rect ROIRectangle = Rect(xAnchor, yAnchor, ROIWidth, ROIHeight);

		// find max value:
		double minValue, maxValue;
		Point minIndex, maxIndex;
		
		//minMaxLoc(localFrame(ROIRectangle), &minValue, &maxValue, &minIndex, &maxIndex);
		minMaxLoc(localFrame(ROIRectangle), &maxValue, &minValue, &maxIndex, &minIndex);

		//maxIndex.x += (xAnchor);
		//maxIndex.y += (yAnchor);

		minIndex.x += (xAnchor);
		minIndex.y += (yAnchor);

		xDeltaFromCenter = minIndex.x - (imageWidth / 2);
		yDeltaFromCenter = minIndex.y - (imageHeight / 2);

		// Update anchors if relevant: 
		if (trackingIsEnabled == true)
		{
			/*xAnchor = (maxIndex.x - ROIWidth / 2);
			yAnchor = (maxIndex.y - ROIHeight / 2);*/

			xAnchor = (minIndex.x - ROIWidth / 2);
			yAnchor = (minIndex.y - ROIHeight / 2);
		}

		xAnchor += rightLeftMove;
		rightLeftMove = 0;
		yAnchor += upDownMovement;
		upDownMovement = 0;

		if (xAnchor < 0)
		{
			xAnchor = 0;
		}

		if (yAnchor < 0)
		{
			yAnchor = 0;
		}

		int xDelta = imageWidth - (xAnchor + ROIWidth);
		if (xDelta < 0)
		{
			xAnchor += xDelta;
		}

		int yDelta = imageHeight - (yAnchor + ROIHeight);
		if (yDelta < 0)
		{
			yAnchor += yDelta;
		}

		ROIRectangle = Rect(xAnchor, yAnchor, ROIWidth, ROIHeight);

		// Apply galvo operation if relevant: 
		if (galvoIsEnabled == true)
		{
			static int num_of_frame = 0;
			const int limit = 50;

			num_of_frame++;
			if (num_of_frame == 2)
			{
				//if (xDeltaFromCenter > 10.0) //i.e.in the right, need to move left
				//{
				//	galvoDataX[0] = -0.06;
				//}
				//else if (xDeltaFromCenter < -10.0)
				//{
				//	galvoDataX[0] = 0.06;
				//}
				//else
				//{
				//	galvoDataX[0] = 0.0;
				//}


				//if (yDeltaFromCenter > 10.0) //i.e. below the center, need to moove up
				//{
				//	galvoDataY[0] = -0.06;
				//}
				//else if (yDeltaFromCenter < -10.0)
				//{
				//	galvoDataY[0] = 0.06;
				//}
				//else
				//{
				//	galvoDataY[0] = 0.0;
				//}

				if ((xDeltaFromCenter < limit) && ((xDeltaFromCenter > 0))) //i.e.in the right, need to move left
				{
					galvoDataX[0] = -0.01 * ( static_cast<double>(xDeltaFromCenter) / static_cast<double>(limit) );
				}
				else if ((xDeltaFromCenter > -limit) && (xDeltaFromCenter < 0))
				{
					galvoDataX[0] = -0.01 * (static_cast<double>(xDeltaFromCenter) / static_cast<double>(limit));
				}
				else
				{
					galvoDataX[0] = 0.0;
				}


				if ((yDeltaFromCenter < limit) && (yDeltaFromCenter > 0)) //i.e. below the center, need to moove up
				{
					galvoDataY[0] = -0.01 * (static_cast<double>(yDeltaFromCenter) / static_cast<double>(limit));
				}
				else if ((yDeltaFromCenter > -limit) && (yDeltaFromCenter < 0))
				{
					galvoDataY[0] = -0.01 * (static_cast<double>(yDeltaFromCenter) / static_cast<double>(limit));
				}
				else
				{
					galvoDataY[0] = 0.0;
				}

				//galvoDataX[0] = 0.06;
				//galvoDataY[0] = 0.06;
				printf("Sending command to galvo x: %f milivolts \n", galvoDataX[0]);
				if (DAQmxWriteAnalogF64(galvoTaskHandleX, 1, 1, 10.0, DAQmx_Val_GroupByChannel, galvoDataX, NULL, NULL) < 0)
					printf("Error in DAQ card\n");

				printf("Sending command to galvo y: %f milivolts \n", galvoDataY[0]);
				if (DAQmxWriteAnalogF64(galvoTaskHandleY, 1, 1, 10.0, DAQmx_Val_GroupByChannel, galvoDataY, NULL, NULL) < 0)
					printf("Error in DAQ card\n");

				num_of_frame = 0;
			}
		}
		else
		{
			galvoDataX[0] = 0;
			if (DAQmxWriteAnalogF64(galvoTaskHandleX, 1, 1, 10.0, DAQmx_Val_GroupByChannel, galvoDataX, NULL, NULL) < 0)
				printf("Error in DAQ card\n");

			galvoDataY[0] = 0;
			if (DAQmxWriteAnalogF64(galvoTaskHandleY, 1, 1, 10.0, DAQmx_Val_GroupByChannel, galvoDataY, NULL, NULL) < 0)
				printf("Error in DAQ card\n");
		}

		// convert to 3 channels:
		cvtColor(localFrame, localFrame, COLOR_GRAY2BGR);

		// Draw circles and ROI: 
		circle(localFrame, minIndex, 29, Scalar(0, 0, 255), 3);
		rectangle(localFrame, ROIRectangle, Scalar(0, 255, 0), 3);

		// Save video:
		//resize(localFrame, localFrame, Size(640, 480));
		//videoWriter << localFrame;

		// Save trjectory: 
		if (writeToFile == true)
		{
			myfile << minIndex.x << " " << minIndex.y << "\n";
		}

		// Measure time:
		measureTime();

		sharedCirclesMutex.lock();
		sharedCircles = localFrame.clone();
		sharedCirclesMutex.unlock();

		circlesIsReady = true;

		/* Exit thread if done: */
		if (applicationDone == true)
		{
			break;
		}
	}

	// Close galvo handle
	if (galvoTaskHandleX != 0)
	{
		DAQmxStopTask(galvoTaskHandleX);
		DAQmxClearTask(galvoTaskHandleX);
		printf("galvo x handle closed successfuly\n");
	}

	if (galvoTaskHandleY != 0)
	{
		DAQmxStopTask(galvoTaskHandleY);
		DAQmxClearTask(galvoTaskHandleY);
		printf("galvo y handle closed successfuly\n");
	}

	myfile.close();

}

void userInterface()
{
	/* Wait until first frame is ready: */
	while (frameIsReady == false);
	while (circlesIsReady == false);

	namedWindow("sharedCircles");
	//createTrackbar("ROI Width", "sharedCircles", (int *)&ROIWidth, 1000);
	//("ROI Height", "sharedCircles", (int *)&ROIHeight, 1000);
	//createTrackbar("ROI x position", "sharedCircles", (int *)&ROIXPosition, 1000);
	//createTrackbar("ROI y position", "sharedCircles", (int *)&ROIYPosition, 1000);

	Mat localFrame;

	while (1)
	{
		/* Try to get the mutex: */
		if (sharedCirclesMutex.try_lock() == true)
		{
			localFrame = sharedCircles.clone();
			sharedCirclesMutex.unlock();

			// Resize:
			resize(localFrame, localFrame, Size(640, 480));

			imshow("sharedCircles", localFrame);
		}

		char keyPressed = waitKey(25);
		if (keyPressed >= 0)
		{
			switch (keyPressed)
			{
			case 'q':
				applicationDone = true;
				return;
				break;

			case 't':
				if (trackingIsEnabled == true)
				{
					trackingIsEnabled = false;
				}
				else
				{
					trackingIsEnabled = true;
				}
				break;

			case 'g':
				if (galvoIsEnabled == true)
				{
					galvoIsEnabled = false;
				}
				else
				{
					galvoIsEnabled = true;
				}
				break;

			case 'l':
				rightLeftMove = 10;
				break;

			case 'k':
				rightLeftMove = -10;
				break;

			case 'y':
				upDownMovement = 10;
				break;

			case 'h':
				upDownMovement = -10;
				break;

			case 'a':
				if (writeToFile == true)
				{
					writeToFile = false;
					std::cout << "Finishing to write into file" << std::endl;
				}
				else
				{
					writeToFile = true;
					std::cout << "starting to write into file" << std::endl;
				}
				break;

			default:
				break;
			}
		}
	}
}

void run(CameraPtr pCam)
{
	// Print help:
	printf("Keyboard interface: \n");
	printf("	q for quit \n");
	printf("	t for toggling tracking-enable state \n");
	printf("	l for moving the rectangle right \n");
	printf("	k for moving the rectangle left \n");
	printf("	y for moving the rectangle up \n");
	printf("	h for moving the rectangle down \n");
	printf("	a for write to file \n");

	// Initialize camera
	pCam->Init();

	// Set trigger off, software, and then on again: 
	pCam->TriggerMode.SetValue(TriggerModeEnums::TriggerMode_Off);
	pCam->TriggerSource.SetValue(TriggerSourceEnums::TriggerSource_Software);
	pCam->TriggerMode.SetValue(TriggerModeEnums::TriggerMode_On);

	//pCam->Width.SetValue(640);
	//pCam->Height.SetValue(480);

	// Set aqcuisition mode to continuous and start: 
	pCam->AcquisitionMode.SetValue(AcquisitionModeEnums::AcquisitionMode_Continuous);
	pCam->BeginAcquisition();

	thread acquisitionThread(&acquisition, pCam);
	thread processingThread(&processing);
	thread userInterfaceThread(&userInterface);

	acquisitionThread.join();
	processingThread.join();
	userInterfaceThread.join();

	// Deinitialize camera
	pCam->EndAcquisition();
	pCam->DeInit();
}


int main(int, char**)
{
	// Retrieve singleton reference to system object
	SystemPtr system = System::GetInstance();

	// Retrieve list of cameras from the system
	CameraList camList = system->GetCameras();

	// Retrieve pointer for the camera: 
	run(camList.GetByIndex(0));

	// Clear camera list before releasing system
	camList.Clear();
	// Release system
	system->ReleaseInstance();
}