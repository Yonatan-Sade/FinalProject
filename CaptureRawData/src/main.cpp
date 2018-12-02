#include <atomic>
#include <thread>
#include <mutex>

#include "Spinnaker.h"
#include "opencv2/opencv.hpp"

#include <stdio.h>

using namespace Spinnaker;
using namespace cv;
using namespace std;

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
atomic<bool> captureIsEnabled = false;

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

	std::cout << "Width is " << imageWidth << " Height is " << imageHeight << std::endl;

	// open video writer: 
	videoWriter.open("outputVideo.avi", CV_FOURCC('M', 'J', 'P', 'G'), 24.0, Size(imageWidth, imageHeight));

	while (1)
	{
		/* Lock the shared memory: */
		sharedFrameMutex.lock();
		localFrame = sharedFrame.clone();
		sharedFrameMutex.unlock();

		// convert to 3 channels:
		cvtColor(localFrame, localFrame, COLOR_GRAY2BGR);

		if (captureIsEnabled == true)
		{
			videoWriter << localFrame;
		}

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
}

void userInterface()
{
	/* Wait until first frame is ready: */
	while (frameIsReady == false);
	while (circlesIsReady == false);

	namedWindow("sharedCircles", 1);

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

			case 'v':
				if (captureIsEnabled == true)
				{
					captureIsEnabled = false;
					std::cout << "stopping to capture" << std::endl;
				}
				else
				{
					std::cout << "starting to capture" << std::endl;
					captureIsEnabled = true;
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
	printf("	v for toggling capture-enable state \n");

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