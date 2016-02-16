#pragma once
//opencv
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include <opencv2/opencv.hpp>  
#include<opencv/cv.h>

//std


class CameraCalibrator
{
public:
	CameraCalibrator(void);
	CameraCalibrator(cv::Size);
	virtual ~CameraCalibrator(void);
	// input points: 
	// the points in world coordinates 
	std::vector<std::vector<cv::Point3f>> objectPoints; 

	// the point positions in pixels 
	std::vector<std::vector<cv::Point2f>> imagePoints; 

	// output Matrices 
	cv::Mat A;//cameraMatrix; (f,u0,v0,dx,dy)
	cv::Mat D;//distCoeffs; (k1-k3,p1-p2)
	std::vector<cv::Mat> rvecs, tvecs; 
	// flag to specify how calibration is done 
	int flag; 

	// used in image undistortion  
	cv::Mat map1,map2;  
	bool mustInitUndistort; 
	cv::Size  boardSize_;

public:
	// Open chessboard images and extract corner points 
	int addChessboardPoints(const std::vector<std::string>& filelist,  cv::Size & boardSize);
	// Add scene points and corresponding image points 
	void addPoints(const std::vector<cv::Point2f>&imageCorners, const std::vector<cv::Point3f>& objectCorners);
	// Calibrate the camera 
	// returns the re-projection error 
	void calibrate(cv::Size &imageSize);
	// remove distortion in an image (after calibration) 
	cv::Mat remap(const cv::Mat &image);
	void reprojectFromImageToObject();
	void reprojectFromImageToObject(cv::Mat Image);
	cv::Point3f  getDistanceFromObjToImag(int picNum, int cornerNum);
	void checkCircle(cv::Mat Image);

};

cv::Mat mergeByCols(cv::Mat &a, cv::Mat &b);
cv::Mat reprojectFromImageToObject2TO3(cv::Point2f imagP, cv::Mat cameraMatrix, cv::Mat disMatrix,
									   cv::Mat RodMatrix, cv::Mat TMatrix, cv::Mat *RandTMaxtix=NULL, int flags=0);
