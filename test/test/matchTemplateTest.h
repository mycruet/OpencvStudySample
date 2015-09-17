#pragma once
#include "testbase.h"

#include <string>

#include "opencv2/highgui/highgui.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include <opencv2\opencv.hpp>  

#define SPIC "source.jpg"
#define TEMPIC "template.png"
#define TBAR   "Method: \n 0: SQDIFF \n 1: SQDIFF NORMED \n 2: TM CCORR \n 3: TM CCORR NORMED \n 4: TM COEFF \n 5: TM COEFF NORMED"

using namespace std; 
using namespace cv; 

class matchTemplateTest :
	public testBase
{
public:
	matchTemplateTest(void);
	~matchTemplateTest(void);
	virtual void doTest();
	static void MatchingMethod( int i, void*);

private:
	string testName;
    static Mat img;
	static Mat templ; 
	static Mat result;	
	static int match_method;

};

//void MatchingMethod( int, void* );