#include "matchTemplateTest.h"
#include <iostream> 


/// 载入原图像和模板块
Mat matchTemplateTest::img;   //静态成员必须声明
Mat matchTemplateTest::templ; 
Mat matchTemplateTest::result;
int matchTemplateTest::match_method;
/**
 * @函数 MatchingMethod
 * @简单的滑动条回调函数
 */
void  matchTemplateTest::MatchingMethod( int, void* )
{
	cout<<"match_method is"<<match_method<<endl;

	Mat img_display;
	img.copyTo( img_display );

	/// 创建输出结果的矩阵
	int result_cols =  img.cols - templ.cols + 1;
	int result_rows =  img.rows - templ.rows + 1;

	result.create( result_cols, result_rows, CV_32FC1 );

	/// 进行匹配和标准化
	matchTemplate( img, templ, result, 	match_method);
	normalize( result, result, 0, 1, NORM_MINMAX, -1, Mat() );

	/// 通过函数 minMaxLoc 定位最匹配的位置
	double minVal; double maxVal; Point minLoc; Point maxLoc;
	Point matchLoc;

	minMaxLoc( result, &minVal, &maxVal, &minLoc, &maxLoc, Mat() );

	/// 对于方法 SQDIFF 和 SQDIFF_NORMED, 越小的数值代表更高的匹配结果. 而对于其他方法, 数值越大匹配越好
	if( match_method  == CV_TM_SQDIFF || match_method == CV_TM_SQDIFF_NORMED )
	{ matchLoc = minLoc; }
	else
	{ matchLoc = maxLoc; }

	/// 让我看看您的最终结果
	rectangle( img_display, matchLoc, Point( matchLoc.x + templ.cols , matchLoc.y + templ.rows ), Scalar::all(0), 2, 8, 0 );
	rectangle( result, matchLoc, Point( matchLoc.x + templ.cols , matchLoc.y + templ.rows ), Scalar::all(0), 2, 8, 0 );

	imshow("Result window", img_display );
	imshow("TempResult window", result );
}

matchTemplateTest::matchTemplateTest(void)
{
	testName.append("matchTemplateTest");
}


matchTemplateTest::~matchTemplateTest(void)
{
}
void matchTemplateTest::doTest()
{
	cout<<"run test----->"<<testName<<endl;

	/// 载入原图像和模板块
	img =   imread(SPIC, 1);
	templ = imread(TEMPIC, 1);
	//cout<<"templ="<<templ.rows<<"X"<<templ.cols<<endl;
	/// 创建窗口
	namedWindow("Source Image", CV_WINDOW_AUTOSIZE);
	namedWindow("Template Image", CV_WINDOW_AUTOSIZE);

	namedWindow("Result window", CV_WINDOW_AUTOSIZE);
	namedWindow("TempResult window", CV_WINDOW_AUTOSIZE);

	/// 创建滑动条
	
	createTrackbar(TBAR, "Source Image", &match_method, 5, matchTemplateTest::MatchingMethod);
	imshow("Source Image", img);
	imshow("Template Image", templ);
	MatchingMethod(0, NULL);
	waitKey(0);//和窗口有关系
}

