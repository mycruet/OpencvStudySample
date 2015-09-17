#include "matchTemplateTest.h"
#include <iostream> 


/// ����ԭͼ���ģ���
Mat matchTemplateTest::img;   //��̬��Ա��������
Mat matchTemplateTest::templ; 
Mat matchTemplateTest::result;
int matchTemplateTest::match_method;
/**
 * @���� MatchingMethod
 * @�򵥵Ļ������ص�����
 */
void  matchTemplateTest::MatchingMethod( int, void* )
{
	cout<<"match_method is"<<match_method<<endl;

	Mat img_display;
	img.copyTo( img_display );

	/// �����������ľ���
	int result_cols =  img.cols - templ.cols + 1;
	int result_rows =  img.rows - templ.rows + 1;

	result.create( result_cols, result_rows, CV_32FC1 );

	/// ����ƥ��ͱ�׼��
	matchTemplate( img, templ, result, 	match_method);
	normalize( result, result, 0, 1, NORM_MINMAX, -1, Mat() );

	/// ͨ������ minMaxLoc ��λ��ƥ���λ��
	double minVal; double maxVal; Point minLoc; Point maxLoc;
	Point matchLoc;

	minMaxLoc( result, &minVal, &maxVal, &minLoc, &maxLoc, Mat() );

	/// ���ڷ��� SQDIFF �� SQDIFF_NORMED, ԽС����ֵ������ߵ�ƥ����. ��������������, ��ֵԽ��ƥ��Խ��
	if( match_method  == CV_TM_SQDIFF || match_method == CV_TM_SQDIFF_NORMED )
	{ matchLoc = minLoc; }
	else
	{ matchLoc = maxLoc; }

	/// ���ҿ����������ս��
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

	/// ����ԭͼ���ģ���
	img =   imread(SPIC, 1);
	templ = imread(TEMPIC, 1);
	//cout<<"templ="<<templ.rows<<"X"<<templ.cols<<endl;
	/// ��������
	namedWindow("Source Image", CV_WINDOW_AUTOSIZE);
	namedWindow("Template Image", CV_WINDOW_AUTOSIZE);

	namedWindow("Result window", CV_WINDOW_AUTOSIZE);
	namedWindow("TempResult window", CV_WINDOW_AUTOSIZE);

	/// ����������
	
	createTrackbar(TBAR, "Source Image", &match_method, 5, matchTemplateTest::MatchingMethod);
	imshow("Source Image", img);
	imshow("Template Image", templ);
	MatchingMethod(0, NULL);
	waitKey(0);//�ʹ����й�ϵ
}

