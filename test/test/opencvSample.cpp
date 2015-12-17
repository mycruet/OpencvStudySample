#include <stdio.h>
#include <opencv2/opencv.hpp> 
#include <opencv2/core/core.hpp>
#include <opencv2/core/mat.hpp>
#include <iostream>  
#include <string>  
#include "matchTemplateTest.h"
#include "CameraCalibrator.h"


using namespace cv;  
using namespace std;  

int main()  
{  
#if 0
  printf("hell\n");
  FILE *f = NULL;
  f =  fopen("D:\liu.jpg", "r");
  if(f == NULL)
  {
	    printf("hell========\n");
  }
  else
  {
	 fclose(f);
  }
  getchar();
#endif
#if 0
  Mat img = imread("chessboard01.jpg", 0);
  //Mat img = imread("CalibIm1.png", 0);
  //Mat img = imread("chess1.jpg", 1);

  if(img.empty())
  {
	   cout<<"error";  
	   return -1;

  }
  printf("%d\n",img.channels());
  cout << "rows="<<img.rows<<endl; 
  cout << "cols="<<img.cols<<endl;
  imshow("mypic", img);
#endif

#if 0
  Mat grayimg(300,200,CV_8UC1);

  imshow("mypic", grayimg); 
  
  
  waitKey(0);


 // Mat img(300,200,CV_8UC3,Scalar(0,0,255));
  for(int i = 0; i < grayimg.rows; i++)
  {

	   uchar *p = grayimg.ptr<uchar>(i);
	   for(int j = 0; j < grayimg.cols; j++)
	   {
		   p[j] = 255;

	   }

  }

  imshow("mypic", grayimg);
  waitKey(0);

#endif
#if 0
   


  testBase *t = new matchTemplateTest();

  t->doTest();


  delete t;
  //getchar();//not use waitKey of opencv due to its need a window to active
#endif
//partial case
#if 0
  const int divideWith = 100;//将颜色元素数值按照10为单位分级,即0-9为0， 10-19为10， 20-29为20....
  uchar table[256];//将0-255的元素值变换
  for (int i = 0; i < 256; ++i)
	  table[i] = divideWith* (i/divideWith);

  
  double t = (double)getTickCount();
  //Mat M =   imread( "s2.jpg", 1);
  Mat M =   imread( "s2.jpg", 0);
  //Mat M(4,2, CV_8UC3, Scalar(0,0,255));   
  //cout << "M = " << endl << " " << M << endl;   
  cout << "channels="<<M.channels()<<endl;
  cout << "rows="<<M.rows<<endl; 
  cout << "cols="<<M.cols<<endl;
  cout << "isContinuous="<<M.isContinuous()<<endl;
  cout << "depth="<<M.depth()<<endl;
  imshow("mypic", M);//时间不稳定
  t = ((double)getTickCount() - t)/getTickFrequency();
  cout << "Times passed in seconds: " << t << endl;

  //change 
  t = (double)getTickCount();
#if 0
  Mat_<float> J(M);//error when M is not gray
  
  cout << "depth="<<J.depth()<<endl; 
  cout << "channels="<<J.channels()<<endl;
#endif
#if 0
  uchar *p;
  int i,j;
  for(i = 0; i < M.rows; i++)
  {
	  p = M.ptr<uchar>(i);
	  for ( j = 0; j < M.channels()*M.cols; ++j)
	  {
		  p[j] = table[p[j]];             
	  }
  }

#endif
#if 0
  int nRows = M.rows * M.channels(); 
  int nCols = M.cols;

  if (M.isContinuous())
  {
	  nCols *= nRows;
	  nRows = 1;         
  }

  int i,j;
  uchar* p; 
  for( i = 0; i < nRows; ++i)
  {
	  p = M.ptr<uchar>(i);
	  for ( j = 0; j < nCols; ++j)
	  {
		  p[j] = table[p[j]];             
	  }
  }
   
#endif
#if 0
  uchar* p = M.data;

  for( unsigned int i =0; i < M.cols*M.rows*M.channels(); ++i)
	  *p++ = table[*p];
#endif 

#if 0
  CV_Assert(M.depth() != sizeof(uchar));     
  if(M.channels() == 3)
  {
	  MatIterator_<Vec3b> it, end;
	  for(it = M.begin<Vec3b>(),end = M.end<Vec3b>(); it!=end; it++)
	  {
		  (*it)[0] = table[(*it)[0]];
		  (*it)[1] = table[(*it)[1]];
		  (*it)[2] = table[(*it)[2]];

	  }
  }
#endif 
#if 0
  CV_Assert(M.depth() == CV_8U);     
  Mat J;
  Mat lookUpTable(1, 256, CV_8U);
  uchar* p = lookUpTable.data; 
  for( int i = 0; i < 256; ++i)
	  p[i] = table[i];

  t = (double)getTickCount();    

 
  LUT(M, lookUpTable, J);
  M = J;

#endif
#if 0
  Mat J;
  Mat kern = (Mat_<char>(3,3) <<  0, -1,  0,
	                             -1,  5, -1,
	                              0, -1,  0);
  filter2D(M, J, M.depth(), kern );
  M=J;
#endif
#if 0
  int a = 2;//对比度
  int b = 0;//亮度
  Mat J = Mat::zeros(M.rows, M.cols, M.type());
  for(int i=0 ; i< M.rows; i++)
  {
	  for(int j=0; j<M.cols; j++)
	  {//Mat的at函数是个模板，他的数据类型决定了像素的存储格式，参数是像素的坐标，对齐到每个像素的起始地址
		  for(int c = 0; c<3; c++)
		  {
			   J.at<Vec3b>(i, j)[c] = saturate_cast<uchar>(a*M.at<Vec3b>(i, j)[c] + b);
		  }
		 
	  }
  }
  M = J;
#endif
#if 0//DFT
  Mat padded;                            //将输入图像延扩到最佳的尺寸(2,3,5的倍数速度快）
  int m = getOptimalDFTSize( M.rows );
  int n = getOptimalDFTSize( M.cols ); // 在边缘添加0
  cout<<"m="<<m<<"n="<<n<<endl;
  copyMakeBorder(M, padded, 0, m - M.rows, 0, n - M.cols, BORDER_CONSTANT, Scalar::all(0));
 // M=padded;
  //cout << "rows="<<M.rows<<endl; 
  //cout << "cols="<<M.cols<<endl;
 // cout << "channels="<<complexI.channels()<<endl;
  Mat planes[] = {Mat_<float>(padded),Mat::zeros(padded.rows,padded.cols,CV_32F)};
  Mat complexI;
  merge(planes, 2, complexI);         // 为延扩后的图像增添一个初始化为0的通道
  cout << "channels="<<complexI.channels()<<endl;
  dft(complexI, complexI);            // 变换结果很好的保存在原始矩阵中
  split(complexI,planes);// planes[0] = 实数, planes[1] = 复数

  //imshow("changepic",  planes[1]);//error when channels are not 1,3,4
  //M=planes[1];
 // cout<<"M="<<endl<<M<<endl;
  magnitude(planes[0], planes[1], planes[0]);// planes[0] = magnitude
  Mat magI = planes[0];
  magI += Scalar::all(1);                    // 转换到对数尺度
  log(magI, magI);
  //M=magI;

  magI = magI(Rect(0, 0, magI.cols & -2, magI.rows & -2));
  int cx = magI.cols/2;
  int cy = magI.rows/2;

  Mat q0(magI, Rect(0, 0, cx, cy));   // Top-Left - 为每一个象限创建ROI
  Mat q1(magI, Rect(cx, 0, cx, cy));  // Top-Right
  Mat q2(magI, Rect(0, cy, cx, cy));  // Bottom-Left
  Mat q3(magI, Rect(cx, cy, cx, cy)); // Bottom-Right

  Mat tmp;                           // 交换象限 (Top-Left with Bottom-Right)
  q0.copyTo(tmp);
  q3.copyTo(q0);
  tmp.copyTo(q3);

  q1.copyTo(tmp);                    // 交换象限 (Top-Right with Bottom-Left)
  q2.copyTo(q1);
  tmp.copyTo(q2);


  normalize(magI, magI, 0, 1, CV_MINMAX); // 将float类型的矩阵转换到可显示图像范围
  // (float [0， 1]).

  M=magI;
#endif

  imshow("changepic", M);
  t = ((double)getTickCount() - t)/getTickFrequency();
  cout << "Times passed in seconds: " << t << endl;
  waitKey(0);
  getchar();
#endif
#if 0
  // output vectors of image points 
  std::vector<cv::Point2f> imageCorners;  
  //std::vector<cv::Point2i> imageCorners; 

  // number of corners on the chessboard 
 // cv::Size boardSize(8,6); 
    //cv::Size boardSize(4,2); 
  cv::Size boardSize(6,4); 
  //cv::Size boardSize(8,6); 

  // Get the chessboard corners 
  bool found = cv::findChessboardCorners(img,  
	  boardSize, imageCorners); 
  if(found == true)
  {
	  //Draw the corners 
	 cout << "findChessboardCorners:OK" << "===="<<found<< "===="<<imageCorners.size()<<endl;

#if 1
	 for(int i = 0; i < imageCorners.size(); i++)
	 {
		 //cv::Point2f *t;
		 //t = (cv::Point2f *)imageCorners.pop_back()
		 cv::Point2f t = imageCorners[i];  
		 //cv::Point2i t = imageCorners[i];

		 cout<<"("<<t.x<<","<<t.y<<")"<<"    ";
		 if((i+1)%6 == 0)
			 cout<<endl;

	 }
#endif
	 
	        
          #if 1
		  cv::cornerSubPix(img, imageCorners,  
			  cv::Size(5,5),  
			  cv::Size(-1,-1),  
			  cv::TermCriteria(cv::TermCriteria::MAX_ITER + 
			  cv::TermCriteria::EPS,  
			  30,      // max number of iterations  
			  0.1));  // min accuracy 
         #endif

		  cv::drawChessboardCorners(img,  
			  boardSize, imageCorners,  
			  found); // corners have been found 
		  imshow("mypicc", img);


          #if 0
		  for(int i = 0; i < imageCorners.size(); i++)
		  {
			  //cv::Point2f *t;
			  //t = (cv::Point2f *)imageCorners.pop_back()
			  cv::Point2f t = imageCorners[i];
			  cout<<"("<<t.x<<","<<t.y<<")"<<"    ";
			  if((i+1)%6 == 0)
				  cout<<endl;

		  }
           #endif

  }
  else
	   cout << "findChessboardCorners:error" << found<< endl;


  

#endif
#if 0
   int bw = 9;
   int bh = 6;
   int countCorners = 0;
   int foundt = 0;
   cv::Size boardSize(bw, bh); 

   CvPoint2D32f *corners = new CvPoint2D32f[bw*bh];
   IplImage* img = NULL;
   img =  cvLoadImage("my2.jpg", 1);
   //img =  cvLoadImage("chessboard01.jpg", 1);  
   //img =  cvLoadImage("CalibIm1.jpg", 1);

   if(img == NULL)
   {
	   cout<<"cvload error"<<endl;
   }
   cvShowImage("mypicd", img);
   IplImage *colimg = NULL;
   foundt =  cvFindChessboardCorners(img, boardSize, corners, &countCorners,3);

   if(foundt > 0)
   {
	   //Draw the corners 
	   cout << "cvfindChessboardCorners:OK" <<"====="<< foundt<<"===="<<countCorners<<endl;
       #if 1
	   //cvCvtColor(colimg, img,CV_BGR2GRAY);
	   cvDrawChessboardCorners(img,  
		   boardSize, corners,  countCorners,
		   foundt); // corners have been found 
	   cvShowImage("mypicd", img);
       #endif

   }
   else
	   cout << "cvfindChessboardCorners:error" << foundt<< endl;

#endif
    // char buffer[10] ={0};
     //cout<<itoa(13,buffer,10)<<endl;
	 //cin>>buffer;
#if 1

   CameraCalibrator *c = new CameraCalibrator();
 
   cv::Size boardSize(9, 6); 
   //vector<int> v ={0,1};//C++11
  // vector<int> v; 
   //vector<int> v;
   //v.push_back(10);
   //for(auto &r : v)
	  // r+=2;
   //set pictures
  // String s[] = {"chessboard01.jpg","chessboard02.jpg","chessboard03.jpg","chessboard04.jpg"};
   vector<string> s;
   //vector<string> s ={"chessboard01.jpg","chessboard02.jpg","chessboard03.jpg","chessboard04.jpg"};
   s.push_back("test0.jpg");
//   s.push_back("my2.jpg"); 
//   s.push_back("my3.jpg");
//   s.push_back("my4.jpg");
//   s.push_back("my5.jpg");
//   s.push_back("my6.jpg"); 
//   s.push_back("my7.jpg");
//   s.push_back("my8.jpg");
//   s.push_back("my9.jpg");
//   s.push_back("my10.jpg"); 
 //  s.push_back("chessboard01.jpg");
 //  s.push_back("chessboard02.jpg"); 
 //  s.push_back("chessboard03.jpg");
//   s.push_back("chessboard04.jpg");

  
   if(c->addChessboardPoints(s,boardSize) != 1)
	   cout<<"error"<<endl;
   c->calibrate(boardSize);
   Mat iamge = imread(s[0],0);
//   c->remap(iamge);
//   c->addChessboardPoints2(s,boardSize);

   delete c;

   
#endif


  waitKey(0);
  return 0;  
}  
