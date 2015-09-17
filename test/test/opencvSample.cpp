#include <stdio.h>
#include <opencv2\opencv.hpp>  
#include <iostream>  
#include <string>  
#include "matchTemplateTest.h"
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
  Mat img = imread("D:\\li.jpg", 0);

  if(img.empty())
  {
	   cout<<"error";  
	   return -1;

  }
  printf("%d\n",img.channels());
  cout<<"oooo";
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
#if 1
   


  testBase *t = new matchTemplateTest();

  t->doTest();


  delete t;
  //getchar();//not use waitKey of opencv due to its need a window to active
#endif
  return 0;  
}  
