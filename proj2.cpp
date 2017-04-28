/*
Aniket Dhobe
1998239554
dhobe@usc.edu
February 26,2017
*/
#include <opencv2/ximgproc.hpp>
#include "opencv2/highgui.hpp"
#include "opencv2/imgproc.hpp"
#include "opencv2/core/utility.hpp"
#include <iostream>
#include <vector>
#include <stdio.h>
#include <stdlib.h>
#include <cmath>
#include <string>

using namespace std;
using namespace cv;
using namespace cv::ximgproc;

vector < vector < vector < unsigned char> > > ImagetoVector(const char * filename, int heigth, int width, int BytesPerPixel){
  FILE *file;
  if (!(file=fopen(filename,"rb"))) {
        cout << "Cannot open file: " << filename <<endl;
        exit(1);
  }
  unsigned char * imagedata = (unsigned char *)malloc(heigth*width*BytesPerPixel);
  fread(imagedata, sizeof(unsigned char), heigth*width*BytesPerPixel, file);
  fclose(file);
  vector < vector < vector < unsigned char> > > v1(heigth,vector < vector < unsigned char> >(width,vector < unsigned char>(BytesPerPixel,0)));
  int imageidx = 0;
  for(int i = 0; i < heigth; i++){
    for(int j = 0; j < width; j++){
      for(int k = 0; k < BytesPerPixel; k++){
          v1[i][j][k] = imagedata[imageidx];
          imageidx++;
      }
    }
  }
  free(imagedata);
  return v1;
}

Mat convertVectoMat(vector < vector < vector < unsigned char> > > &v1, int heigth, int width, int BytesPerPixel ){
    if(BytesPerPixel > 1){
        Mat outMat(heigth,width,CV_8UC3);
        Vec3b vec;
        for(int i = 0; i < heigth; i++){
            for(int j = 0; j < width; j++){
                vec.val[0] = v1[i][j][2];
                vec.val[1] = v1[i][j][1];
                vec.val[2] = v1[i][j][0];
                outMat.at<Vec3b>(i,j) = vec;
              }
        }
        return outMat;
    }else{
        Mat outMat(heigth,width,CV_8UC1);
        for(int i = 0; i < heigth; i++){
            for(int j = 0; j < width; j++){
                outMat.at<unsigned char>(i,j) = v1[i][j][0];
              }
        }
        return outMat;
    }
}

int structured_edge_detection( const char* infile, int heigth, int width, int BytesPerPixel, const char* outfile, const char* mFile, int thr)
{

//    Mat img = imread(infile, CV_LOAD_IMAGE_COLOR);
    vector < vector < vector < unsigned char> > > v1 = ImagetoVector(infile, heigth, width, BytesPerPixel);
    Mat img = convertVectoMat(v1, heigth, width, BytesPerPixel );
    cv::imwrite("tmp.jpg", img);
    img.convertTo(img, cv::DataType<float>::type, thr/255.0);
    Mat edges(img.size(), img.type());
    cv::Ptr<StructuredEdgeDetection> SED =  createStructuredEdgeDetection(mFile);
    SED->detectEdges(img, edges);
    cv::imwrite(outfile, 255*edges);
    return 0;
}

int canny( const char* infile, int heigth, int width, int BytesPerPixel, const char* outfile, int lowThreshold, int upthr)
{
    Mat src, src_gray;
    Mat dst, detected_edges;
    vector < vector < vector < unsigned char> > > v1 = ImagetoVector(infile, heigth, width, BytesPerPixel);
    src = convertVectoMat(v1, heigth, width, BytesPerPixel );
    dst.create( src.size(), src.type() );
    cvtColor( src, src_gray, CV_BGR2GRAY );
    blur( src_gray, detected_edges, Size(3,3) );
    Canny( detected_edges, detected_edges, lowThreshold, upthr, 3 );
    dst = Scalar::all(0);
    src.copyTo( dst, detected_edges);
    cv::imwrite(outfile, 255*detected_edges);
    return 0;
}

int main( int argc, char** argv ){
  string s1 = argv[1];
  string s2 = argv[2];
  structured_edge_detection(argv[1], 321, 481, 3, "Castle_SE1.jpg", "model.yml.gz", 1);
  structured_edge_detection(argv[1], 321, 481, 3, "Castle_SE2.jpg", "model.yml.gz", 2);
  structured_edge_detection(argv[2], 321, 481, 3, "Boat_SE1.jpg", "model.yml.gz", 1);
  structured_edge_detection(argv[2], 321, 481, 3, "Boat_SE2.jpg", "model.yml.gz", 2);
  canny(argv[1], 321, 481, 3, "Castle_C_10.jpg", 10, 200);
  canny(argv[1], 321, 481, 3, "Castle_C_50.jpg", 50, 150);
  canny(argv[1], 321, 481, 3, "Castle_C_100.jpg", 100, 120);
  canny(argv[2], 321, 481, 3, "Boat_C_10.jpg", 10, 200);
  canny(argv[2], 321, 481, 3, "Boat_C_50.jpg", 50, 150);
  canny(argv[2], 321, 481, 3, "Boat_C_100.jpg", 100, 120);
  return 1;  
}
