#include "opencv2/core.hpp"
#include "opencv2/features2d.hpp"
#include "opencv2/xfeatures2d.hpp"
#include "opencv2/highgui.hpp"
#include "opencv2/imgcodecs.hpp"
#include <iostream>
#include <vector>
#include <stdio.h>
#include <stdlib.h>
#include <cmath>
#include <string>

using namespace std;
using namespace cv;
using namespace cv::xfeatures2d;

/*http://docs.opencv.org/3.0-beta/doc/tutorials/features2d/feature_detection/feature_detection.html*/

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

void Ext_of_Salient_Point( const char * filename, int heigth, int width, int BytesPerPixel, const char * Ofilename1,const char * Ofilename2){

  vector < vector < vector < unsigned char> > > v1 = ImagetoVector(filename, heigth, width, BytesPerPixel);
  Mat img_1 = convertVectoMat( v1,  heigth, width, BytesPerPixel);

  Ptr<SURF> detector_SURF = SURF::create( 400 );
  Ptr<SIFT> detector_SIFT = SIFT::create( 400 );

  std::vector<KeyPoint> keypoints_1, keypoints_2;

  detector_SURF->detect( img_1, keypoints_1 );
  detector_SIFT->detect( img_1, keypoints_2 );

  Mat img_keypoints_1; 
  Mat img_keypoints_2; 
  
  drawKeypoints( img_1, keypoints_1, img_keypoints_1, Scalar::all(-1), DrawMatchesFlags::DEFAULT );
  drawKeypoints( img_1, keypoints_2, img_keypoints_2, Scalar::all(-1), DrawMatchesFlags::DEFAULT );

  imwrite(Ofilename1, img_keypoints_1 );
  imwrite(Ofilename2, img_keypoints_2 );
  return;
  }

void feature_match_SIFT(const char * filename1, const char * filename2, int heigth, int width, int BytesPerPixel, const char * Ofilename){

    vector < vector < vector < unsigned char> > > v1 = ImagetoVector(filename1, heigth, width, BytesPerPixel);
    Mat img_1 = convertVectoMat( v1,  heigth, width, BytesPerPixel);
    v1 = ImagetoVector(filename2, heigth, width, BytesPerPixel);
    Mat img_2 = convertVectoMat( v1,  heigth, width, BytesPerPixel);

    Ptr<SIFT> detector_SIFT = SIFT::create( 400 );
    std::vector<KeyPoint> KP1;
    
    Mat DESC1;
    detector_SIFT->detectAndCompute( img_1, Mat(), KP1, DESC1 );
    std::vector<KeyPoint> KP2;
    Mat DESC2;
    detector_SIFT->detectAndCompute( img_2, Mat(), KP2, DESC2 );
  
    FlannBasedMatcher matcher;
    std::vector< DMatch > matches, good_matches;
    matcher.match( DESC1, DESC2, matches );
    double max_dist = 0; double min_dist = 100;

    for( int i = 0; i < DESC1.rows; i++ )
    {
        double dist = matches[i].distance;
        min_dist = min(dist, min_dist);
        max_dist = max(dist, max_dist);
    }
  
    for( int i = 0; i < DESC1.rows; i++ )
    {
        if( matches[i].distance <= max(2*min_dist, 0.02) )
            { good_matches.push_back( matches[i]); }
    }
  
  Mat img_matches;
  drawMatches( img_1, KP1, img_2, KP2,
               good_matches, img_matches, Scalar::all(-1), Scalar::all(-1),
               vector<char>(), DrawMatchesFlags::NOT_DRAW_SINGLE_POINTS );
  imwrite( Ofilename, img_matches );
  
}

void feature_match_SURF(const char * filename1, const char * filename2, int heigth, int width, int BytesPerPixel, const char * Ofilename){

    vector < vector < vector < unsigned char> > > v1 = ImagetoVector(filename1, heigth, width, BytesPerPixel);
    Mat img_1 = convertVectoMat( v1,  heigth, width, BytesPerPixel);
    v1 = ImagetoVector(filename2, heigth, width, BytesPerPixel);
    Mat img_2 = convertVectoMat( v1,  heigth, width, BytesPerPixel);

    Ptr<SURF> detector_SURF = SURF::create( 400 );
    std::vector<KeyPoint> KP1;
    
    Mat DESC1;
    detector_SURF->detectAndCompute( img_1, Mat(), KP1, DESC1 );
    std::vector<KeyPoint> KP2;
    Mat DESC2;
    detector_SURF->detectAndCompute( img_2, Mat(), KP2, DESC2 );
  
    FlannBasedMatcher matcher;
    std::vector< DMatch > matches, good_matches;
    matcher.match( DESC1, DESC2, matches );
    double max_dist = 0; double min_dist = 100;

    for( int i = 0; i < DESC1.rows; i++ )
    {
        double dist = matches[i].distance;
        min_dist = min(dist, min_dist);
        max_dist = max(dist, max_dist);
    }
  
    for( int i = 0; i < DESC1.rows; i++ )
    {
        if( matches[i].distance <= max(2*min_dist, 0.02) )
            { good_matches.push_back( matches[i]); }
    }
  
  Mat img_matches;
  drawMatches( img_1, KP1, img_2, KP2,
               good_matches, img_matches, Scalar::all(-1), Scalar::all(-1),
               vector<char>(), DrawMatchesFlags::NOT_DRAW_SINGLE_POINTS );
  imwrite( Ofilename, img_matches );
  
}

int main(int argc, char ** argv){
    Ext_of_Salient_Point(argv[1], 300, 500, 3, "SURF_suv.jpg", "SIFT_suv.jpg");
    Ext_of_Salient_Point(argv[2], 300, 500, 3, "SURF_truck.jpg","SIFT_truck.jpg");
    feature_match_SIFT(argv[1], argv[3], 300, 500, 3, "Conv1_suv_SIFT.jpg");
    feature_match_SIFT(argv[2], argv[4], 300, 500, 3, "Conv2_suv_SIFT.jpg");
    feature_match_SURF(argv[1], argv[3], 300, 500, 3, "Conv1_truck_SURF.jpg");
    feature_match_SURF(argv[2], argv[4], 300, 500, 3, "Conv2_truck_SURF.jpg");
    return 1;
}
