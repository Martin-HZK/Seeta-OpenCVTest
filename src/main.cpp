#include <iostream>
 #include "testdb.cpp"
#include "seeta/FaceDatabase.h"
#include "seeta/FaceDetector.h"
#include "seeta/FaceLandmarker.h"
#include "seeta/FaceDetector.h"
#include "seeta/Struct.h"
#include "opencv2/opencv.hpp"
#include "Struct_cv.h"
using namespace std;

using namespace cv;


#define MIN_FACE_SIZE 80

std::vector<SeetaFaceInfo> DetectFace(seeta::FaceDetector &FD, const SeetaImageData &image) {
    auto faces = FD.detect(image);

    return vector<SeetaFaceInfo>(faces.data, faces.data + faces.size);
}

std::vector<SeetaPointF> DetectPoints(seeta::FaceLandmarker &PD, const SeetaImageData &image, const SeetaRect &face) {
    vector<SeetaPointF> points(PD.number());
    PD.mark(image, face, points.data());

    return std::move(points);
}



static int64_t RegisterFace(const cv::String& filePath, seeta::FaceDetector &FD, seeta::FaceLandmarker &PD,
                            seeta::FaceDatabase &FDB) {
    cout << "Start Registering face..." << endl;
    FD.set(seeta::FaceDetector::PROPERTY_MIN_FACE_SIZE, MIN_FACE_SIZE);
    seeta::cv::ImageData image = cv::imread(filePath);
    vector<SeetaFaceInfo> faces = DetectFace(FD, image);
    vector<SeetaPointF> points = DetectPoints(PD, image, faces[0].pos);
    auto id = FDB.Register(image, points.data());

    cout << "The face register ends!" <<endl;

    return id;
}




int main(void) {

    cout << "Hello seeta" << endl;

    seeta::ModelSetting::Device device = seeta::ModelSetting::CPU;
    int id = 0;

    seeta::ModelSetting FR_model("../models/fr_2_10.dat", device, id);
    seeta::FaceDatabase FDB(FR_model);

    testgdb();

    bool ret = FDB.Save("../tester/Seeta.db");

    if(ret)
        cout << "Success in save" << endl;
    else
        cout << "Fail in save" << endl;

    bool ret1 = FDB.Load("../tester/Seeta.db");

    if(ret1)
        cout << "Success in load" << endl;
    else
        cout << "Fail in load" << endl;


    cv::Mat image = cv::imread("../tester/pics/test1.jpg");

    // 检查图片是否成功读取
    if (image.empty()) {
        std::cerr << "Error: Image could not be read." << std::endl;
        return -1;
    }
    cout << "Image successfully read" << endl;
    return 0;

}



//int main( int argc,char** argv )
//{
//    cv::Mat img_rgb,img_gry,img_any,img_dep,img_unc;
//    cv::namedWindow("Example RGB", cv::WINDOW_AUTOSIZE );
//    cv::namedWindow("Example GRY", cv::WINDOW_AUTOSIZE );
//    cv::namedWindow("Example ANY", cv::WINDOW_AUTOSIZE );
//    cv::namedWindow("Example DEP", cv::WINDOW_AUTOSIZE );
//    cv::namedWindow("Example UNC", cv::WINDOW_AUTOSIZE );
//
//    //
//    img_rgb=cv::imread( argv[1],cv::IMREAD_COLOR);
//    if( !img_rgb.empty() ) 	cv::imshow("Example RGB",img_rgb );
//    //
//    img_gry=cv::imread( argv[1],cv::IMREAD_GRAYSCALE);
//    if( !img_gry.empty() ) 	cv::imshow("Example GRY",img_gry );
//    //
//    img_any=cv::imread( argv[1],cv::IMREAD_ANYCOLOR);
//    if( !img_any.empty() ) 	cv::imshow("Example ANY",img_any );
//    //
//    img_dep=cv::imread( argv[1],cv::IMREAD_ANYDEPTH);
//    if( !img_dep.empty() ) 	cv::imshow("Example DEP",img_dep );
//    //
//    img_unc=cv::imread( argv[1],cv::IMREAD_UNCHANGED);
//    if( !img_unc.empty() ) 	cv::imshow("Example UNC",img_unc );
//
//    cv::waitKey(0);
//    return(0);
//}
