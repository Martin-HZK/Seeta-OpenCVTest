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

    


    return 0;
    
}