//
// Created by hzk on 24-9-18.
//

#ifndef SEETAFACETEST_STRUCT_CV_H
#define SEETAFACETEST_STRUCT_CV_H
#pragma once

#include <opencv2/core.hpp>
#include <seeta/CStruct.h>

namespace seeta
{
    namespace cv
    {
        class ImageData : public SeetaImageData {
        public:
            ImageData(const ::cv::Mat &mat)
                    : cv_mat(mat.clone()) {
                this->width = cv_mat.cols;
                this->height = cv_mat.rows;
                this->channels = cv_mat.channels();
                this->data = cv_mat.data;
            }

            ImageData(int width, int height, int channels = DEFAULT_CHNNAL)
                    : cv_mat(height, width, CV_8UC(channels)) {
                this->width = cv_mat.cols;
                this->height = cv_mat.rows;
                this->channels = cv_mat.channels();
                this->data = cv_mat.data;
            }
            ImageData(const SeetaImageData &img)
                    : cv_mat(img.height, img.width, CV_8UC(img.channels), img.data) {
                this->width = cv_mat.cols;
                this->height = cv_mat.rows;
                this->channels = cv_mat.channels();
                this->data = cv_mat.data;
            }
            ImageData()
                    : cv_mat() {
                this->width = cv_mat.cols;
                this->height = cv_mat.rows;
                this->channels = cv_mat.channels();
                this->data = cv_mat.data;
            }
            bool empty() const {
                return cv_mat.empty();
            }
            operator ::cv::Mat() const {
                return cv_mat.clone();
            }
            ::cv::Mat toMat() const {
                return cv_mat.clone();
            }
        private:
            ::cv::Mat cv_mat;

            const static int16_t DEFAULT_CHNNAL = 3;
        };
    }
}
#endif //SEETAFACETEST_STRUCT_CV_H
