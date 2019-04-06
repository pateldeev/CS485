//Source file for functions relating Question 1 of Project 2 of CS 485 
//Spring 2019, Deev Patel

#include "PA02.h"

namespace Q1 {

    cv::Mat performToonShading(const cv::Mat &img, unsigned int levels_remaining) {
        CV_Assert(img.type() == CV_8UC3 && levels_remaining <= 256 && levels_remaining > 0);
        cv::Mat result(img.rows, img.cols, CV_8UC3); //resultant image

        const float level_size = 256.f / levels_remaining;

        //iterate through data and change each pixel value
        const uchar *img_data = img.datastart;
        uchar *tooned_data = result.data;
        while (img_data != img.dataend)
            *tooned_data++ = uchar(int(*img_data++ / level_size) * level_size); //compute new value and go onto next pixel

        return result;
    }

    cv::Mat performColorCompression(const cv::Mat &img, unsigned int levels_remaining) {
        CV_Assert(img.type() == CV_8UC3);
        cv::Mat result; //resultant image

        const float level_size = 256.f / levels_remaining;

        cv::cvtColor(img, result, cv::COLOR_BGR2HSV); //convert to HSV domain

        //quantize saturation domain - will have little affect on visual appearance
        for (unsigned int row = 0; row < result.rows; ++row)
            for (unsigned int col = 0; col < result.cols; ++col)
                result.at<cv::Vec3b>(row, col)[1] = uchar(result.at<cv::Vec3b>(row, col)[1] / level_size) * level_size;

        cv::cvtColor(result, result, cv::COLOR_HSV2BGR); //convert back to BGR domain for visualization

        return result;
    }
};