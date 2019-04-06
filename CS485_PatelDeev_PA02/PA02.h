//Main header file for Project 2 of CS 485 
//Spring 2019, Deev Patel

#ifndef PA02_H
#define PA02_H

#include <opencv2/opencv.hpp>

//For Question 1
namespace Q1 {

    //function to perform toon shading. The input image must be a 8-bit color image. The the levels_remaining cannot be more than 256
    cv::Mat performToonShading(const cv::Mat &img, unsigned int levels_remaining);

    //function to perform color compression. The input image must be a 8-bit color image. The the levels_remaining cannot be more than 256
    //converts to HSV and compresses saturation
    cv::Mat performColorCompression(const cv::Mat &img, unsigned int levels_remaining);
};

//For Question 2
namespace Q2 {

    //function to detect circles using the Hough transform - each circle is described by (x,y,r), where r is the radius and (x,y) is the center
    //combines detectCertain and detectSmall
    std::vector<cv::Vec3i> detectAllCircles(const cv::Mat &img);

    //function to detect with high level of certainty. Detected at 4 different levels, blurring more or more of the image to detect larger circles
    std::vector<cv::Vec3i> detectCertain(const cv::Mat &img);

    //function to detect small circles - does multiple passes and keeps circles that appear multiple times - slow and less reliabl than detectCertain()
    std::vector<cv::Vec3i> detectSmall(const cv::Mat &img);

    //internal helper function to use OpenCV Hough Transform with given parameters
    void applyHoughCircles(const cv::Mat &img, std::vector<cv::Vec3f> &circles, const int dp, const int min_r_dist, const int canny_thresh, const int ctr_thresh, const int min_r, const int max_r);

    //internal helper function to remove duplicate detection
    //will combine any two circles that have significant overlap into one circle by averaging the centers and keeping the largest radius
    //specifying a nonzero keep threshold will only keep detections that appear at least the required number of times
    void filterDetections(std::vector<cv::Vec3i> &detections, unsigned int keep_thresh = 0);

};

//For helper functions
namespace Helper {

    //helper function to read images into vector. returns only the successfully read images
    void readImages(std::vector<cv::Mat> &images, const std::vector<std::string> img_file_names, bool read_color = true);

    //helper function to display image to screen. when display_size = 0, window will be sized based on image dimensions
    void displayImg(const cv::Mat &img, const std::string &window_name, const cv::Size &display_size = cv::Size(0, 0));

    //helper function to combine multiple CV_8UC1 or CV_8UC3 images into one - doesn't check number of images provided to make sure it fits
    //will automatically resize images in best way possible to fit new size
    cv::Mat combineImagesIntoOne(const std::vector<cv::Mat> &images, int num_rows = 2, int num_cols = 5, const cv::Size &display_size = cv::Size(1800, 1000));

    //helper function to draw circles onto image - each circle is described by (x,y,r), where r is the radius and (x,y) is the center
    cv::Mat drawCircles(const cv::Mat &img, const std::vector<cv::Vec3i> &circles, const cv::Scalar &color = cv::Scalar(0, 0, 255), bool draw_exactly = true);

    //helper function to save image to file. Saves in lossless mode png mode. (add .png to end of file_name if its not already there)
    void saveImg(const cv::Mat &img, const std::string &file_name);

}

#endif

