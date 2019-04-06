//Source file for helper functions relating to Project 2 of CS 485 
//Spring 2019, Deev Patel

#include "PA02.h"

namespace Helper {

    void readImages(std::vector<cv::Mat> &images, const std::vector<std::string> img_file_names, bool read_color) {
        images.clear();
        for (const std::string file_name : img_file_names) {
            std::cout << "   " << "Reading image file: " << file_name << std::endl;

            //read image
            if (read_color)
                images.push_back(cv::imread(file_name));
            else
                images.push_back(cv::imread(file_name, cv::IMREAD_GRAYSCALE));

            if (images.back().empty()) { //handle unsuccessful read
                images.pop_back();
                std::cout << "   " << "   " << "ERROR: Could not read image. Check file name!" << std::endl;
            } else {
                std::cout << "   " << "   " << "SUCCESS: Image successfully loaded." << std::endl;
            }
        }
    }

    void displayImg(const cv::Mat &img, const std::string &window_name, const cv::Size &display_size) {
        cv::namedWindow(window_name, cv::WINDOW_NORMAL);
        if (display_size.width > 0 && display_size.height > 0)
            cv::resizeWindow(window_name, display_size.width, display_size.height);
        cv::imshow(window_name, img);
        cv::waitKey(100);
    }

    cv::Mat combineImagesIntoOne(const std::vector<cv::Mat> &images, int num_rows, int num_cols, const cv::Size &display_size) {
        cv::Mat disp_img = cv::Mat::zeros(display_size.height, display_size.width, CV_8UC3);
        const int img_target_width = display_size.width / num_cols, img_target_height = display_size.height / num_rows; //target size of each small subimage

        int w = 0, h = 0; //used to keep track of top left corner of each subimage
        cv::Mat img_resized;

        for (const cv::Mat &img : images) {
            CV_Assert(img.type() == CV_8UC1 || img.type() == CV_8UC3);

            //consider the two possible resizing options - must choose optimal one that takes up most space
            int img_resize_byH_width = img.cols * (float(img_target_height) / img.rows);
            int img_resize_byW_height = img.rows * (float(img_target_width) / img.cols);

            if (img_resize_byH_width <= img_target_width) //rescale so height fits perfectly
                cv::resize(img, img_resized, cv::Size(img_resize_byH_width, img_target_height));
            else //rescale so width fits perfectly
                cv::resize(img, img_resized, cv::Size(img_target_width, img_resize_byW_height));

            //copy over relevant pixel values
            for (unsigned int r = 0; r < img_resized.rows; ++r)
                for (unsigned int c = 0; c < img_resized.cols; ++c)
                    disp_img.at<cv::Vec3b>(r + h, c + w) = ((img.type() == CV_8UC3) ? img_resized.at<cv::Vec3b>(r, c) : cv::Vec3b(img_resized.at<uint8_t>(r, c), img_resized.at<uint8_t>(r, c), img_resized.at<uint8_t>(r, c)));

            //calculate start position of next image
            w += img_target_width;
            if (w >= display_size.width - img_target_width + 1) { //go to next row if needed
                w = 0;
                h += img_target_height;
            }
        }
        return disp_img;
    }

    cv::Mat drawCircles(const cv::Mat &img, const std::vector<cv::Vec3i> &circles, const cv::Scalar &color, bool draw_exactly) {
        cv::Mat disp;
        if (img.type() == CV_8UC1)
            cv::cvtColor(img, disp, cv::COLOR_GRAY2BGR);
        else if (img.type() == CV_8UC3)
            disp = img.clone();
        else
            CV_Assert(0); //incompatible type

        for (const cv::Vec3i &c : circles)
            cv::circle(disp, cv::Point(c[0], c[1]), (draw_exactly) ? c[2] : c[2] + 10, color, 2);

        return disp;
    }

    void saveImg(const cv::Mat &img, const std::string &file_name) {
        const std::vector<int> save_params = {cv::IMWRITE_PNG_COMPRESSION, 0}; //parameters needed to save in lossless png mode
        //add .png extension if not given
        if (file_name.find_last_of(".png") != file_name.size() - 1)
            cv::imwrite(file_name + ".png", img, save_params);
        else
            cv::imwrite(file_name, img, save_params);
    }
};