//Main source file for Project 2 of CS 485 
//Spring 2019, Deev Patel

#include "PA02.h"

//main driver function for Questions 1 & 2

int main(int argc, char *argv[]) {
    std::cout << "||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||" << std::endl;
    std::cout << "Detected OpenCV version : " << CV_VERSION << std::endl << "Note: This program was tested on 3.2.0, but it should work on most other versions" << std::endl;
    std::cout << "||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||" << std::endl;

    std::vector<cv::Mat> images;
    //hard coded images locations and compression level for simplicity
    const std::vector<std::string> img_files_Q1 = {"images/P-1/ElCapitan.jpg", "images/P-1/Lilly.jpg", "images/P-1/Orchids.jpg", "images/P-1/OrchidsY.jpg", "images/P-1/Parrot.jpg"};
    const std::vector<unsigned int> compression_levels = {16, 8, 4, 2};
    const std::vector<std::string> img_files_Q2 = {"images/P-2/LiDAR01.png"};

    std::cout << "Executing Question 1!" << std::endl;
    std::cout << "  " << "Loading Images" << std::endl;
    Helper::readImages(images, img_files_Q1); //read images
    if (images.size() != img_files_Q1.size()) { //not all images correctly
        std::cout << "  " << "ERROR: One of more images was not read. Program will end. Ensure all the image files are correct!" << std::endl;
        return -1;
    }
    std::cout << "  " << "Done Loading Images" << std::endl;

    std::cout << "  " << "Now Processing Images" << std::endl;
    for (unsigned int i = 0; i < img_files_Q1.size(); ++i) {
        std::cout << "   " << "Processing Image: " << img_files_Q1[i] << std::endl;

        std::vector<cv::Mat> toon_shaded = {images[i].clone()}, color_compressed = {images[i].clone()};

        //perform toon shading
        std::cout << "     " << "Performing Toon Shading at required compression levels" << std::endl;
        for (unsigned int compression_level : compression_levels)
            toon_shaded.push_back(Q1::performToonShading(images[i], compression_level));
        std::cout << "     " << "Toon Shading complete" << std::endl;


        //perform color compression (toon shading with domain change)
        std::cout << "     " << "Performing Color Compression at required compression levels" << std::endl;
        for (unsigned int compression_level : compression_levels)
            color_compressed.push_back(Q1::performColorCompression(images[i], compression_level));
        std::cout << "     " << "Color Compression complete" << std::endl;

        //display and save results
        std::string result_file_name = img_files_Q1[i].substr(0, img_files_Q1[i].find_first_of('.'));
        result_file_name += "_results.png";
        std::cout << "     " << "Now displaying results to screen and saving them to: " << result_file_name << std::endl;
        //combine results into one image
        std::vector<cv::Mat> temp_disp = toon_shaded;
        for (const cv::Mat &m : color_compressed)
            temp_disp.push_back(m.clone());
        cv::Mat result = Helper::combineImagesIntoOne(temp_disp, 2, toon_shaded.size(), cv::Size(images[i].cols * (1 + toon_shaded.size()), 2 * images[i].rows));

        Helper::saveImg(result, result_file_name); //save results to file
        Helper::displayImg(result, img_files_Q1[i] + "|Top:Toon|Bottom:Color|Left-to-Right:Orig,16,8,4,2|", cv::Size(1800, 1000)); //display result to screen
    }

    std::cout << "||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||" << std::endl;
    std::cout << "Executing Question 2!" << std::endl;
    std::cout << "  " << "Loading Images" << std::endl;
    Helper::readImages(images, img_files_Q2, false); //read images
    if (images.size() != img_files_Q2.size()) { //not all images correctly
        std::cout << "  " << "ERROR: One of more images was not read. Program will end. Ensure all the image files are correct!" << std::endl;
        cv::destroyAllWindows();
        return -1;
    }
    std::cout << "  " << "Done Loading Images" << std::endl;

    std::cout << "  " << "Now Processing Images" << std::endl;
    for (unsigned int i = 0; i < img_files_Q2.size(); ++i) {
        std::cout << "   " << "Processing Image: " << img_files_Q2[i] << std::endl;
        std::vector<cv::Vec3i> circles = Q2::detectAllCircles(images[i]);
        std::cout << "   " << "Detected Circles: " << circles.size() << std::endl;

        //save results and display to screen
        std::cout << "   " << "Now Displaying and Saving Results" << std::endl;
        std::string result_file_name = img_files_Q2[i].substr(0, img_files_Q2[i].find_first_of('.'));
        cv::Mat results_broad = Helper::drawCircles(images[i], circles, cv::Scalar(255, 0, 0), false);
        cv::Mat results_fine = Helper::drawCircles(images[i], circles);

        Helper::displayImg(results_broad, img_files_Q2[i] + "_circles_broad", cv::Size(1800, 1000));
        Helper::displayImg(results_fine, img_files_Q2[i] + "_cirlces_fine", cv::Size(1800, 1000));

        Helper::saveImg(results_broad, result_file_name + "_results_broad.png"); //save results to file
        Helper::saveImg(results_fine, result_file_name + "_results_fine.png"); //save results to file
    }

    //done
    std::cout << "||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||" << std::endl;
    cv::waitKey(); //wait for key press to close everything
    cv::destroyAllWindows();
    return 0;
}