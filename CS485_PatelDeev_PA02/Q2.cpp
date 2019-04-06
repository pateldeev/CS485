//Source file for functions relating Question 2 of Project 2 of CS 485 
//Spring 2019, Deev Patel

#include "PA02.h"

namespace Q2 {

    std::vector<cv::Vec3i> detectAllCircles(const cv::Mat &img) {
        std::vector<cv::Vec3i> circles_detected_certain = detectCertain(img); //fast detection of larger circles
        std::vector<cv::Vec3i> circles_detected_small = detectSmall(img); //detect small circles through slow process

        //combine detections and remove repeated
        std::vector<cv::Vec3i> circles_detected = circles_detected_certain;
        circles_detected.insert(circles_detected.end(), circles_detected_small.begin(), circles_detected_small.end());
        filterDetections(circles_detected); //remove repeated

        //return circles_detected
        return circles_detected;
    }

    std::vector<cv::Vec3i> detectCertain(const cv::Mat &img) {
        CV_Assert(img.type() == CV_8UC1);
        std::vector<cv::Vec3f> hough_circles; //used to hold detections from hough_transform
        std::vector<cv::Vec3i> circles_detected;
        cv::Mat blurred_img; //hold blurred image

        //hough transform parameters
        int ctr_thresh, min_r_dist, min_r, max_r; //will change these parameters
        const int canny_thresh = 250; //fix canny threshold
        const int dp = 1; //fix dp

        //detect tiny trees with certainty
        min_r = 2;
        max_r = 5;
        min_r_dist = 20;
        ctr_thresh = 16;
        cv::GaussianBlur(img, blurred_img, cv::Size(5, 5), 1, 1);
        applyHoughCircles(blurred_img, hough_circles, dp, min_r_dist, canny_thresh, ctr_thresh, min_r, max_r);

        //detect small trees with certainty by increasing radius requirements by 2
        min_r *= 2;
        max_r *= 2;
        min_r_dist += 5; //increase distance between possible detections as circles will be bigger
        ctr_thresh = 19; //increase threshold as circles are bigger and will have more votes
        cv::GaussianBlur(img, blurred_img, cv::Size(5, 5), 1.5, 1.5); //increase blur rate
        applyHoughCircles(blurred_img, hough_circles, dp, min_r_dist, canny_thresh, ctr_thresh, min_r, max_r);

        //detect medium trees with certainty by increasing radius requirements by 2
        min_r *= 2;
        max_r *= 2;
        min_r_dist += 5; //increase distance between possible detections as circles will be bigger
        ctr_thresh = 26; //increase threshold as circles are bigger and will have more votes
        cv::GaussianBlur(img, blurred_img, cv::Size(7, 7), 1.75, 1.75); //increase blur rate
        applyHoughCircles(blurred_img, hough_circles, dp, min_r_dist, canny_thresh, ctr_thresh, min_r, max_r);

        //detect large trees with certainty by increasing radius requirements by 2
        cv::GaussianBlur(img, blurred_img, cv::Size(7, 7), 2, 2);
        min_r *= 2;
        max_r *= 2;
        min_r_dist += 5; //increase distance between possible detections as circles will be bigger
        ctr_thresh = 32; //increase threshold as circles are bigger and will have more votes
        cv::GaussianBlur(img, blurred_img, cv::Size(7, 7), 1.75, 1.75); //increase blur rate
        applyHoughCircles(blurred_img, hough_circles, dp, min_r_dist, canny_thresh, ctr_thresh, min_r, max_r);

        //convert detections to usable form
        for (const cv::Vec3f &c : hough_circles)
            circles_detected.emplace_back(cvRound(c[0]), cvRound(c[1]), cvRound(c[2]));

        filterDetections(circles_detected); //ensure no detections are repeated

        return circles_detected;
    }

    std::vector<cv::Vec3i> detectSmall(const cv::Mat &img) {
        CV_Assert(img.type() == CV_8UC1);
        std::vector<cv::Vec3f> hough_circles; //used to hold detections from hough_transform
        std::vector<cv::Vec3i> circles_detected;

        //blur image
        cv::Mat blurred_img;
        cv::GaussianBlur(img, blurred_img, cv::Size(5, 5), 1, 1);
        //hough transform parameters
        int min_r, max_r; //will change these parameters
        const int canny_thresh = 250; //fix canny threshold
        const int dp = 1; //fix dp
        const int min_r_dist = 10; //fix min radius
        const int ctr_thresh = 10; //fix detection threshold at small value

        for (int r = 2; r <= 6; ++r) { //detect tiny circles in both blurred and normal image at each radius level
            min_r = max_r = r;
            applyHoughCircles(img, hough_circles, dp, min_r_dist, canny_thresh, ctr_thresh, min_r, max_r);
            applyHoughCircles(blurred_img, hough_circles, dp, min_r_dist, canny_thresh, ctr_thresh, min_r, max_r);
        }

        //convert detections to usable form
        for (const cv::Vec3f &c : hough_circles)
            circles_detected.emplace_back(cvRound(c[0]), cvRound(c[1]), cvRound(c[2]));

        filterDetections(circles_detected, 3); //remove circles that aren't detected enough times 

        return circles_detected;
    }

    void applyHoughCircles(const cv::Mat &img, std::vector<cv::Vec3f> &circles, const int dp, const int min_r_dist, const int canny_thresh, const int ctr_thresh, const int min_r, const int max_r) {
        std::vector<cv::Vec3f> new_circles;
        cv::HoughCircles(img, new_circles, cv::HOUGH_GRADIENT, dp, min_r_dist, canny_thresh, ctr_thresh, min_r, max_r);
        circles.insert(circles.end(), new_circles.begin(), new_circles.end());
    }

    void filterDetections(std::vector<cv::Vec3i> &detections, unsigned int keep_thresh) {
        std::vector<unsigned int> detection_frequecies(detections.size(), 1); //keeps track of number of merges per detection - needed to apply threshold
        for (unsigned int i = detections.size() - 1; i > 0; --i) {
            //look for similar regions from front to merge with
            for (unsigned int j = 0; j < i; ++j) {
                int center_distance = std::abs(detections[i][0] - detections[j][0]) + std::abs(detections[i][1] - detections[j][1]);
                if (center_distance < 8 && std::abs(detections[i][2] - detections[j][2]) < 1.5 * std::min(detections[i][2], detections[j][2])) { //merge circles if centers and radii are close enough together
                    //form new circle by averaging centers and taking maximum radius
                    detections[j] = cv::Vec3i(cvRound(0.5 * (detections[i][0] + detections[j][0])), cvRound(0.5 * (detections[i][1] + detections[j][1])), std::max(detections[i][2], detections[j][2]));
                    ++detection_frequecies[j]; //record merge

                    //remove detection j by replacing it with last element and reducing size of vector by 1 - for effeciency 
                    detections[i] = detections.back();
                    detection_frequecies[i] = detection_frequecies.back();
                    detections.pop_back();
                    detection_frequecies.pop_back();
                }
            }
        }

        if (keep_thresh) { //threshold results if requested
            detections.erase(std::remove_if(detections.begin(), detections.end(), [&keep_thresh, &detection_frequecies, &detections](const cv::Vec3i & v)->bool {
                int index = &v - &detections[0]; //calculate index based on pointers since vector is continuous
                return (detection_frequecies[index] < keep_thresh);
            }), detections.end());
        }
    }
};