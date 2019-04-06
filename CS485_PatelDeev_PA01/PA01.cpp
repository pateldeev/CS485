#include <opencv2/opencv.hpp>

//helper function to display image to screen

char displayImg(const cv::Mat &img, const std::string &window_name, bool wait = false, const cv::Size &display_size = cv::Size(1200, 800)) {
    cv::namedWindow(window_name, cv::WINDOW_NORMAL);
    cv::resizeWindow(window_name, display_size.width, display_size.height);
    cv::imshow(window_name, img);
    return (wait) ? cv::waitKey() : 0;
}

//helper function to draw feature points on image

void drawPoints(const cv::Mat &img, const std::vector<cv::Point> &points, const cv::Scalar &color = cv::Scalar(255, 0, 0), int size = 0) {
    for (const cv::Point &p : points)
        cv::circle(img, p, size, color);
}

//helper function to parse command line arguments and get buffer of locations data

bool intitalizeAndGetLocationsBuffer(int argc, char *argv[], std::stringstream &locations_buffer) {
    std::cout << std::endl << "Detected OpenCV version : " << CV_VERSION << std::endl << "Note: This program was tested and ran on 3.2.0" << std::endl;

    std::string locations_file_name = "annotated_locations.txt";
    if (argc == 1) {
        std::cout << "No parameters passed. Using default file location for annotated data: " << locations_file_name << std::endl;
    } else if (argc == 2) {
        locations_file_name = argv[1];
        std::cout << "Using given location for annotated data: " << locations_file_name << std::endl;
    } else {
        std::cerr << "Detected incorrect number of command line arguments. Either provide none to run with default configurations or provide the name of the annotated date file." << std::endl;
        return false;
    }

    std::ifstream locations_file(locations_file_name); //open data file 

    if (!locations_file.is_open()) { //ensure file opened
        std::cerr << "Could not open locations file: " << locations_file_name << std::endl;
        return false;
    }

    //move data to string stream buffer
    locations_buffer.clear();
    locations_buffer << locations_file.rdbuf();
    locations_file.close();
    return true;
}

//function to parse annotated data file, opens next image and get points mentioned in location buffer
//assumes location buffer/file is correctly formatted --- see example "annotated_locations.txt" file
//stores points in the order they appear in the file
//Note: requires stringstring version of datafile - can be created via ifstream.rdbuf()

bool getNextImage(std::stringstream &data, cv::Mat &img, std::vector<cv::Point> &points, std::string &img_file_name) {
    //skip any comments/extra lines
    std::string temp;
    while (!data.eof() && (!std::isalpha(data.peek()) || data.peek() == '#'))
        std::getline(data, temp);

    if (data.eof())
        return false; //no more data to get

    //open image
    std::getline(data, img_file_name, ';');
    img = cv::imread(img_file_name, cv::IMREAD_COLOR);
    CV_Assert(img.type() == CV_8UC3); //only works with RBG image

    //get points
    points.clear();
    int x, y;
    char temp_char;
    while (std::isdigit(data.peek())) {
        data >> x >> temp_char >> y;
        if (data.peek() == ';')
            data >> temp_char;
        points.emplace_back(x, y);
    }
    return true;
}

//function to use OpenCV's SVD class to obtain a solution/estimate to a set of linear system(s).
//requires inputs to be floating point matrices representing a set of system(s)
//computes SVD on one matrix and can efficiently solve multiple systems with same that only differ on the right hand side
//returns one floating point column matrix for each solution set

std::vector<cv::Mat> getSVDSolution(const cv::Mat &coefficients, const std::vector<cv::Mat> &rhs_list) {
    CV_Assert(coefficients.type() == CV_32F); //ensure coeffecients are correctly formatted
    std::vector<cv::Mat> solutions(rhs_list.size()); //solutions vector

    cv::SVD solver(coefficients); //create solver - performs a single SVD decomposition internally

    for (int i = 0; i < rhs_list.size(); ++i) {
        CV_Assert(rhs_list[i].type() == CV_32F && coefficients.rows == rhs_list[i].rows && rhs_list[i].cols == 1); //ensure rhs is correctly formated
        solver.backSubst(rhs_list[i], solutions[i]); //get solution
    }
    return solutions;
}

//function to apply affine transformation to image - uses inverse mapping with nearest neighbor interpolation
//img_transformed must already be of the size you want to map to
//also applies forward transformation to each of the key points given to see where they likely end up
//affine_x = [a11,a12,b1]^T
//affine_y = [a21,a22,b2]^T

void applyAffineTransformation(const cv::Mat &img, cv::Mat &img_transformed, const cv::Mat &affine_x, const cv::Mat &affine_y, const std::vector<cv::Point> &features, std::vector<cv::Point> &features_transformed) {
    //ensure parameters are correctly formatted/allocated
    CV_Assert(img.type() == CV_8UC3 && img_transformed.type() == CV_8UC3 && affine_x.type() == CV_32F && affine_y.type() == CV_32F);
    CV_Assert(affine_x.rows == 3 && affine_x.cols == 1 && affine_y.rows == 3 && affine_y.cols == 1 && img_transformed.rows > 0 && img_transformed.cols > 0);
    features_transformed.clear();

    //create 2x2 affine matrix (A) that can be used as a coefficient matrix in SVD
    cv::Mat affine(2, 2, CV_32F);
    for (int i = 0; i < 2; ++i) {
        affine.at<float>(0, i) = affine_x.at<float>(i, 0);
        affine.at<float>(1, i) = affine_y.at<float>(i, 0);
    }

    //store 2x1 matrix corresponding to every point in the transformed image so that we solve an SVD system for its pre-image
    std::vector<cv::Mat> transformed_points(img_transformed.rows * img_transformed.cols);
    std::vector<cv::Mat>::iterator iter_transformed = transformed_points.begin();
    for (int x_transformed = 0; x_transformed < img_transformed.cols; ++x_transformed) {
        for (int y_transformed = 0; y_transformed < img_transformed.rows; ++y_transformed) {
            *iter_transformed = cv::Mat(2, 1, CV_32F);
            iter_transformed->at<float>(0, 0) = float(x_transformed) - affine_x.at<float>(2, 0);
            iter_transformed->at<float>(1, 0) = float(y_transformed) - affine_y.at<float>(2, 0);
            ++iter_transformed;
        }
    }
    assert(iter_transformed == transformed_points.end()); //ensure we got all the points

    //compute solutions to every backward mapping via SVD
    std::vector<cv::Mat> original_locations = getSVDSolution(affine, transformed_points);

    //go through and find transformed pixel values by interpolating to nearest neighbor in original image 
    std::vector<cv::Mat>::iterator iter_original = original_locations.begin();
    for (int x_transformed = 0; x_transformed < img_transformed.cols; ++x_transformed) {
        for (int y_transformed = 0; y_transformed < img_transformed.rows; ++y_transformed) {
            int x_original = int(iter_original->at<float>(0, 0) + 0.5); //add 0.5 for correct rounding - nearest neighbor
            int y_original = int(iter_original->at<float>(1, 0) + 0.5); //add 0.5 for correct rounding - nearest neighbor
            ++iter_original;

            //ensure x and y are in bounds of image - if not make them
            if (x_original < 0)
                x_original = 0;
            else if (x_original >= img.cols)
                x_original = img.cols - 1;
            if (y_original < 0)
                y_original = 0;
            else if (y_original >= img.rows)
                y_original = img.rows - 1;

            //copy over value to transformed image
            img_transformed.at<cv::Vec3b>(y_transformed, x_transformed) = img.at<cv::Vec3b>(y_original, x_original);
        }
    }
    assert(iter_original == original_locations.end()); //ensure we got all the points

    //transform features in the forward direction and round to get an idea of where they end up
    for (const cv::Point &p : features) {
        float new_x = p.x * affine_x.at<float>(0, 0) + p.y * affine_x.at<float>(1, 0) + affine_x.at<float>(2, 0);
        float new_y = p.x * affine_y.at<float>(0, 0) + p.y * affine_y.at<float>(1, 0) + affine_y.at<float>(2, 0);
        features_transformed.emplace_back(int(new_x + 0.5), int(new_y + 0.5)); //add 0.5 for correct rounding - nearest neighbor
    }
}

//helper function to save and display results of question 1 - normalized image and affine transformation
//affine_x = [a11,a12,b1]^T
//affine_y = [a21,a22,b2]^T
//returns name of parameters file created for image

std::string save_and_display_results_Q1(const cv::Mat &img, const std::vector<cv::Point> &img_features, const std::string &img_file, const cv::Mat &img_normalized, const std::vector<cv::Point> &img_features_normalized, const std::vector<cv::Point> &features_desired, const cv::Mat &affine_x, const cv::Mat &affine_y) {
    const std::string log_file = img_file.substr(0, img_file.find_first_of('.')) + "_recovered_parameters.txt";
    const std::string save_file_name = img_file.substr(0, img_file.find_first_of('.')) + "_features_normalized.png";

    //save normalized version
    cv::imwrite(save_file_name, img_normalized);

    //log statistics - including transformation parameters
    std::ofstream log(log_file);
    log << "Results of normalization on image: " << img_file << std::endl << std::endl;
    log << "affine_matrix_a:" << "[[" << affine_x.at<float>(0, 0) << ' ' << affine_x.at<float>(1, 0) << "] [" << affine_y.at<float>(0, 0) << ' ' << affine_y.at<float>(1, 0) << "]]" << std::endl;
    log << "affine_matrix_b:" << "[[" << affine_x.at<float>(2, 0) << "] [" << affine_y.at<float>(2, 0) << "]]" << std::endl << std::endl;

    const std::vector<std::string> feature_names = {"left_eye_center", "right_eye_center", "nose_tip", "lip_center"};
    cv::Point error;
    float error_mag, error_sum;
    for (int i = 0; i < feature_names.size(); ++i) {
        error = img_features_normalized[i] - features_desired[i];
        error_mag = std::sqrt(error.x * error.x + error.y * error.y);
        error_sum += error_mag;
        log << feature_names[i] << ':' << img_features[i] << " --> " << img_features_normalized[i] << "  (Desired: " << features_desired[i] << "---> Error: " << error << '=' << error_mag << ')' << std::endl;
    }
    log << std::endl << "avg_error_magnitude:" << error_sum / feature_names.size() << std::endl;
    log.close();

    //draw feature points and display on screen for verification
    cv::Mat disp_original = img.clone(), disp_normalized = img_normalized.clone();
    drawPoints(disp_original, img_features, cv::Scalar(0, 50, 255), 1);
    drawPoints(disp_normalized, img_features_normalized, cv::Scalar(0, 50, 255)); //draw actual, normalized features in blue
    drawPoints(disp_normalized, features_desired, cv::Scalar(255, 0, 0)); //draw desired features in blue    displayImg(disp_original, "Original");
    displayImg(disp_original, "Original_Features");
    displayImg(disp_normalized, "Normalized_Features---Blue=desired");

    return log_file;
}

//helper function to save and display results of question 2 - lighting normalized image
//lighting_parameters = [a,b,c,d]

void save_and_display_results_Q2(const cv::Mat &img_normalized, const cv::Mat &lighting_model, const std::vector<float> lighting_parameters, const std::string save_file_name, const std::string &log_file) {
    assert(lighting_parameters.size() == 4); //make sure parameters are valid

    cv::imwrite(save_file_name, img_normalized); //save normalized version

    //log statistics - lighting parameters
    std::ofstream log(log_file, std::ios_base::app);
    log << std::endl << "Parameters of lighting model: a=" << lighting_parameters[0] << " b=" << lighting_parameters[1] << " c=" << lighting_parameters[2] << " d=" << lighting_parameters[3];

    displayImg(lighting_model, "Lighting_Model");
    displayImg(img_normalized, "Normalized_Lighting");
}

//helper function to perform lighting normalization - question 2
//saves parameters and result to given files
//also displays results

void performLightingNormalization(const cv::Mat &img, cv::Mat &img_normalized, const std::string &save_file, const std::string &log_file) {
    img_normalized = img.clone();
    if (img_normalized.type() == CV_8UC3)
        cv::cvtColor(img_normalized, img_normalized, cv::COLOR_RGB2GRAY);
    CV_Assert(img_normalized.type() == CV_8UC1); //normalization only setup to work on gray-scale images

    //create and populate matrices to represent all nxm equations (and 4 unknowns)
    cv::Mat coeffecients(img.rows * img.cols, 4, CV_32F), rhs(img.rows * img.cols, 1, CV_32F);
    float *coeff_ptr = coeffecients.ptr<float>(0); //use pointers to underlying data for simplicity
    float *rhs_ptr = rhs.ptr<float>(0); //use pointers to underlying dasta for simplicity
    for (int x = 0; x < img.cols; ++x) {
        for (int y = 0; y < img.rows; ++y) {
            coeff_ptr[0] = x; //scale factor of 'a' parameter of lighting model
            coeff_ptr[1] = y; //scale factor of 'b' parameter of lighting model
            coeff_ptr[2] = x * y; //scale factor of 'c' parameter of lighting model
            coeff_ptr[3] = 1; //scale factor of 'd' parameter
            rhs_ptr[0] = float(img_normalized.at<uint8_t>(y, x));
            //move onto to next row
            coeff_ptr += 4;
            ++rhs_ptr;
        }
    }

    //solve overdetermined system using SVD and get parameters of lighting model
    cv::Mat parameters = getSVDSolution(coeffecients, std::vector<cv::Mat>(1, rhs))[0];
    float a = parameters.at<float>(0, 0); //'a' parameter of lighting model
    float b = parameters.at<float>(1, 0); //'b' parameter of lighting model
    float c = parameters.at<float>(2, 0); //'c' parameter of lighting model
    float d = parameters.at<float>(3, 0); //'d' parameter of lighting model

    //create image with values predicted by lighting
    cv::Mat lighting_model(img.rows, img.cols, CV_8UC1);
    for (int x = 0; x < img.cols; ++x) {
        for (int y = 0; y < img.rows; ++y) {
            int predicted_value = int(a * x + b * y + c * x * y + d + 0.5); //add 0.5 for rounding
            //ensure predicted value is within domain
            if (predicted_value > 255)
                predicted_value = 255;
            if (predicted_value < 0)
                predicted_value = 0;
            lighting_model.at<uint8_t>(y, x) = predicted_value; //set predicted value
        }
    }

    //create corrected image - have to first used signed integer image because the prediction could be greater or less than original
    cv::Mat corrected_img(img.rows, img.cols, CV_8SC1);
    int temp_val;
    for (int r = 0; r < img.rows; ++r) {
        for (int c = 0; c < img.cols; ++c) {
            temp_val = img_normalized.at<uint8_t>(r, c) - lighting_model.at<uint8_t>(r, c);
            //ensure value is in acceptable range
            if (temp_val > 127)
                temp_val = 127;
            else if (temp_val < -128)
                temp_val = -128;
            corrected_img.at<int8_t>(r, c) = temp_val;
        }
    }

    //add 128 to each pixel value to renormalize corrected image to 0-255 range --> could have done this in earlier step for computational efficiency
    for (int r = 0; r < img.rows; ++r)
        for (int c = 0; c < img.cols; ++c)
            img_normalized.at<uint8_t>(r, c) = corrected_img.at<int8_t>(r, c) + 128;

    save_and_display_results_Q2(img_normalized, lighting_model, std::vector<float>({a, b, c, d}), save_file, log_file);
}

int main(int argc, char *argv[]) {
    //fixed locations of where we want points to map to
    std::vector<cv::Point> features_desired;
    features_desired.emplace_back(12, 19); //desired location of left-eye-center
    features_desired.emplace_back(27, 19); //desired location of right-eye-center
    features_desired.emplace_back(20, 27); //desired location of nose-tip
    features_desired.emplace_back(20, 38); //desired location of lip-center

    //move desired x and y values to matrices - and place them in one vector - needed to latter perform SVD
    cv::Mat p_desired_x(features_desired.size(), 1, CV_32F), p_desired_y(features_desired.size(), 1, CV_32F);
    for (int i = 0; i < features_desired.size(); ++i) {
        p_desired_x.at<float>(i, 0) = features_desired[i].x;
        p_desired_y.at<float>(i, 0) = features_desired[i].y;
    }
    const std::vector<cv::Mat> p_desired = {p_desired_x, p_desired_y};

    //open locations data file and parse command line arguments
    std::stringstream locations_data_buffer;
    if (!intitalizeAndGetLocationsBuffer(argc, argv, locations_data_buffer))
        return -1;

    //variables to hold image, and normalized image
    std::string img_file;
    const cv::Size normalized_size(40, 48); //fixed size of normalized image
    cv::Mat img, img_normalized(normalized_size.height, normalized_size.width, CV_8UC3);

    //variables to store locations of features, and their coordinates in homogeneous form
    std::vector<cv::Point> img_features, img_features_normalized; //feature coordinates
    cv::Mat p_homogeneous(features_desired.size(), 3, CV_32F); //feature coordinates in homogeneous form

    //go through and process every image
    while (getNextImage(locations_data_buffer, img, img_features, img_file)) {
        assert(img_features.size() == features_desired.size()); //ensure each feature has a corresponding point it will be mapped to

        //move feature points to homogenous coordinate matrix
        for (int i = 0; i < img_features.size(); ++i) {
            p_homogeneous.at<float>(i, 0) = img_features[i].x;
            p_homogeneous.at<float>(i, 1) = img_features[i].y;
            p_homogeneous.at<float>(i, 2) = 1.f;
        }

        //compute least squares solutions using OpenCV SVD functionality - solve both systems using same SVD of homogeneous coordinate matrix
        std::vector<cv::Mat> affine_solutions = getSVDSolution(p_homogeneous, p_desired);
        CV_Assert(affine_solutions.size() == 2); //solutions: [a11, a12, b1]^T ; [a21,a22,b2]^T

        //apply affine transformation
        applyAffineTransformation(img, img_normalized, affine_solutions[0], affine_solutions[1], img_features, img_features_normalized);

        //save and display results for part 1
        std::string log_file = save_and_display_results_Q1(img, img_features, img_file, img_normalized, img_features_normalized, features_desired, affine_solutions[0], affine_solutions[1]);

        //perform part 2
        cv::Mat img_lighting_normalized;
        performLightingNormalization(img_normalized, img_lighting_normalized, img_file.substr(0, img_file.find_first_of('.')) + "_lighting_normalized.png", log_file);

        std::cout << std::endl << "Finished with:" << img_file << "! Now displaying results to screen." << std::endl << "Press any key to go onto next image. Enter q or ESC to exit" << std::endl;

        char key = cv::waitKey();
        if (key == 'q' || key == 27)
            break; //press q or ESC to exit early.
    }

    return 0;
}