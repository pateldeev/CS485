# CS 485 (Computer Vision) Project #1
Singular Value Decomposition Applications
Deev Patel

## Building
All the code, for both questions, is contained in the PA01.cpp file. It must be linked to [OpenCV](https://opencv.org/releases.html) to work. I tested the program with OpenCV version [3.2.0](https://docs.opencv.org/3.2.0/), but it may very well work with other versions. When run, the program will display the detected version of OpenCV.
```bash
g++ -std=c++0x PA01.cpp -o PA01 `pkg-config --cflags --libs opencv`
./PA01
```

## Running & Parameters
In order to run correctly, the program needs a file containing information about each image (including its location in memory and the four feature points).
See example file, "annotated_locations.txt", for details
You can provide a different file (although it must be of the same format) via command line arguments
```bash
./PA01 <path_to_location_file>
```
Note: if no file is provided, the program will try to open "annotated_locations.txt", which must be located in the same folder as the executable.

## Results & Output
The program executes both questions (feature normalize and lighting normalization). After every image, the results are displayed to the screen in various windows showing everything from the original features to the normalized image to the lighting model. You must PRESS A KEY in order to move onto the next image mentioned in the data file. Pressing 'q' or 'ESC' will end the program early. The output of every processed image, include the recovered parameters are stored in files located in the same directory of the input image and start with the same name. 
* <img>_features_normalized.png: contains the normalized image for question 1.
* <img>_lighting_normalized.png: contains the normalized image for question 2.
* <img>_recovered_parameters.txt: contains the recovered parameters for both the affine transformation and lighting model.
