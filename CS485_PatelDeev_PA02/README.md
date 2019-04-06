# CS 485 (Computer Vision) Project #2
Compression & Hough Transform
Deev Patel

## Building
The project is split into various files. PA02.cpp contains the main driver function for both questions and PA02.h contains the declaration and usage of all the functions I created. The project must be linked to [OpenCV](https://opencv.org/releases.html) to work. Addititionally, the compiler must have c++11 or higher support. I tested the program with OpenCV version [3.2.0](https://docs.opencv.org/3.2.0/), but it may very well work with other versions. When run, the program will display the detected version of OpenCV.
```bash
g++ -std=c++0x PA02.cpp Q1.cpp Q2.cpp Helper.cpp -o PA02 `pkg-config --cflags --libs opencv`
./PA02
```
Note: I ran this project on Ubuntu 18.04 and everything worked fine. See the sample results.

## Running & Images Folder
In order to run correctly, the program needs a folder called '/images' in the same location as the executable. The images folder must have the following layout
* /images/
  * /P-1/    #folder for images for problem 1
    * /ElCapitan.jpg  #1st image to compress in problem 1
    * /Lilly.jpg      #2nd image to compress in problem 1
    * /Orchids.jpg    #3rd image to compress in problem 1
    * /OrchidsY.jpg   #4th image to compress in problem 1
    * /Parrot.jpg     #5th image to compress in problem 1
  * /P-2/    #folder for images for problem 2
    * /LiDAR01.png    #1st image to detect cirlces in problem 2
If one or more the above images cannot be opened, the program will not run. It will give an error as to which one could not be opened.

## Results & Output
The program executes both questions automatically. After every image, the results are displayed to the screen in various windows. 

In problem 1, one window will show both toon shading and color compression for each image. The top row shows toon shading with the original image on the left and progressively more and more compressed versions on the right. Similar, the bottom row shows color compression with the original image on the left and progressively more and more compressed versions on the right. The right most column will show compression with 2 color levels.

In problem 2, there will be two windows for each image. One window labled "fine" will display the exact detections in red. Another window labeled "broad" will display an outline of the actual detections in blue. This is to allow one to see the ground truth image in addition to the detection.

At the end, the program will pause to allow the user to inspect the results. Simply press any keyboard key to exit the program and close all the windows. Before closing, the program will save all the results to appropriately named files in the "/images" folder mentioned above.
