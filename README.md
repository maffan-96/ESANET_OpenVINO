# RealSense Semantic Segmentation with OpenVINO and Open3D
## Description:
This project demonstrates real-time semantic segmentation using a RealSense depth camera, the OpenVINO toolkit for deep learning inference, and Open3D for 3D visualization. The code captures color and depth frames from a RealSense camera, performs semantic segmentation on the color frames using a pre-trained deep learning model, and then combines the segmented results with depth information to create a point cloud visualization.

## Key Features:

* Utilizes the RealSense camera to capture color and depth frames.
* Integrates OpenVINO to perform real-time semantic segmentation using a pre-trained model.
* Visualizes the segmentation results overlaid on color frames.
* Creates a 3D point cloud visualization by combining segmented images and depth data.
* Provides average inference time metrics for performance evaluation.

##  Usage:

* Ensure all required dependencies (RealSense SDK, OpenVINO, Open3D) are properly installed.
* Configure the model path, device settings, and other parameters as needed.
* Run the Python script, and it will start capturing and processing frames.
* Press 'q' to exit the application.

##  Contributing:
Contributions and enhancements to this project are welcome. Please submit issues and pull requests if you find bugs or have ideas for improvements.

##  Acknowledgments:

* Credits to the RealSense, OpenVINO, and Open3D communities for their contributions and open-source libraries.
* Original [ESANET Repository ]([url](https://github.com/TUI-NICR/ESANet)https://github.com/TUI-NICR/ESANet) implementation
