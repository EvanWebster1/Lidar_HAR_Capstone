
# Exercise Pose Detection with LiDAR: Mapping Human Movement

Database: https://curesearch.sharefile.com/home/shared/fo9d5d0c-7e98-41e3-8691-f7001f125458
Website: https://sites.google.com/view/lidar-h-a-r/home

##Abstract

This research presents a comprehensive approach to detecting and classifying human exercise poses using 3D LIDAR technology coupled with deep learning methodologies. Our study builds upon the foundational work of point cloud processing, introducing a series of deep learning models to recognize and classify human exercises accurately. 

The project progressed through initial concept, design, implementation of a human activity recognition model, and culminated in the development of a convolutional neural network (CNN) model that processes 3D LIDAR data for exercise pose detection. This paper outlines our methodology, including the novel use of the Point Pillars network and pseudo-image transformation for effective 3D to 2D data conversion, enhancing the applicability of conventional 2D CNNs for 3D spatial data analysis. By leveraging this network and developing bespoke CNNs, we mapped human movements to detect initially three, then six key exercises, overcoming traditional limitations of 2D vision-based systems. The subsequent application of the "smush" technique, demonstrated a significant improvement in the detection accuracy of the additional dynamic exercises, underscoring the potential of our approach in enhancing fitness monitoring technologies. 

We further discuss our iterative development process, the challenges faced, and propose directions for future research. This research not only demonstrates the effective use of LIDAR in capturing complex activities, but also lays the groundwork for future enhancements in mapping human movement using LIDAR.

## Authors

- Ray Huda
- Evan Webster [@EvanWebster1](https://www.github.com/EvanWebster1)
- Mitchell Zinck [@Zinckle](https://www.github.com/Zinckle)

## Advisor
- Dr. Marzieh Amini [@MaAmini](https://www.github.com/MaAmini)

## Documentation

### convert_multipcap_to_pcd.m
This file is responsible for converting multiple .pcap files to many individual ,pcd files which are used by the network for training.

### PointPillars_showFullVideo.m
This file reads in a folder full of .pcd files and runs them through a selected detector them displays the boxes associated with the detected labels.

### PointPillars_TrainDetector_usingPCD.m
This is the file responsible for training the neural network through the use of a folder made of many .pcd files and a seccond folder consisting of .mat label files. The output is a trained detector saved as a .mat file.

### PreprocessPcapFiles.m
When we recorded some of our data, there were outlier points that caused labeling to be very difficult. This file was created to preprocess the .pcap files and crop them to a selected range as .pcd files for labeling.

### ReadFromLidarAndDetect.m
This file is what is used to read the data directly from the lidar for running the detector with live data. the Do6Action variable can be toggled to select either the 6 action detector or the 3 action detector. bestBoxesOnly controlles if you want to show all boxes returned or only the most confident one.
