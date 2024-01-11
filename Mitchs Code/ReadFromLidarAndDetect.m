clear lidar
lidar = velodynelidar('VLP16');

preview(lidar)
pause(10)
closePreview(lidar)

pretrainedDetector = load('C:\Users\mzinc\OneDrive\Desktop\OSS CAPSTONE\Pandaset\my_trained_detector.mat','detector');
detector = pretrainedDetector.detector;
disp("here")

start(lidar)

[frame,timestamp] = read(lidar,1);

lidarViewer = pcplayer(frame.XLimits,frame.YLimits,frame.ZLimits);


while isOpen(lidarViewer)
        [ptCloud,timestamp] = read(lidar,1);

        % Specify the confidence threshold to use only detections with
        % confidence scores above this value.
        confidenceThreshold = 0.10;
        [box,test,labels] = detect(detector,ptCloud,'Threshold',confidenceThreshold);
        disp(test)
        boxlabelsHuman = box(labels'=='Human',:);

        % Display the predictions on the point cloud.
        
        view(lidarViewer,ptCloud); 
        showShape('cuboid',boxlabelsHuman ,'Parent',lidarViewer.Axes,'Opacity',0.1,...
        'Color',"green",'LineWidth',0.5);
        
end 


stop(lidar)
clear lidar