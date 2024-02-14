clear lidar
lidar = velodynelidar('VLP16');

preview(lidar)
pause(2)
closePreview(lidar)

pretrainedDetector = load('C:\Users\mzinc\OneDrive\Desktop\OSS CAPSTONE\Database\my_trained_detector.mat','detector');
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

        % Display the predictions on the point cloud.
        
        view(lidarViewer,ptCloud); 
        for j = 1:size(labels,1)
            boxlabels = box(labels'==labels(j),:);
            switch labels(j)
                case "Walking"
                    labelColour = "green";
                case "Standing"
                    labelColour = "yellow";
                case "Crouching"
                    labelColour = "red";
                otherwise
                    labelColour = "purple";
            end

            showShape('cuboid',boxlabels,'Parent',player.Axes,'Opacity',0.1, ...
                'Color',labelColour,'LineWidth',0.5, "Label",labels(j),"LabelOpacity",0.5);
        end
        
end 


stop(lidar)
clear lidar