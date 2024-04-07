clear lidar
lidar = velodynelidar('VLP16');
outputFolderDatabase = "C:\Users\mzinc\OneDrive\Documents\GitHub\Lidar_HAR_Capstone\Database\Trained Detectors\"


Do6Action = false;
bestBoxesOnly = false;
confidenceThreshold = 0.5;

 %walk stand squat thresh: 0.3
%6thresh: 0.1
%6: detector_Smush0_bigguy40
%6smush: my_trained_detector_Smush2_40EP_5frame.mat
%3: WalkStandCrouch_Detector_V2

if (Do6Action)
    outputFile = fullfile(outputFolderDatabase, "final_detector_Smush2_100EP_5frame.mat");
    doSmush = true;
    preframes = 5;
    preLoad = true

else
    outputFile = fullfile(outputFolderDatabase, "WalkStandCrouch_Detector_V2.mat");
    doSmush = false;
    preframes = 0;
    preLoad = false

end

pretrainedDetector = load(outputFile,'detector');
detector = pretrainedDetector.detector;

start(lidar)

[frame,timestamp] = read(lidar,1);


xLim = [-4.6 6.3]
yLim = [-1.15 9.55]
zLim = [-1 3]
lidarViewer = pcplayer(xLim,yLim,zLim);

smushArray = cell(1, preframes);

% Populate the array with empty point clouds

while isOpen(lidarViewer)
    if(lidar.NumPointCloudsAvailable>preframes+1 || ~preLoad)
        if (preLoad)

            [ptCloud,timestamp] = read(lidar,lidar.NumPointCloudsAvailable);

            numPoints = size(ptCloud, 1);
    
            lastPoints = ptCloud(numPoints-preframes:end, :);
            currFrame = lastPoints(6);
            showFrame = lastPoints(6);
            for j = 1:preframes
                smushArray{j} = lastPoints(j);
                currFrame = pccat([currFrame;lastPoints(j)]);
            end
            lastPoint = currFrame
            view(lidarViewer,showFrame);
            preLoad = false;

        else
            [ptCloud,timestamp] = read(lidar,lidar.NumPointCloudsAvailable);
    
            numPoints = size(ptCloud, 1);
    
            lastPoint = ptCloud(numPoints:end, :);
            if (doSmush)
                currFrame = lastPoint;
                backframes = [];
                for j = 1:preframes
                    if j > 1
                        backFrames(j-1) = smushArray(j);
                    end
                    
                    currFrame = pccat([currFrame;smushArray{j}]);
                end
                smushArray = backFrames;
                smushArray{5} = lastPoint;
                lastPoint = currFrame;
                view(lidarViewer,lastPoint);
            else
                lastPoint = ptCloud(numPoints:end, :);
                view(lidarViewer,lastPoint);
            end
        end


        %Do the detection
        [box,score,labels] = detect(detector,lastPoint,'Threshold',confidenceThreshold);
        disp(score)
    
        % Display the predictions on the point cloud
        if(bestBoxesOnly)
            [M,bestIndex] = max(score);
            if(~isempty(bestIndex))
                guessTable = labels;
                bestGuess = guessTable(bestIndex);
                bestBox = box(bestIndex,:);
    
                switch bestGuess
                    case "Walking"
                        labelColour = "red";
                    case "Standing"
                        labelColour = "blue";
                    case "Squatting"
                        labelColour = "cyan";
                    case "Jumping_Jacks"
                        labelColour = "green";
                    case "Push_Ups"
                        labelColour = "magenta";
                    case "Sit_Ups"
                        labelColour = "yellow";
        
                    otherwise
                        labelColour = "White";
                end
                
                showShape('cuboid',bestBox,'Parent',lidarViewer.Axes,'Opacity',0.1, ...
                    'Color',labelColour,'LineWidth',0.5, "Label",bestGuess,"LabelOpacity",0.5);
            end
        else
            labelColour = [""];
            for j = 1:size(box,1)
    
                boxlabels = box(j,:);
                switch labels(j)
                    case "Walking"
                        labelColour(j) = "red";
                    case "Standing"
                        labelColour(j) = "blue";
                    case "Squatting"
                        labelColour(j) = "cyan";
                    case "Jumping_Jacks"
                        labelColour(j) = "green";
                    case "Push_Ups"
                        labelColour(j) = "magenta";
                    case "Sit_Ups"
                        labelColour(j) = "yellow";
    
                    otherwise
                        labelColour(j) = "White";
                end
    
            end
            showShape('cuboid',box,'Parent',lidarViewer.Axes,'Opacity',0.1, ...
                    'Color',labelColour,'LineWidth',0.5, "Label",labels,"LabelOpacity",0.5);
            disp(score)
        end
    end
end



stop(lidar)
clear lidar