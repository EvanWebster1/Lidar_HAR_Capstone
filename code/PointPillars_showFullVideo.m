
%controlls:
doEvaluate = false;
doShowFullLidar = true;

tic;
% tic;
%         elapsedTime = toc;
% % Display the elapsed time
% fprintf('Elapsed Time: %.4f seconds\n', elapsedTime);
%

close all

outputFolder= 'C:\Users\mzinc\OneDrive\Documents\GitHub\Lidar_HAR_Capstone\Database\';

%Load Data------------------------------------------------------------

path = fullfile(outputFolder,'my_Lidar');

lidarData = fileDatastore(path,'ReadFcn',@(x) pcread(x));

%Preprocess Data-----------------------------------------------------------
disp("here")
i = 1;
while(hasdata(lidarData))
       
        processedPointCloud{i,1} = read(lidarData);
        roi = [-4.6 6.3 -1.15 9.55 -1 3];
        indices = findPointsInROI(processedPointCloud{i,1},roi);
        processedPointCloud{i,1} = select(processedPointCloud{i,1},indices);
    i = i + 1;

end
disp("here1")
reset(lidarData);

%Create Datastore Objects for Training-------------------------------------

testData = processedPointCloud;
%testLabels = processedLabels;

pretrainedDetector = load('C:\Users\mzinc\OneDrive\Documents\GitHub\Lidar_HAR_Capstone\Database\Trained Detectors\my_trained_detector_Smush2_40EP_5frame.mat','detector');
detector = pretrainedDetector.detector;
disp("here2")

%Generate Detections-------------------------------------------------------
if doShowFullLidar
    helperDisplay3DBoxesAndFullLidar2(testData, detector)
end
%Evaluate Detector Using Test Set------------------------------------------
if(doEvaluate)
    numInputs = size(testData)-1;
    
    % Generate rotated rectangles from the cuboid labels.
    bds = boxLabelDatastore(testLabels(1:numInputs,:));
    groundTruthData = transform(bds,@(x)createRotRect(x));
    
    % Set the threshold values.
    nmsPositiveIoUThreshold = 0.5;
    confidenceThreshold = 0.25;
    
    detectionResults = detect(detector,testData(1:numInputs,:),...
        'Threshold',confidenceThreshold);
    
    % Convert the bounding boxes to rotated rectangles format and calculate
    % the evaluation metrics.
    for i = 1:height(detectionResults)
        box = detectionResults.Boxes{i};
        detectionResults.Boxes{i} = box(:,[1,2,4,5,9]);
    end
    
    metrics = evaluateDetectionAOS(detectionResults,groundTruthData,...
        nmsPositiveIoUThreshold);
    disp(metrics(:,1:2))
end
elapsedTime = toc;
% Display the elapsed time
fprintf('Elapsed Time: %.4f seconds\n', elapsedTime);
%helper fuctions-----------------------------------------------------------



function helperDisplay3DBoxesAndFullLidar(pointCloudList,labelsList,humanColor,titleForFigure, detector)
    figure;
    title(titleForFigure);
    for i = 1:size(pointCloudList)
        ptCloud = pointCloudList{i,1};
        gtLabels = labelsList(i,:);

        % Specify the confidence threshold to use only detections with
        % confidence scores above this value.
        confidenceThreshold = 0.5;

        [box,score,labels] = detect(detector,ptCloud,'Threshold',confidenceThreshold);

        boxlabelsHuman = box(labels'=='Human',:);

        % Display the predictions on the point cloud.

        ax = pcshow(ptCloud);

        showShape('cuboid',boxlabelsHuman ,'Parent',ax,'Opacity',0.1,...
        'Color',humanColor,'LineWidth',0.5);

        zoom(ax,1.5);
        drawnow;
        pause(0.05);
        
    end
end



function helperDisplay3DBoxesAndFullLidar2(pointCloudList, detector)
doSmush = true;
preFrames = 5;
    xLim = [-4.6 6.3]
    yLim = [-1.15 9.55]
    zLim = [-1 3]


    if(doSmush)
        xlimits = [-6 6];
        ylimits = [-8 8];
        zlimits = [-2 2];
        player = pcplayer(xlimits,ylimits,zlimits);
    
        for i = preFrames+1:size(pointCloudList,1)
            currFrame = pointCloudList{i,1};
            for j = 1:preFrames
                disp(i-j)
                backFrame = pointCloudList{(i-j),1};
                currFrame = pccat([currFrame;backFrame]);
            end
            newPCArray{i-preFrames,1} = currFrame;
            view(player,currFrame)
    
        end
        pointCloudList = newPCArray;
   
    end



    player = pcplayer(xLim,yLim ,zLim);
    for i = 1:size(pointCloudList)
        ptCloud = pointCloudList{i,1};

        % Specify the confidence threshold to use only detections with
        % confidence scores above this value.
        confidenceThreshold = 0.25;
        [box,score,labels] = detect(detector,ptCloud,'Threshold',confidenceThreshold);

        disp(labels)
        % Display the predictions on the point cloud.
        
        view(player,ptCloud); 
        %score(score==max(score)) = []
        [M,bestIndex] = max(score);

        %[y bestIndex] = min(abs(score)-median(score));

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
            
            showShape('cuboid',bestBox,'Parent',player.Axes,'Opacity',0.1, ...
                'Color',labelColour,'LineWidth',0.5, "Label",bestGuess,"LabelOpacity",0.5);
            
        end
        if(false)
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
            showShape('cuboid',box,'Parent',player.Axes,'Opacity',0.1, ...
                    'Color',labelColour,'LineWidth',0.5, "Label",labels,"LabelOpacity",0.5);
            disp(score)
        end
    end

end

