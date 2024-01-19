
%controlls:
doEvaluate = false;
doShowFullLidar = true
tic;
% tic;
%         elapsedTime = toc;
% % Display the elapsed time
% fprintf('Elapsed Time: %.4f seconds\n', elapsedTime);
%

close all

%outputFolder = fullfile('C:\Users\mzinc\OneDrive\Desktop\OSS CAPSTONE','Pandaset');

%lidarURL = ['https://ssd.mathworks.com/supportfiles/lidar/data/' ...
    %'Pandaset_LidarData.tar.gz'];
%helperDownloadPandasetData(outputFolder,lidarURL);

outputFolder= 'C:\Users\mzinc\OneDrive\Desktop\OSS CAPSTONE\Database\';

%Load Data------------------------------------------------------------
%creates the full path to read from
path = fullfile(outputFolder,'my_Lidar');
%lidarData is a datastore object that can be read, each time it is read it
%moves to the next files
lidarData = fileDatastore(path,'ReadFcn',@(x) pcread(x));

figure
%read the datastore object which also advances the next read to the next
%one
ptCld = read(lidarData);
%pointcloud.location is the locations of all points
%https://www.mathworks.com/help/vision/ref/pointcloud.html
ax = pcshow(ptCld.Location);
%set the sizing for the 3d view
%set zoom amount
zoom(ax,1.5);
axis off;

%reset let lidat datastore to be back at the first file
reset(lidarData);



%Preprocess Data-----------------------------------------------------------
i = 1;
while(hasdata(lidarData))

    ptCloud = read(lidarData);            
    processedData = removeInvalidPoints(ptCloud);
    processedPointCloud{i,1} = processedData;
    i = i + 1;

end

reset(lidarData);

%Create Datastore Objects for Training-------------------------------------

testData = processedPointCloud;
testLabels = processedLabels;

%Data Augmentation---------------------------------------------------------

%Create PointPillars Object Detector---------------------------------------

%Train Pointpillars Object Detector----------------------------------------





pretrainedDetector = load('C:\Users\mzinc\OneDrive\Desktop\OSS CAPSTONE\Database\my_trained_detector_12356.mat','detector');
detector = pretrainedDetector.detector;
disp("here")

%Generate Detections-------------------------------------------------------
if doShowFullLidar
    helperDisplay3DBoxesAndFullLidar2(testData, "green","title", detector)
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

function helperDisplay3DBoxesOverlaidHuman(ptCld,labelsHuman,humanColor,titleForFigure)
% Display the point cloud with different colored bounding boxes for different
% classes.
    figure;
    ax = pcshow(ptCld);
    showShape('cuboid',labelsHuman ,'Parent',ax,'Opacity',0.1,...
        'Color',humanColor,'LineWidth',0.5);
    title(titleForFigure);
    zoom(ax,1.5);
end

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



function helperDisplay3DBoxesAndFullLidar2(pointCloudList,humanColor,titleForFigure, detector)

    player = pcplayer([-2 10],[-5 5],[-2 2]);
    for i = 1:size(pointCloudList)
        ptCloud = pointCloudList{i,1};

        % Specify the confidence threshold to use only detections with
        % confidence scores above this value.
        confidenceThreshold = 0.50;
        [box,~,labels] = detect(detector,ptCloud,'Threshold',confidenceThreshold);

        boxlabelsHuman = box(labels'=='Human',:);

        % Display the predictions on the point cloud.
        
        view(player,ptCloud); 
        showShape('cuboid',boxlabelsHuman ,'Parent',player.Axes,'Opacity',0.1,...
        'Color',humanColor,'LineWidth',0.5);

    end

end

