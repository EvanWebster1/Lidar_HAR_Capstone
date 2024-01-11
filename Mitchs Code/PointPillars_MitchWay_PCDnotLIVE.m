%%----
%data=load('C:\Users\mzinc\OneDrive\Desktop\OSS CAPSTONE\Pandaset\my_Labels\Capture3_120_labels.mat')
%disp(data.gTruth.LabelData);
%%
doTraining = false;
canUseParallelPool = false;

%outputFolder = fullfile('C:\Users\mzinc\OneDrive\Desktop\OSS CAPSTONE','Pandaset');

%lidarURL = ['https://ssd.mathworks.com/supportfiles/lidar/data/' ...
    %'Pandaset_LidarData.tar.gz'];
%helperDownloadPandasetData(outputFolder,lidarURL);

outputFolder= 'C:\Users\mzinc\OneDrive\Desktop\OSS CAPSTONE\Pandaset';

%Load Data------------------------------------------------------------
%creates the full path to read from
path = fullfile(outputFolder,'my_Lidar');
%lidarData is a datastore object that can be read, each time it is read it
%moves to the next files
lidarData = fileDatastore(path,'ReadFcn',@(x) pcread(x));

%create path for ground truths
gtPath = fullfile(outputFolder,'my_Labels','Capture3_120_labels.mat');
%load data from path
%data is a timetable object(first column is the date/time)
data = load(gtPath,'gTruth');
%convert timetable to a table we can actualy use
Labels = timetable2table(data.gTruth.LabelData);
%extract just the labels, i think that it a list of lists that contain 9
%points each that represent the bounding box of the cuboid
boxLabels = Labels(:,2)

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
xMin = 0.0;     % Minimum value along X-axis.
yMin = -39.68;  % Minimum value along Y-axis.
zMin = -5.0;    % Minimum value along Z-axis.
xMax = 69.12;   % Maximum value along X-axis.
yMax = 39.68;   % Maximum value along Y-axis.
zMax = 5.0;     % Maximum value along Z-axis.
xStep = 0.16;   % Resolution along X-axis.
yStep = 0.16;   % Resolution along Y-axis.
dsFactor = 2.0; % Downsampling factor.

% Calculate the dimensions for the pseudo-image.
Xn = round(((xMax - xMin)/xStep));
Yn = round(((yMax - yMin)/yStep));

% Define point cloud parameters.
pointCloudRange = [xMin xMax yMin yMax zMin zMax];
voxelSize = [xStep yStep];

[croppedPointCloudObj,processedLabels] = cropFrontViewFromLidarData(lidarData,boxLabels,pointCloudRange);

pc = croppedPointCloudObj{100,1};
gtLabelsHuman = processedLabels.Human{100};
%gtLabelsTruck = processedLabels.Truck{1};

helperDisplay3DBoxesOverlaidHuman(pc.Location,gtLabelsHuman,'green','Cropped Point Cloud');

reset(lidarData);
%Create Datastore Objects for Training-------------------------------------

rng(1);
shuffledIndices = randperm(size(processedLabels,1));
idx = floor(0.7 * length(shuffledIndices));

trainData = croppedPointCloudObj(shuffledIndices(1:idx),:);
testData = croppedPointCloudObj(shuffledIndices(idx+1:end),:);

trainLabels = processedLabels(shuffledIndices(1:idx),:);
testLabels = processedLabels(shuffledIndices(idx+1:end),:);

writeFiles = true;
dataLocation = fullfile(outputFolder,'my_InputData');
[trainData,trainLabels] = saveptCldToPCD(trainData,trainLabels,...
    dataLocation,writeFiles);

lds = fileDatastore(dataLocation,'ReadFcn',@(x) pcread(x));

bds = boxLabelDatastore(trainLabels);

cds = combine(lds,bds);

%Data Augmentation---------------------------------------------------------
augData = read(cds);
augptCld = augData{1,1};
augLabels = augData{1,2};
augClass = augData{1,3};

labelsHuman = augLabels(augClass=='Human',:);
%labelsTruck = augLabels(augClass=='Truck',:);

helperDisplay3DBoxesOverlaidHuman(augptCld.Location,gtLabelsHuman,'green','Before Data Augmentation');

reset(cds);

classNames = {'Human'};
sampleLocation = fullfile(outputFolder,'my_GTsamples');
[ldsSampled,bdsSampled] = sampleLidarData(cds,classNames,'MinPoints',20,...                  
                            'Verbose',false,'WriteLocation',sampleLocation);
cdsSampled = combine(ldsSampled,bdsSampled);

numObjects = 4;
cdsAugmented = transform(cds,@(x)pcBboxOversample(x,cdsSampled,classNames,numObjects));

cdsAugmented = transform(cdsAugmented,@(x)augmentData(x));

augData = read(cdsAugmented);
augptCld = augData{1,1};
augLabels = augData{1,2};
augClass = augData{1,3};

labelsHuman = augLabels(augClass=='Human',:);

helperDisplay3DBoxesOverlaidHuman(augptCld.Location,labelsHuman,'green','After Data Augmentation');

reset(cdsAugmented);

%Create PointPillars Object Detector---------------------------------------
% Define the number of prominent pillars.
P = 12000; 

% Define the number of points per pillar.
N = 100;  

anchorBoxes = calculateAnchorsPointPillars(trainLabels);
classNames = trainLabels.Properties.VariableNames;

detector = pointPillarsObjectDetector(pointCloudRange,classNames,anchorBoxes,...
    'VoxelSize',voxelSize,'NumPillars',P,'NumPointsPerPillar',N);

%Train Pointpillars Object Detector----------------------------------------
executionEnvironment = "auto";
if canUseParallelPool
    dispatchInBackground = true;
else
    dispatchInBackground = false;
end

options = trainingOptions('adam',...
    'Plots',"none",...
    'MaxEpochs',60,...
    'MiniBatchSize',3,...
    'GradientDecayFactor',0.9,...
    'SquaredGradientDecayFactor',0.999,...
    'LearnRateSchedule',"piecewise",...
    'InitialLearnRate',0.0002,...
    'LearnRateDropPeriod',15,...
    'LearnRateDropFactor',0.8,...
    'ExecutionEnvironment',executionEnvironment,...
    'DispatchInBackground',dispatchInBackground,...
    'BatchNormalizationStatistics','moving',...
    'ResetInputNormalization',false,...
    'CheckpointPath',tempdir);

if doTraining    
    [detector,info] = trainPointPillarsObjectDetector(cdsAugmented,detector,options);

    outputFile = fullfile(outputFolder, "my_trained_detector.mat");
    save(outputFile, "detector");
else
    pretrainedDetector = load('C:\Users\mzinc\OneDrive\Desktop\OSS CAPSTONE\Pandaset\my_trained_detector.mat','detector');
    detector = pretrainedDetector.detector;
end



%Generate Detections-------------------------------------------------------
ptCloud = testData{45,1};
gtLabels = testLabels(45,:);

% Specify the confidence threshold to use only detections with
% confidence scores above this value.
confidenceThreshold = 0.5;
[box,score,labels] = detect(detector,ptCloud,'Threshold',confidenceThreshold);

boxlabelsHuman = box(labels'=='Human',:);

% Display the predictions on the point cloud.

helperDisplay3DBoxesOverlaidHuman(ptCloud.Location,boxlabelsHuman,'green','After Data Augmentation');

%Evaluate Detector Using Test Set------------------------------------------
numInputs = 50;

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


%helper fuctions-----------------------------------------------------------
disp("done")
function helperDownloadPandasetData(outputFolder,lidarURL)
% Download the data set from the given URL to the output folder.

    lidarDataTarFile = fullfile(outputFolder,'Pandaset_LidarData.tar.gz');
    
    if ~exist(lidarDataTarFile,'file')
        mkdir(outputFolder);
        
        disp('Downloading PandaSet Lidar driving data (5.2 GB)...');
        websave(lidarDataTarFile,lidarURL);
        untar(lidarDataTarFile,outputFolder);
    end
    
    % Extract the file.
    if (~exist(fullfile(outputFolder,'Lidar'),'dir'))...
            &&(~exist(fullfile(outputFolder,'Cuboids'),'dir'))
        untar(lidarDataTarFile,outputFolder);
    end

end

function helperDisplay3DBoxesOverlaidPointCloud(ptCld,labelsCar,carColor,...
    labelsTruck,truckColor,titleForFigure)
% Display the point cloud with different colored bounding boxes for different
% classes.
    figure;
    ax = pcshow(ptCld);
    showShape('cuboid',labelsCar,'Parent',ax,'Opacity',0.1,...
        'Color',carColor,'LineWidth',0.5);
    hold on;
    showShape('cuboid',labelsTruck,'Parent',ax,'Opacity',0.1,...
        'Color',truckColor,'LineWidth',0.5);
    title(titleForFigure);
    zoom(ax,1.5);
end

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