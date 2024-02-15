%%----
%data=load('C:\Users\mzinc\OneDrive\Desktop\OSS CAPSTONE\Pandaset\my_Labels\Capture3_120_labels.mat')
%disp(data.gTruth.LabelData);
%%
doTraining = true;
canUseParallelPool = false;

outputFolder= 'C:\Users\mzinc\OneDrive\Desktop\OSS CAPSTONE\Pandaset';
outputFolderDatabase= 'C:\Users\mzinc\OneDrive\Desktop\OSS CAPSTONE\Database\';

%Load Data------------------------------------------------------------
%creates the full path to read from
path = fullfile(outputFolderDatabase,'my_Lidar');
%lidarData is a datastore object that can be read, each time it is read it
%moves to the next files
lidarData = fileDatastore(path,'ReadFcn',@(x) pcread(x));

labelPath = fullfile(outputFolderDatabase,'Labels\');
labelData = dir(labelPath);
boxLabels = {};
for i = 3:size(labelData)
    labelFilePath = append(labelPath,labelData(i).name);
    data = load(labelFilePath,'gTruth');

    Labels = timetable2table(data.gTruth.LabelData);
    newBoxLabels = Labels(:,2:end);
    boxLabels = [boxLabels;newBoxLabels];
end




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

numFrames = size(boxLabels,1);
processedPointCloud = cell(numFrames, 1);
processedLabels = boxLabels;
for i = 1:numFrames

    ptCloud = read(lidarData);            
    processedData = removeInvalidPoints(ptCloud);
    processedPointCloud{i,1} = processedData;
end
%Create Datastore Objects for Training-------------------------------------

rng(1);
shuffledIndices = randperm(size(processedLabels,1));
idx = floor(0.7 * length(shuffledIndices));

trainData = processedPointCloud(shuffledIndices(1:idx),:);
testData = processedPointCloud(shuffledIndices(idx+1:end),:);

trainLabels = processedLabels(shuffledIndices(1:idx),:);
testLabels = processedLabels(shuffledIndices(idx+1:end),:);

writeFiles = true;
dataLocation = fullfile(outputFolderDatabase,'my_InputData');
[trainData,trainLabels] = saveptCldToPCD(trainData,trainLabels,...
    dataLocation,writeFiles);

lds = fileDatastore(dataLocation,'ReadFcn',@(x) pcread(x));

bds = boxLabelDatastore(trainLabels);

cds = combine(lds,bds);

%Data Augmentation---------------------------------------------------------

classNames = trainLabels.Properties.VariableNames;
sampleLocation = fullfile(outputFolderDatabase,'my_GTsamples');
[ldsSampled,bdsSampled] = sampleLidarData(cds,classNames,'MinPoints',20,...                  
                            'Verbose',false,'WriteLocation',sampleLocation);
cdsSampled = combine(ldsSampled,bdsSampled);

numObjects = 4;
cdsAugmented = transform(cds,@(x)pcBboxOversample(x,cdsSampled,classNames,numObjects));

cdsAugmented = transform(cdsAugmented,@(x)augmentData(x));


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

    outputFile = fullfile(outputFolderDatabase, "my_trained_detector_2+3.mat");
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