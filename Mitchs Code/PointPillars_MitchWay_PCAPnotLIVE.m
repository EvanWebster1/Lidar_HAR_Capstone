
canUseParallelPool = true;
trainTestSplit = 0.7;
doTraining = false;

doAugment = doTraining & true;
executionEnvironment = "auto";
confidenceThreshold = 0.5;

pretrainedDetectorName= 'detector.mat';
detectorOutputName= 'detector.mat';

outputFolderDatabase= 'C:\Users\mzinc\OneDrive\Desktop\OSS CAPSTONE\Database\';

%Load Data------------------------------------------------------------
%creates the full path to read from
tic;
path = fullfile(outputFolderDatabase,'my_Lidar');
lidarData = fileDatastore(path,'ReadFcn',@(x) pcread(x));

gtPath = fullfile(outputFolderDatabase,'my_Labels','Capture3_120_labels.mat');

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
disp("Finished loading labels")
toc

%Preprocess Data-----------------------------------------------------------
tic;
xMin = -70.0;     % Minimum value along X-axis.
yMin = -40.0;  % Minimum value along Y-axis.
zMin = -5.0;    % Minimum value along Z-axis.
xMax = 70.0;   % Maximum value along X-axis.
yMax = 40.0;   % Maximum value along Y-axis.
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
disp("Finnished loading and preprocessing lidar clouds")
toc

%Create Datastore Objects for Training-------------------------------------
tic;
rng("default");
shuffledIndices = randperm(size(processedLabels,1));
idx = floor(trainTestSplit * length(shuffledIndices));

trainData = croppedPointCloudObj(shuffledIndices(1:idx),:);
testData = croppedPointCloudObj(shuffledIndices(idx+1:end),:);

trainLabels = processedLabels(shuffledIndices(1:idx),:);
testLabels = processedLabels(shuffledIndices(idx+1:end),:);

writeFiles = true;
dataLocation = fullfile(outputFolderDatabase,'my_InputData');
[trainData,trainLabels] = saveptCldToPCD(trainData,trainLabels,...
    dataLocation,writeFiles);

lds = fileDatastore(dataLocation,'ReadFcn',@(x) pcread(x));

bds = boxLabelDatastore(trainLabels);

cds = combine(lds,bds);
disp("")
disp("Datastore creation and train/test split finnished")
toc
%Data Augmentation---------------------------------------------------------
if (doAugment)
    tic
    classNames = trainLabels.Properties.VariableNames;
    sampleLocation = fullfile(outputFolderDatabase,'my_GTsamples');
    [ldsSampled,bdsSampled] = sampleLidarData(cds,classNames,'MinPoints',20,...                  
                                'Verbose',false,'WriteLocation',sampleLocation);
    cdsSampled = combine(ldsSampled,bdsSampled);
    
    numObjects = 10;
    cdsAugmented = transform(cds,@(x)pcBboxOversample(x,cdsSampled,classNames,numObjects));
    
    cdsAugmented = transform(cdsAugmented,@(x)augmentData(x));
    disp("Data augmentation finished")
    toc
end

%Train/Create  Pointpillars Object Detector--------------------------------
if doTraining    
    
    % Define the number of prominent pillars.
    P = 12000; 
    % Define the number of points per pillar.
    N = 100;  
    
    anchorBoxes = calculateAnchorsPointPillars(trainLabels);
    classNames = trainLabels.Properties.VariableNames;
    
    detector = pointPillarsObjectDetector(pointCloudRange,classNames,anchorBoxes,...
        'VoxelSize',voxelSize,'NumPillars',P,'NumPointsPerPillar',N);
    
    if canUseParallelPool
        dispatchInBackground = true;
    else
        dispatchInBackground = false;
    end
    
    options = trainingOptions('adam',...
        'Plots','training-progress',...
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
    
    [detector,info] = trainPointPillarsObjectDetector(cdsAugmented,detector,options);
    
    outputFile = fullfile(outputFolderDatabase, "WalkStandCrouch_Detector_134_V2.mat");
    save(outputFile, "detector");
    disp("Detector finished training");
else
    pretrainedDetector = load('C:\Users\mzinc\OneDrive\Desktop\OSS CAPSTONE\Database\WalkStandCrouch_Detector_134.mat','detector');
    detector = pretrainedDetector.detector;
    disp("Pretrained detector loaded");
end


%Evaluate Detector Using Test Set------------------------------------------
tic;
numInputs = 50;

% Generate rotated rectangles from the cuboid labels.
bds = boxLabelDatastore(testLabels(1:numInputs,:));
groundTruthData = transform(bds,@(x)createRotRect(x));

% Set the threshold values.
nmsPositiveIoUThreshold = 0.5;

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

% Calculate the average value for each row/column
average_orientation_similarity = cellfun(@mean, metrics.OrientationSimilarity);
metrics.OrientationSimilarity = average_orientation_similarity;

average_precision = cellfun(@mean, metrics.Precision);
metrics.Precision = average_precision;

average_recall = cellfun(@mean, metrics.Recall);
metrics.Recall = average_recall;

disp(metrics(:,:))

disp("Program finished")
toc
%helper fuctions-----------------------------------------------------------

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