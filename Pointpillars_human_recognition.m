doTraining = true;
veloReader = velodyneFileReader("Database\Data\capture6_120.pcap","VLP16");

outputFolder = fullfile(pwd, "Database");

gtPath = fullfile(outputFolder, 'Labels', 'Capture6_120_labels.mat');
% labels = load("Database\Labels\Capture3_120_labels.mat");
labelData = load(gtPath);
labels = timetable2table(labelData.gTruth.LabelData);
boxLabels = labels(:,"Human");

xlimits = [0 10];
ylimits = [-6 6];
zlimits = [-2 3];
xStep = 0.10;
yStep = 0.10;
pointcldRange = [xlimits(1) xlimits(2) ylimits(1) ylimits(2) zlimits(1) zlimits(2)];
voxelSize = [xStep yStep];


% Preprocessing the data with the labes
[trainData, testData, trainLabels, testLabels, dataLocation] = preProcessData(boxLabels, ...
    veloReader, outputFolder);

lds = fileDatastore(dataLocation, 'ReadFcn', @(x) pcread(x));
bds = boxLabelDatastore(trainLabels);
cds = combine(lds, bds);

% testing current state of the data augmentation----------
augData = read(cds);
augData = read(cds);
augptCld = augData{1,1};
augLabels = augData{1,2};
augClass = augData{1,3};

labelsHuman = augLabels(augClass=='Human',:);

helperDisplay3DBoxesOverlaidPointCloud(augptCld.Location,labelsHuman,'green',...
    'Before Data Augmentation');

reset(cds);

classNames = {'Human'};
sampleLocation = fullfile(outputFolder, "Samples");
[ldsSampled,bdsSampled] = sampleLidarData(cds,classNames,'MinPoints',20,...                  
                            'Verbose',false,'WriteLocation',sampleLocation);
cdsSampled = combine(ldsSampled,bdsSampled);

numObjects = 5;
cdsAugmented = transform(cds,@(x)pcBboxOversample(x,cdsSampled,classNames,numObjects));

cdsAugmented = transform(cdsAugmented,@(x)augmentData(x));

augData = read(cdsAugmented);
augptCld = augData{1,1};
augLabels = augData{1,2};
augClass = augData{1,3};

labelsHuman = augLabels(augClass=='Human',:);

helperDisplay3DBoxesOverlaidPointCloud(augptCld.Location,labelsHuman,'green',...
    'After Data Augmentation');

reset(cdsAugmented);

%Create PointPillars Object Detector---------------------------------------
% Define the number of prominent pillars.
P = 12000; 

% Define the number of points per pillar.
N = 100;  

anchorBoxes = calculateAnchorsPointPillars(trainLabels);
classNames = trainLabels.Properties.VariableNames;

detector = pointPillarsObjectDetector(pointcldRange,classNames,anchorBoxes,...
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
else
    pretrainedDetector = load('pretrainedPointPillarsDetector.mat','detector');
    detector = pretrainedDetector.detector;
end

%Generate Detections-------------------------------------------------------
ptCloud = testData{45,1};
gtLabels = testLabels(45,:);

% Specify the confidence threshold to use only detections with
% confidence scores above this value.
confidenceThreshold = 0.5;
[box,score,labels] = detect(detector,ptCloud,'Threshold',confidenceThreshold);

boxlabelsCar = box(labels'=='Car',:);
boxlabelsTruck = box(labels'=='Truck',:);

% Display the predictions on the point cloud.
helperDisplay3DBoxesOverlaidPointCloud(ptCloud.Location,boxlabelsCar,'green',...
    boxlabelsTruck,'magenta','Predicted Bounding Boxes');

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

function helperDisplay3DBoxesOverlaidPointCloud(ptCld,label,labelcolor,...
    titleForFigure)
% Display the point cloud with different colored bounding boxes for different
% classes.
    figure;
    ax = pcshow(ptCld);
    showShape('cuboid',label,'Parent',ax,'Opacity',0.1,...
        'Color',labelcolor,'LineWidth',0.5);
    title(titleForFigure);
    zoom(ax,1.5);
end