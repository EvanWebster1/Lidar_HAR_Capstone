%%----
%data=load('C:\Users\mzinc\OneDrive\Desktop\OSS CAPSTONE\Pandaset\my_Labels\Capture3_120_labels.mat')
%disp(data.gTruth.LabelData);
%%
doTraining = true;
canUseParallelPool = false;
dataIsReady = false;

outputFolder= 'C:\Users\mzinc\OneDrive\Desktop\OSS CAPSTONE\Pandaset';
outputFolderDatabase= 'C:\Users\mzinc\OneDrive\Documents\GitHub\Lidar_HAR_Capstone\Database\';

%if(~dataIsReady)
%Load Data------------------------------------------------------------
disp("Load Data")
path = fullfile(outputFolderDatabase,'my_Lidar\');


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

%Remove Empty Columns/Lables---------------------------------
disp("Remove Empty Columns/Lables")
%[boxLabels, removedColumns] = removeZeroColumns(boxLabels)

classNames = boxLabels.Properties.VariableNames;
%commonIndices = ismember(classNames, removedColumns);
%classNames(commonIndices) = [];

labelCounts = [];
for i = 1:width(boxLabels)
    count = 0;
    for j = 1:height(boxLabels)
        if(isstruct(cell2table(boxLabels{j,i}).Var1))
            boxLabels{j,i} = {cell2table(boxLabels{j,i}).Var1.Position};
        end
        if(sum(cell2mat(boxLabels{j,i})) ~= 0)
            count = count + 1;
        end
    end
    labelCounts(i) = count;
end
% Display the results

activityTable = table(classNames', labelCounts', (labelCounts/height(boxLabels))', 'VariableNames', {'Activity', 'Count', 'Percent of Total'});
disp(activityTable)
%Preprocess Data-----------------------------------------------------------
disp("Preprocess Data")
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

processedPointCloud = {};
processedLabels = boxLabels;

lidarDataFiles = dir(path);
for i = 3:size(lidarDataFiles)
    
    lidarFilePath = append(path,lidarDataFiles(i).name);

    lidarData = load(lidarFilePath,"processedPointCloud");
    disp(lidarData.processedPointCloud{1})
    processedPointCloud = vertcat(processedPointCloud,lidarData.processedPointCloud);

end
%Create Datastore Objects for Training-------------------------------------
disp("Create Datastore Objects for Training")
rng(1);
shuffledIndices = randperm(size(processedLabels,1));
idx = floor(0.7 * length(shuffledIndices));

trainData = processedPointCloud(shuffledIndices(1:idx),:);
testData = processedPointCloud(shuffledIndices(idx+1:end),:);
trainLabels = processedLabels(shuffledIndices(1:idx),:);
testLabels = processedLabels(shuffledIndices(idx+1:end),:);


dataLocation = fullfile(outputFolderDatabase,'my_InputData');

if(~dataIsReady)
    writeFiles = true;
    [trainData,trainLabels] = saveptCldToPCD(trainData,trainLabels,...
        dataLocation,writeFiles);
end
lds = fileDatastore(dataLocation,'ReadFcn',@(x) pcread(x));
bds = boxLabelDatastore(trainLabels);
cds = combine(lds,bds);

%Data Augmentation---------------------------------------------------------
disp("Data Augmentation")
cdsAugmented = cds;
 
%Create PointPillars Object Detector---------------------------------------
disp("Create PointPillars Object Detector")
% Define the number of prominent pillars.
P = 12000; 

% Define the number of points per pillar.
N = 100;  

anchorBoxes = calculateAnchorsPointPillars(trainLabels);

detector = pointPillarsObjectDetector(pointCloudRange,classNames,anchorBoxes,...
    'VoxelSize',voxelSize,'NumPillars',P,'NumPointsPerPillar',N);

%Train Pointpillars Object Detector----------------------------------------
disp("Train Pointpillars Object Detector")
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

    outputFile = fullfile(outputFolderDatabase, "test.mat");
    save(outputFile, "detector");
else
    pretrainedDetector = load('C:\Users\mzinc\OneDrive\Documents\GitHub\Lidar_HAR_Capstone\my_trained_detector.mat','detector');
    detector = pretrainedDetector.detector;
end




%Evaluate Detector Using Test Set------------------------------------------
disp("Evaluate Detector Using Test Set")
numInputs = 100;

% Generate rotated rectangles from the cuboid labels.
bds = boxLabelDatastore(testLabels(1:numInputs,:));
groundTruthData = transform(bds,@(x)createRotRect(x));
disp(bds(:,2))
% Set the threshold values.
nmsPositiveIoUThreshold = 0.5;
confidenceThreshold = 0.5;

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

function [newTable, removedColumns] = removeZeroColumns(inputTable)
    % Initialize arrays to store the new table and removed column names
    newTable = inputTable;
    removedColumns = {};

    % Iterate through each variable (column) of the input table
    for col = 1:width(inputTable)
        % Check if all elements in the current column are empty or 0x0 doubles
        if all(cellfun('isempty', inputTable{:, col}) | cellfun(@(x) all(x == 0), inputTable{:, col}))
            % Record the name of the removed column
            removedColumns = [removedColumns, inputTable.Properties.VariableNames{col}];
            % Remove the column from the new table
            newTable(:, col) = [];
        end
    end
end