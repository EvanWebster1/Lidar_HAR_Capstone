%%----
%data=load('C:\Users\mzinc\OneDrive\Desktop\OSS CAPSTONE\Pandaset\my_Labels\Capture3_120_labels.mat')
%disp(data.gTruth.LabelData);
%%
doTraining = false;
canUseParallelPool = false;
dataIsReady = false;
doSmush = false;
doSmushV2 = true;
doSmushV3 = false;
doSmushV4 = false;
preFrames = 5
outputFolder= 'C:\Users\mzinc\OneDrive\Documents\GitHub\Lidar_HAR_Capstone\';
outputFolderDatabase= 'C:\Users\mzinc\OneDrive\Documents\GitHub\Lidar_HAR_Capstone\Database\';

%if(~dataIsReady)
%Load Data------------------------------------------------------------
disp("Load Data")
path = fullfile(outputFolderDatabase,'my_Lidar');
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
yMin = -19.84;  % Minimum value along Y-axis.
zMin = -2.5;    % Minimum value along Z-axis.
xMax = 34.56;   % Maximum value along X-axis.
yMax = 19.84;   % Maximum value along Y-axis.
zMax = 2.5;     % Maximum value along Z-axis.
xStep = 0.16;   % Resolution along X-axis.
yStep = 0.16;   % Resolution along Y-axis.
dsFactor = 2.0; % Downsampling factor.

% Calculate the dimensions for the pseudo-image.
Xn = round(((xMax - xMin)/xStep));
Yn = round(((yMax - yMin)/yStep));

% Define point cloud parameters.
pointCloudRange = [xMin xMax yMin yMax zMin zMax];
voxelSize = [xStep yStep];

outputFileName = "preProcLidarData.mat"
processedLabels = boxLabels;
doReadIn = true; 
if isfile(outputFileName)
     % File exists.
    lidarFilePath = append(outputFolder ,outputFileName);
    lidar = load(lidarFilePath,"processedPointCloud");
    processedPointCloud = lidar.processedPointCloud;
    if(size(processedPointCloud,1) == size(boxLabels,1))
        doReadIn = false;
    end
end
if (doReadIn)
    numFrames = size(boxLabels,1);
    processedPointCloud = cell(numFrames, 1);
    
    for i = 1:numFrames
        
        processedPointCloud{i,1} = read(lidarData);
        roi = [-4.6 6.3 -1.15 9.55 -1 3];
        indices = findPointsInROI(processedPointCloud{i,1},roi);
        processedPointCloud{i,1} = select(processedPointCloud{i,1},indices);
    end
    
    save(outputFileName, 'processedPointCloud','-v7.3');
end

numFrames = size(boxLabels,1);
%smush pointClouds
if(doSmush)
    xlimits = [-6 6];
    ylimits = [-8 8];
    zlimits = [-2 2];
    player = pcplayer(xlimits,ylimits,zlimits);

    for i = preFrames+1:numFrames
        currFrame = processedPointCloud{i,1};
        for j = 1:preFrames
            disp(i-j)
            backFrame = processedPointCloud{(i-j),1};
            currFrame = pccat([currFrame;backFrame]);
        end
        newPCArray{i-preFrames,1} = currFrame;
        view(player,currFrame)

    end
    processedPointCloud = newPCArray;
    processedLabels = processedLabels(preFrames+1:end, :);

end


if(doSmushV2)
    xlimits = [-6 6];
    ylimits = [-8 8];
    zlimits = [-2 2];
    player = pcplayer(xlimits,ylimits,zlimits);
    for i = 1:numFrames
        

        for j = 1:size(processedLabels,2)
            index = 1;
            if( isempty(cell2mat(processedLabels{i,j})))
                continue
            end
            %cell2mat(processedLabels{i,1})
            %disp(cell2mat(processedLabels{i,1}))
            %disp(processedPointCloud{i,1})
            
            test = processedPointCloud{i,1};
            params = cell2mat(processedLabels{i,j});
            model = cuboidModel(params);
    
            indices = findPointsInsideCuboid(model, test);
            cubPtCloud = select(test,indices);
            view(player,cubPtCloud)

            roiPoints{i,index} = cubPtCloud;

            index = index + 1;
        end
    end

    for i = preFrames+1:numFrames
        currFrame = processedPointCloud{i,1};
        if(~isempty(roiPoints{i,1}))
            for j = 1:preFrames
                for k = 1:size(roiPoints{preFrames+1-j,:}, 2)
                    backFrame = roiPoints{i-j,k};
                    currFrame = pccat([currFrame;backFrame]);
                end
            end
        newPCArray{i-preFrames,1} = currFrame;
        view(player,currFrame)
        end
        
    end
    processedPointCloud = newPCArray;
    processedLabels = processedLabels(preFrames+1:end, :);
    
end

if(doSmushV3)
    xlimits = [-6 6];
    ylimits = [-8 8];
    zlimits = [-2 2];
    player = pcplayer(xlimits,ylimits,zlimits);
    for i = 1:numFrames
        

        for j = 1:size(processedLabels,2)
            index = 1;
            if( isempty(cell2mat(processedLabels{i,j})))
                continue
            end
            %cell2mat(processedLabels{i,1})
            %disp(cell2mat(processedLabels{i,1}))
            %disp(processedPointCloud{i,1})
            
            test = processedPointCloud{i,1};
            params = cell2mat(processedLabels{i,j});
            model = cuboidModel(params);
    
            indices = findPointsInsideCuboid(model, test);
            cubPtCloud = select(test,indices);
            %view(player,cubPtCloud)

            roiPoints{i,index} = cubPtCloud;

            index = index + 1;
        end
    end

    
    
    for i = (preFrames+1):numFrames
        testThing = [];
        currFrame = processedPointCloud{i,1};
        if(~isempty(roiPoints{i,1}))
            non_empty_cells = cellfun(@(x) ~isempty(x), processedLabels{i,:});
            targetLabel = find(non_empty_cells);

            for j = 1:preFrames
                for k = 1:size(roiPoints{preFrames+1-j,:}, 2)

                    non_empty_cells = cellfun(@(x) ~isempty(x), processedLabels{i-j,:});
                    prevFrameLabel = find(non_empty_cells);


                    if (targetLabel == prevFrameLabel)
                        testThing(j) = prevFrameLabel;
                        backFrame = roiPoints{i-j,k};
                        currFrame = pccat([currFrame;backFrame]);
                    end
                end
            end
        %disp(testThing)
        newPCArray{i-preFrames,1} = currFrame;
        %view(player,currFrame)
        end
        
    end
    processedPointCloud = newPCArray;
    processedLabels = processedLabels(preFrames+1:end, :);

end

if(doSmushV4)
    xlimits = [-6 6];
    ylimits = [-8 8];
    zlimits = [-2 2];
    %player = pcplayer(xlimits,ylimits,zlimits);
    for i = 1:numFrames
        

        for j = 1:size(processedLabels,2)
            index = 1;
            if( isempty(cell2mat(processedLabels{i,j})))
                continue
            end
            %cell2mat(processedLabels{i,1})
            %disp(cell2mat(processedLabels{i,1}))
            %disp(processedPointCloud{i,1})
            
            test = processedPointCloud{i,1};
            params = cell2mat(processedLabels{i,j});
            model = cuboidModel(params);
    
            indices = findPointsInsideCuboid(model, test);
            cubPtCloud = select(test,indices);
            %view(player,cubPtCloud)

            roiPoints{i,index} = cubPtCloud;

            index = index + 1;
        end
    end
    

    optimalPrevFrame = [0 5 5 5 6 4];
    preFrames = max(optimalPrevFrame);
    %kvPairs = containers.Map(cat, optimalPrevFrame)

    for i = (preFrames+1):numFrames
        testThing = [];
        currFrame = processedPointCloud{i,1};
        if(~isempty(roiPoints{i,1}))
            non_empty_cells = cellfun(@(x) ~isempty(x), processedLabels{i,:});
            targetLabel = find(non_empty_cells);
            preFramesOpt = optimalPrevFrame(targetLabel);
            for j = 1:preFramesOpt
                for k = 1:size(roiPoints{preFramesOpt+1-j,:}, 2)

                    non_empty_cells = cellfun(@(x) ~isempty(x), processedLabels{i-j,:});
                    prevFrameLabel = find(non_empty_cells);


                    if (targetLabel == prevFrameLabel)
                        testThing(j) = prevFrameLabel;
                        backFrame = roiPoints{i-j,k};
                        currFrame = pccat([currFrame;backFrame]);
                    end
                end
            end
        % %disp(testThing
        newPCArray{i-preFrames,1} = currFrame;
        %view(player,currFrame)
        end
        
    end
    processedPointCloud = newPCArray;
    processedLabels = processedLabels(preFrames+1:end, :);

end


%processedLabels = [processedLabels;processedLabels;processedLabels;processedLabels;processedLabels;processedLabels];
%processedPointCloud = [processedPointCloud;processedPointCloud;processedPointCloud;processedPointCloud;processedPointCloud;processedPointCloud];
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
lds = fileDatastore(dataLocation,'ReadFcn',@(x)pcread(x));
bds = boxLabelDatastore(trainLabels);
cds = combine(lds,bds);

%Data Augmentation---------------------------------------------------------
disp("Data Augmentation")

classNames = trainLabels.Properties.VariableNames;
sampleLocation = fullfile(outputFolderDatabase,'my_GTsamples');
[ldsSampled,bdsSampled] = sampleLidarData(cds,classNames,'MinPoints',20,...                  
                            'Verbose',false,'WriteLocation',sampleLocation);
cdsSampled = combine(ldsSampled,bdsSampled);

numObjects = 1;
cdsAugmented = transform(cds,@(x)pcBboxOversample(x,cdsSampled,classNames,numObjects));

cdsAugmented = transform(cdsAugmented,@(x)augmentData(x));
 
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
executionEnvironment = "gpu";
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
    'LearnRateDropPeriod',20,...
    'LearnRateDropFactor',0.8,...
    'ExecutionEnvironment',executionEnvironment,...
    'DispatchInBackground',dispatchInBackground,...
    'BatchNormalizationStatistics','moving',...
    'ResetInputNormalization',false,...
    'CheckpointPath',tempdir);

if doTraining    
   
    [detector,info] = trainPointPillarsObjectDetector(cdsAugmented,detector,options);

    outputFile = fullfile(outputFolderDatabase, "my_trained_detector_Smush3_bigguy.mat");
    save(outputFile, "detector");
else
    outputFile = fullfile(outputFolderDatabase, "my_trained_detector_Smush2_3action.mat");
    pretrainedDetector = load(outputFile,'detector');
    detector = pretrainedDetector.detector;
end




%Evaluate Detector Using Test Set------------------------------------------
disp("Evaluate Detector Using Test Set")
numInputs = 2000;

% Generate rotated rectangles from the cuboid labels.
bds = boxLabelDatastore(testLabels(1:numInputs,:));
groundTruthData = transform(bds,@(x)createRotRect(x));
%disp(testLabels(1:numInputs,:))

%disp(bds.LabelData(:,2))
testValues = cell2table(bds.LabelData(:,2));
testValues = string(table2array(testValues).');

categories = unique(testValues);
data_categorical = categorical(testValues);

% Count occurrences
occurrences = histcounts(data_categorical);
disp("Testing Value Counts:")
testTable = array2table(occurrences, 'VariableNames', categories);
disp(testTable)
% Set the threshold values.
nmsPositiveIoUThreshold = 0.25;
confidenceThreshold = 0.5;

detectionResults = detect(detector,testData(1:numInputs,:),...
    'Threshold',confidenceThreshold);
%disp(detectionResults)

answerTable = testValues;
resultsArray = zeros(size(categories));
for i = 1:height(detectionResults)
    [M,bestIndex] = max(detectionResults.Scores{i});
    if(~isempty(bestIndex))
        guessTable = detectionResults.Labels{i};
        bestGuess = guessTable(bestIndex);
        if(bestGuess == answerTable(i))
            result = find(categories == answerTable(i));
            resultsArray(result) = resultsArray(result)+1;
        end
    end
end

percentConfAcc = resultsArray./occurrences;
disp("Percent of times highest confidence value was correct the label:")
confAccTable = array2table(percentConfAcc, 'VariableNames', categories);
disp(confAccTable);



% Convert the bounding boxes to rotated rectangles format and calculate
% the evaluation metrics.
for i = 1:height(detectionResults)
    box = detectionResults.Boxes{i};
    detectionResults.Boxes{i} = box(:,[1,2,4,5,9]);
end
metrics = evaluateDetectionAOS(detectionResults,groundTruthData,...
    nmsPositiveIoUThreshold);
disp(metrics(:,1:end))






















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