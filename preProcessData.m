function [trainData, testData, trainLabels, testLabels, dataLocation] = preProcessData(boxLabels, veloReader, outputFolder)
    % This function takes in the veloReader object, and the corresponding
    % labels variable. Formatting the labels and data into two tables of
    % cells.
    
    label_size = size(boxLabels,1);
    % setting up processed labels and data cell arrays
    processed_labels = cell(label_size, 1);
    numFrames = size(boxLabels, 1);
    processed_data = cell(size(numFrames));
    classNames = boxLabels.Properties.VariableNames;

    for i = 1:numFrames
        ptCloudObj = readFrame(veloReader);
        groundTruth = boxLabels(i,:);
        procData = removeInvalidPoints(ptCloudObj);
        del = false;
        for ii = 1:numel(classNames)

            labels = groundTruth(1,classNames{ii}).Variables;
            if (iscell(labels))
                labels = labels{1};
            end
            if ~isempty(labels)
                % Find the number of points inside each ground truth
                % label.
                numPoints = arrayfun(@(x)(findPointsInsideCuboid(cuboidModel(labels(x,:)),procData)),...
                            (1:size(labels,1)).','UniformOutput',false);

                posLabels = cellfun(@(x)(length(x) > 50), numPoints);
                labels = labels(posLabels,:);
            end
            processed_labels{i, ii} = labels;
        end
        processed_data{i,1} = ptCloudObj;
    end
    % processed_labels{count,1} = [];
    % at the end of this loop processed_labels should be a table of cells
    processed_labels = cell2table(processed_labels);
    numClasses = size(processed_labels,2);
    for j = 1:numClasses
        processed_labels.Properties.VariableNames{j} = classNames{j};
    end

    % Shuffling indicies  randomlyand splitting data into 70% training and 
    % 30% testing
    rng(1);
    shuffledIndices = randperm(size(processed_labels,1));
    idx = floor(0.7 * length(shuffledIndices));
    
    trainData = processed_data(shuffledIndices(1:idx),:);
    testData = processed_data(shuffledIndices(idx+1:end),:);
    
    trainLabels = processed_labels(shuffledIndices(1:idx),:);
    testLabels = processed_labels(shuffledIndices(idx+1:end),:);

    writeFiles = true;
    dataLocation = fullfile(outputFolder,'InputData');
    [trainData,trainLabels] = saveptCldToPCD(trainData,trainLabels,...
        dataLocation,writeFiles);

end
