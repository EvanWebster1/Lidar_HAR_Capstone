function [trainData, testData, trainLabels, testLabels, dataLocation] = preProcessData(labels, veloReader, outputFolder)
    % This function takes in the veloReader object, and the corresponding
    % labels variable. Formatting the labels and data into two tables of
    % cells.

    processed_labels = cell(size(labels.gTruth.LabelData));
    processed_data = cell(size(labels.gTruth.LabelData));
    numFrames = size(labels.gTruth.LabelData, 1);
    
    for i = 1:numFrames
        ptCloudObj = readFrame(veloReader);
        processed_data{i,1} = ptCloudObj;
        processed_labels{i,1} = labels.gTruth.LabelData.Human(i);
    end
    
    processed_labels = cell2table(processed_labels);
    processed_labels.Properties.VariableNames = labels.gTruth.LabelDefinitions.Name;

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
