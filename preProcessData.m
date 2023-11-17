function [trainData, testData, trainLabels, testLabels, dataLocation] = preProcessData(boxLabels, veloReader, outputFolder)
    % This function takes in the veloReader object, and the corresponding
    % labels variable. Formatting the labels and data into two tables of
    % cells.
    
    % count how many cells have no label
    count_empty = 0;
    for i = 1:size(boxLabels, 1)
        if isempty(cell2mat(boxLabels{i,1}))
            count_empty = count_empty + 1;
        end
    end
    % ISSUE: if count_empty is used in the definition of processed_label
    % the format of processed_label after doing cell2table gets messed up.
    % format of processed_labels after doing cell2table should be the same
    % as boxlabels without the empty cells.
    label_size = size(boxLabels,1) - count_empty;

    % setting up processed labels and data cell arrays
    processed_labels = cell(label_size, 1);
    numFrames = size(boxLabels, 1);
    processed_data = cell(size(numFrames));
    classNames = boxLabels.Properties.VariableNames;
    count = 1;

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
            % ignore any frames where the label is empty
            if isempty(labels)
                del = true;
            else
                processed_labels{count,ii} = labels;
            end
        end
        % if the label has been ignored for this frame make sure the
        % ptcloud for this frame also gets ignored
        if del == false
            processed_data{count,1} = ptCloudObj;
            count = count + 1;
        end
    end
  
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
