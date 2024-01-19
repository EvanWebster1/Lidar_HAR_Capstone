function augmentedPcBoxLabels = pcBboxOversampler(pcBoxLabels,sampleData,classNames,totalObjects)
% This is a slightly altered version of the matlab lidar toolbox function
% pcBboxOversampler to work with the case where there is no available Bbox
% to add to the pointcloud. All credit for the original function goes to
% mathworks as noted at the end of this comment section.
% pcBboxOversample Sample additional objects to a point cloud
%  pcBboxOversample performs ground truth augmentation by adding additional
%  objects from the datastore to the input point cloud.
%
%  augmentedPcBoxLabels =
%  pcBboxOversample(pcBoxLabels,sampleData,classNames,totalObjects)
%  randomly inserts objects with specified class names, from sampleData,
%  such that the number of objects in the output point cloud is equal to
%  totalObjects. The augmented point cloud and the corresponding box labels
%  are returned as augmentedPcBoxLabels.
%
%  Inputs:
%  ------
%
%  pcBoxLabels              - A cell array of size 1-by-3 that contains the
%                             point cloud, the box annotation and the
%                             respective box categories.
%
%  sampleData               - This must be a datastore or table, with each
%                             row containing only one object
%
%                             Datastore format:
%                             -----------------
%                             A datastore that returns a cell array on the
%                             read methods with three columns.
%                             1st Column: A cell vector of sampled points.
%                             2nd Column: A cell vector that contain 1-by-9
%                                         matrix of [xctr, yctr, zctr,
%                                         xlen, ylen, zlen, xrot, yrot,
%                                         zrot] bounding box specifying
%                                         object locations of the sampled
%                                         points.
%                             3rd Column: A categorical vector containing
%                                         the object class name.
%
%                             Table format:
%                             -------------
%                             A table with 2 or more columns. The first
%                             column must contain point cloud file names
%                             with path. The point cloud can be in any
%                             format supported by pcread. The remaining
%                             columns must contain 1-by-9 matrices of
%                             [xctr, yctr, zctr, xlen, ylen, zlen, xrot,
%                             yrot, zrot] bounding box specifying sampled
%                             object location. Each column represents a
%                             single object class, e.g. pedestrian, car,
%                             truck etc. The table variable names define
%                             the object class name.
%
%  classNames               - Specify the names of object classes that
%                             must be added from the datastore. classNames
%                             must be a string vector, a categorical
%                             vector, or a cell array of character vectors.
%
%  totalObjects             - The total number of objects in the output
%                             point cloud. This can be a scalar, in which
%                             case the same value is used for all the
%                             classes, or it can be a vector where each
%                             value corresponds to the respective class
%                             specified in classNames.
%
%  Output
%  ------
%
%  augmentedPcBoxLabels     - A cell array of size 1-by-3 containing the
%                             augmented point cloud, box annotation and the
%                             respective box categories.
%
%  Notes:
%  -----
%  Use sampleLidarData and pcBboxOversample functions together to perform
%  Ground truth data augmentation for lidar object detection. This is an
%  augmentation technique, where randomly selected ground truth boxes from
%  other point clouds are introduced into the input point cloud. Using this
%  approach, the number of ground truth boxes per point cloud are
%  increased. A collision test is performed on the samples to be added and
%  the ground truth boxes of the input point cloud to avoid overlap. This
%  technique alleviates class imbalance problem present while performing
%  lidar object detection.
%
%  Example 1: Perform the ground truth data augmentation on a point cloud
%  ----------------------------------------------------------------------
%  % Load point cloud data and its labels.
%  dataLocation = fullfile(toolboxdir('lidar'), 'lidardata', ...
%      'sampleWPIPointClouds','pointClouds');
%  load('sampleWPILabels.mat','trainLabels');
%
%  % Create the training data.
%  pcds = fileDatastore(dataLocation,'ReadFcn',@(x) pcread(x));
%  blds = boxLabelDatastore(trainLabels);
%  trainingData = combine(pcds,blds);
%
%  % Define the number of objects and its categories.
%  totalObjects = 5;
%  classNames = {'car'};
%
%  % Create the combined datastore of sampled objects.
%  [pcdsSampled,bldsSampled] = sampleLidarData(trainingData,classNames);
%  cdsSampled = combine(pcdsSampled,bldsSampled);
%
%  % Read a sample from the input training data.
%  pcBoxLabels = read(trainingData);
%
%  % Perform the augmentation.
%  augmentedPcBoxLabels = pcBboxOversample(pcBoxLabels,cdsSampled,classNames,totalObjects);
%
%  Example 2: Perform the ground truth data augmentation on a datastore
%  --------------------------------------------------------------------
%  % Load point cloud data and its labels.
%  dataLocation = fullfile(toolboxdir('lidar'), 'lidardata', ...
%      'sampleWPIPointClouds','pointClouds');
%  load('sampleWPILabels.mat','trainLabels');
%
%  % Create the training data.
%  pcds = fileDatastore(dataLocation,'ReadFcn',@(x) pcread(x));
%  blds = boxLabelDatastore(trainLabels);
%  trainingData = combine(pcds,blds);
%
%  % Define the number of objects and its categories.
%  totalObjects = 5;
%  classNames = {'car'};
%
%  % Create the combined datastore of sampled objects.
%  [pcdsSampled,bldsSampled] = sampleLidarData(trainingData,classNames,'WriteLocation',tempdir);
%  cdsSampled = combine(pcdsSampled,bldsSampled);
%
%  % Perform the augmentation on the training data.
%  cdsAugmented = transform(trainingData,@(x) pcBboxOversample(x,cdsSampled,classNames,totalObjects));
%
%  See also sampleLidarData, pointPillarsObjectDetector, combine

%  Copyright 2021-2022 The MathWorks, Inc.

    arguments
        pcBoxLabels (1,3)
        sampleData
        classNames
        totalObjects
    end

    % Validate pcBoxLabels.
    iValidatepcBoxLabels(pcBoxLabels);

    % Validate and Reset the datastore before preview.
    sampleData = iValidateSampleData(sampleData);

    % Convert to row vector if classNames is a column vector.
    if size(classNames,2) == 1
        classNames = classNames';
    end

    % Validate classNames.
    iValidateClassNames(classNames);
    if iscategorical(classNames)
        classNames = cellstr(classNames);
    end

    % Convert to row vector if totalObjects is a column vector.
    if iscolumn(totalObjects)
        totalObjects = totalObjects';
    end

    % Validate totalObjects.
    iValidateTotalObjects(totalObjects);

    % Convert totalObjects to vector for all the classes if it is as scalar.
    if isscalar(totalObjects)
        totalObjects = totalObjects.*cast(ones(1,numel(classNames)),class(totalObjects));
    elseif (numel(totalObjects) ~= numel(classNames))
        error(message('lidar:pcBboxOversample:invalidtotalObjects',4));
    end

    % Create individual datastore.
    persistent outCDS;
    persistent orgSampleData;

    if isempty(orgSampleData)
        orgSampleData = sampleData;
    end

    if isempty(outCDS) || ~isequal(sampleData,orgSampleData)
        boxLabels = sampleData.UnderlyingDatastores{1,2};

        sampleLabels = boxLabels.LabelData;
        indices = string(sampleLabels(:,2));

        for i = 1:numel(classNames)
            idx = find(indices == classNames(i));
            if isempty(idx)
                error(message('lidar:pcBboxOversample:invalidClassName',3));
            end
            tmpCDS = subset(sampleData,idx);
            outCDS.(classNames{i}) = shuffle(tmpCDS);
        end
    end

    if ~all(matches(cellstr(classNames),fieldnames(outCDS)))
        error(message('lidar:pcBboxOversample:invalidClassName',3));
    end

    ptCld = pcBoxLabels{1,1};
    bboxes = pcBoxLabels{1,2};
    labels = pcBoxLabels{1,3};

    for i = 1:numel(classNames)
        
        idxClasses = (labels == classNames(i));
        bboxClasses = bboxes(idxClasses,:);

        % Find the number of samples to add for each class.
        if isempty(bboxClasses)
            numSamplesToAdd = totalObjects(i);
        else
            numSamplesToAdd = totalObjects(i) - sum(idxClasses);
        end

        if numSamplesToAdd <= 0
            continue;
        end

        cds = outCDS.(classNames{i});

        % Reset and shuffle the datastore when it reaches the end.
        if ~hasdata(cds)
            reset(cds);
            outCDS.(classNames{i}) = shuffle(cds);
        end

        % Get the samples from the datastore to append to the existing data.
        [samplesToAdd,bboxesToAdd,labelsToAdd] = getSamplesFromDatastore(cds,numSamplesToAdd);

        % Get non-overlapping boxes within the selected boxes.
        if ~isempty(bboxesToAdd)
            [samplesToAdd,bboxesToAdd,labelsToAdd] = removeOverlappingBoxes(samplesToAdd,bboxesToAdd,labelsToAdd);
        end

        % Get non-overlapping boxes from selected boxes and ground truth
        % boxes.
        if ~isempty(bboxClasses) && ~isempty(bboxesToAdd)
            overlapRatio = bboxOverlapRatio(bboxesToAdd(:,[1,2,4,5,9]),bboxes(:,[1,2,4,5,9]));
            maxOverlap = max(overlapRatio,[],2);
            idxToAdd = maxOverlap == 0;
            samplesToAdd = samplesToAdd(idxToAdd,:);
            bboxesToAdd = bboxesToAdd(idxToAdd,:);
            labelsToAdd = labelsToAdd(idxToAdd,:);
        end

        % Add the samples, boxes and labels to the ground truth data.pcbboxoversample
        if ~isempty(bboxesToAdd)
            samplesToAdd = [samplesToAdd;ptCld];                %#ok<AGROW>
            ptCld = pccat(samplesToAdd);
            bboxes = [bboxes;bboxesToAdd];                      %#ok<AGROW>
            labels = [labels;labelsToAdd];                      %#ok<AGROW>
        else
            % added solution to case where all bBoxs overlap
            [points, colors, normals, intensity, rangeData] = ...
        arrayfun(@extractValidPoints, ptCld, 'UniformOutput', false);
            points = vertcat(points{:});
            colors = vertcat(colors{:});
            colors = cast([], 'uint8');
            normals   = vertcat(normals{:});
            normals = cast([], 'like', points);
            intensity = vertcat(intensity{:});
            intensity = single(intensity);
            rangeData = vertcat(rangeData{:});
            rangeData = cast([], 'like', points);

            ptCld = pointCloud(points, 'Color', colors, 'Normal', normals, 'Intensity', intensity);
            % ptCld = pointCloud(ptCldLocations, 'Intensity', ptCldIntensity);
            ptCld.RangeData = rangeData;
        end
    end

    augmentedPcBoxLabels = {ptCld,bboxes,labels};
end

%--------------------------------------------------------------------------
function [samplesToAdd,bboxesToAdd,labelsToAdd] = getSamplesFromDatastore(cds,numSamples)

    samplesToAdd = [];
    bboxesToAdd = zeros(numSamples,9);
    labelsToAdd = [];

    for q = 1:numSamples
        if ~hasdata(cds)
            reset(cds);
        end
        outData = read(cds);
        if isstruct(outData{1,1})
            names = fieldnames(outData{1,1});
            sampledPoints = outData{1,1}.(names{1});
        else
            sampledPoints = outData{1,1};
        end
        boxData = outData{1,2};
        labelName = outData{1,3};

        bboxesToAdd(q,:) = boxData;
        samplesToAdd = [samplesToAdd;sampledPoints];      %#ok<AGROW>
        labelsToAdd = [labelsToAdd;labelName];            %#ok<AGROW>
    end

end

%--------------------------------------------------------------------------
function [samplesToAdd,bboxesToAdd,labelsToAdd] = removeOverlappingBoxes(samplesToAdd,bboxesToAdd,labelsToAdd)

    bboxesBEV = bboxesToAdd(:,[1,2,4,5,9]);
    boxscores  = rand(size(bboxesBEV,1),1);
    [~,~,idx] = selectStrongestBbox(bboxesBEV,boxscores,'OverlapThreshold',0);
    bboxesToAdd = bboxesToAdd(idx,:);
    samplesToAdd = samplesToAdd(idx);
    labelsToAdd = labelsToAdd(idx);

end

%--------------------------------------------------------------------------
function iValidatepcBoxLabels(pcBoxLabels)
    if ~iscell(pcBoxLabels)
        error(message('lidar:pcBboxOversample:pcBoxLabelMismatch',1));
    end

    samplePC = pcBoxLabels{1,1};
    bbox = pcBoxLabels{1,2};
    labels = pcBoxLabels{1,3};

    % Check if the data is in the right format.
    if ~isa(samplePC, 'pointCloud')
        error(message('lidar:pcBboxOversample:invalidInputData','pcBoxLabels'));
    end

    if ~isempty(bbox)
        if(size(bbox,2)~=9)
            error(message('lidar:pcBboxOversample:boxSizeMismatch',1));
        end

        if iscategorical(labels) && any(isundefined(labels))
            error(message('lidar:pcBboxOversample:invalidClasses',1));
        elseif (iscellstr(labels) || isstring(labels)) && any(value == "")
            error(message('lidar:pcBboxOversample:invalidClasses',1));
        elseif ~(iscategorical(labels) || isstring(labels) || ischar(labels))
            error(message('lidar:pcBboxOversample:invalidClasses',1));
        end

        if (size(bbox,1) ~= size(labels,1))
            error(message('lidar:pcBboxOversample:boxLabelMismatch',1));
        end
    end
end

%--------------------------------------------------------------------------
function iValidateClassNames(value)
    if ~isvector(value) || ~iIsValidDataType(value)
        error(message('lidar:pcBboxOversample:invalidClasses',3));
    end

    if iscategorical(value) && any(isundefined(value))
        error(message('lidar:pcBboxOversample:invalidClasses',3));
    elseif (iscellstr(value) || isstring(value)) && any(value == "")
        error(message('lidar:pcBboxOversample:invalidClasses',3));
    end

    if iHasDuplicates(value)
        error(message('lidar:pcBboxOversample:duplicateClasses',3));
    end
end

%--------------------------------------------------------------------------
function sampleData = iValidateSampleData(sampleData)

    if istable(sampleData)
        % Create a combined datastore out of the input table.
        sampleData = convertTableToDS(sampleData);
    elseif ~(isa(sampleData,'matlab.io.Datastore') || isa(sampleData,'matlab.io.datastore.Datastore'))
        error(message('lidar:pcBboxOversample:invalidsampleData',2));
    end

    data = read(sampleData);
    if size(data,2) ~= 3
        error(message('lidar:pcBboxOversample:invalidTrainInput',2));
    end

    if isstruct(data{1,1})
        names = fieldnames(data{1,1});
        samplePC = data{1,1}.(names{1});
    else
        samplePC = data{1,1};
    end

    bboxes = data{1,2};

    % Check if the data is in the right format.
    if ~isa(samplePC, 'pointCloud')
        error(message('lidar:pcBboxOversample:invalidInputData','sampleData'));
    end

    % Check if the boxes are in the right format.
    if (size(bboxes,2)~=9) || size(bboxes,1) ~= 1
        error(message('lidar:pcBboxOversample:labelSizeMismatch',2));
    end

    reset(sampleData);
end

%--------------------------------------------------------------------------
function tf = iIsValidDataType(value)
    tf = iscategorical(value) || iscellstr(value) || isstring(value);
end

%--------------------------------------------------------------------------
function tf = iHasDuplicates(value)
    tf = ~isequal(value, unique(value, 'stable'));
end

%--------------------------------------------------------------------------
function trainingData = convertTableToDS(trainingData)
    pcFilenames = trainingData{:,1};
    boxLabels = trainingData(:,2:end);

    % Check if all the groundtruth boxes are in the right format.
    iValidateBoxLabels(boxLabels);

    pcds = fileDatastore(pcFilenames,'ReadFcn',@pcread);
    blds = boxLabelDatastore(boxLabels);
    trainingData = combine(pcds,blds);
end

%--------------------------------------------------------------------------
function iValidateBoxLabels(boxes)
    boxes = table2array(boxes);

    for ii = 1:size(boxes,1)
        value = boxes{ii,1};
        validateattributes(value, {'single','double'}, ...
                           {'ncols',9,'real','nonnan','finite'});
        % Positive length, width, height.
        validateattributes(value(:,4:6), {'single','double'}, {'>',0});
    end
end

%--------------------------------------------------------------------------
function iValidateTotalObjects(value)
    validateattributes(value,{'nonempty','single','double'},...
                       {'nonnan','finite','real','nonsparse','integer','positive'},mfilename,'totalObjects');
end
