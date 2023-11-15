function data = augmentData(data)
% Apply random flipping along Y axis, and random scaling, rotation and
% translation.
    minAngle = -45;
    maxAngle = 45;
    
    % Randomly flip the point cloud.
    data = randomFlip(data);    
    
    % Define outputView based on the grid-size and XYZ limits.
    outView = imref3d([32,32,32],[-100,100],...
                      [-100,100],[-100,100]);

    numObservations = size(data,1);
    for i = 1:numObservations
        theta = minAngle + rand(1,1)*(maxAngle - minAngle);
        tform = randomAffine3d('Rotation',@() deal([0,0,1],theta),...
                               'Scale',[0.95,1.05],...
                               'XTranslation',[0,0.2],...
                               'YTranslation',[0,0.2],...
                               'ZTranslation',[0,0.1]);
        tform = affine3d(tform.T);
        
        % Apply the transformation to the point cloud.
        pc = data{i,1};
        ptCloud = pointCloud(pc(:,1:3),'Intensity',pc(:,4));
        ptCloudTransformed = pctransform(ptCloud,tform);
        
        % Apply the same transformation to the boxes.
        bbox = data{i,2};
        [bbox,indices] = bboxwarp(bbox,tform,outView);   
        if ~isempty(indices)
            data{i,1} = ptCloudTransformed;
            data{i,2} = bbox;
            data{i,3} = data{1,3}(indices,:);
        end
    end    
end

% Randomly flip the point cloud.
function data = randomFlip(data)
    numObservations = size(data,1);
    for i = 1:numObservations
        pc = data{i,1};
        pc = [pc.Location pc.Intensity];
        bbox = data{i,2};
        if randi([0,1])
            bbox(:,2) = -bbox(:,2);
            bbox(:,9) = -bbox(:,9);
            pc(:,2) = -pc(:,2);
        end
        data{i,1} = pc;
        data{i,2} = bbox;
    end
end
