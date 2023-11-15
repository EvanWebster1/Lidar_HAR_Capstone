veloReader = velodyneFileReader("Database\Data\capture1_120.pcap","VLP16");

labels = load("Database\Labels\Capture1_120_labels.mat");

dim = size(labels.gTruth.LabelData.Human);
processed_label = zeros(dim(1), 9);
for i = 1:size(labels.gTruth.LabelData.Human)
    if ~isempty(cell2mat(labels.gTruth.LabelData.Human(i)))
        temparr = cell2mat(labels.gTruth.LabelData.Human(i));
        for j = 1:9
            processed_label(i,j) = temparr(j);
        end
    elseif isempty(cell2mat(labels.gTruth.LabelData.Human(i))) && ~isempty(cell2mat(labels.gTruth.LabelData.Human(i+1)))
        temparr = cell2mat(labels.gTruth.LabelData.Human(i-1));
        if isempty(cell2mat(labels.gTruth.LabelData.Human(i-1))) && isempty(cell2mat(labels.gTruth.LabelData.Human(i-2)))
            continue
        else 
            for j = 1:9
                processed_label(i,j) = temparr(j);
            end
        end
    end
end

% Setting the player axis limits 
xlimits = [0 10];
ylimits = [-6 6];
zlimits = [-2 3];
% Creating the axis object to use as parent
ax = axes(XLim=xlimits, YLim=ylimits, ZLim=zlimits);

% Defining the pcplayer
player = pcplayer(xlimits, ylimits, zlimits, 'Parent',ax);

pointcldRange = [xlimits(1) xlimits(2) ylimits(1) ylimits(2) zlimits(1) zlimits(2)];
classNames = labels.gTruth.LabelDefinitions.Name;
% anchorBoxes = calculateAnchorsPointPillars()


% getting the current time
veloReader.CurrentTime = veloReader.StartTime; 
i = 1;
count = 0;
% gifFile = 'capture1.gif';
% exportgraphics(ax,gifFile);
while(hasFrame(veloReader) && player.isOpen() && (veloReader.CurrentTime < veloReader.EndTime))
    ptCloudObj = readFrame(veloReader);
    view(player,ptCloudObj.Location,ptCloudObj.Intensity);
    if ~sum(processed_label(i,:)) == 0
        count = count + 1;
        % Overlaying the box and label onto the pointcloud being output
        showShape('cuboid', processed_label(i,:), 'Parent', ax, ...
            'Opacity', 0.1, 'Color', 'green', 'LineWidth', 0.5, 'LabelOpacity', 0.4, ...
            'LabelFontSize', 6, Label=labels.gTruth.LabelDefinitions.Name);
    else
        % Overlaying the box and label onto the pointcloud being output
        showShape('cuboid', processed_label(count,:), 'Parent', ax, ...
            'Opacity', 0, 'Color', 'green', 'LineWidth', 0.5, 'LineOpacity', 0, 'LabelOpacity', 0, ...
            'LabelFontSize', 6, Label=labels.gTruth.LabelDefinitions.Name);
    end
    % exportgraphics(ax, gifFile, Append=true);
    i = i + 1;
    pause(0.1);
end

clear player