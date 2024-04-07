%TODO: add method for emptying output folder before creating new files

%Saving the PCD files of the data
pcdfolder = fullfile(pwd, '\Database\my_Lidar\')
%Reading the lidar file

%creates the full path to read from
lidarPath = fullfile(pwd,'\Database\Data\');
lidarData = dir(lidarPath);

lidarCount = 0


for i = 3:size(lidarData)

    lidarFilePath = append(lidarPath,lidarData(i).name);
    veloReader = velodyneFileReader(lidarFilePath,'VLP16');

    %Limits of the Lidar
    xlimits = [-60 60];
    ylimits = [-80 80];
    zlimits = [-20 20];
    player = pcplayer(xlimits,ylimits,zlimits);
    %Label the Axes
    xlabel(player.Axes,'X (m)');
    ylabel(player.Axes,'Y (m)');
    zlabel(player.Axes,'Z (m)');
    frame=1;
    %Display
    
    while(hasFrame(veloReader) && player.isOpen())
        ptCloud = readFrame(veloReader,frame);
        ptCloud = pointCloud(reshape(ptCloud.Location, [],3), 'Intensity',single(reshape(ptCloud.Intensity, [],1)));

        name = sprintf('frame%04d.pcd',frame+lidarCount);
        pcwrite(ptCloud, fullfile(pcdfolder,name));
        frame=frame+1;
    end

    lidarCount = lidarCount + veloReader.NumberOfFrames;

end
