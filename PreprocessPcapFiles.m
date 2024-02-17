
%MAKE SURE THAT THE .pcap FILE IS IN THE SAME DIRECTORY AS THE CODE!

%set this line to be the file you want to label:
pcapFileName = "CaptureLidarHAR_Cap2";

%set this to be the location you want to have the .pcd to output, make sure that it is in its own folder and that folder is empty
pcdfolder = append(pwd,"\ProcessedPCD");

%setting this to false will make it run faster
doShowVideo = true;


%Limits of the Lidar
xlimits = [-5 15];
ylimits = [-5 15];
zlimits = [-5 5];

lidarFilePath = append(pwd,"\",pcapFileName,".pcap");
veloReader = velodyneFileReader(lidarFilePath,'VLP16');

player = pcplayer(xlimits,ylimits,zlimits);
%Label the Axes
xlabel(player.Axes,'X (m)');
ylabel(player.Axes,'Y (m)');
zlabel(player.Axes,'Z (m)');
frame=1;

processedPointCloud = cell(veloReader.NumberOfFrames, 1);

while(hasFrame(veloReader) && player.isOpen())
    ptCloud = readFrame(veloReader,frame);
    ptCloud = pointCloud(reshape(ptCloud.Location, [],3), 'Intensity',single(reshape(ptCloud.Intensity, [],1)));

    roi = [-4.6 6.3 -1.15 9.55 -1 3];
    indices = findPointsInROI(ptCloud,roi);
    ptCloud = select(ptCloud,indices);
    if(doShowVideo)
        view(player,ptCloud); 
    end

    processedData = removeInvalidPoints(ptCloud);
    processedPointCloud{frame,1} = processedData;

    name = sprintf('frame%04d.pcd',frame);
    pcwrite(processedData, fullfile(pcdfolder,name));
    frame=frame+1;

end
outFileName = append(pcapFileName,".mat");
save(outFileName, "processedPointCloud");

