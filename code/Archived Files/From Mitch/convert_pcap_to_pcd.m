outputFolder = fullfile('C:\Users\mzinc\OneDrive\Desktop\OSS CAPSTONE');
%Saving the PCD files of the data
pcdfolder = fullfile(outputFolder, 'output')
%Reading the lidar file
veloReader = velodyneFileReader('C:\Users\mzinc\OneDrive\Desktop\OSS CAPSTONE\Database\Data\Capture3_120.pcap','VLP16');
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
    name = sprintf('frame%04d.pcd',frame);
    pcwrite(ptCloud, fullfile(pcdfolder,name));
    frame=frame+1;
end