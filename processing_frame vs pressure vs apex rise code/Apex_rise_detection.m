
%% Syntax
% The code receives the FFT of the reference and the shifted images, and an
% (integer) upsampling factor. The code expects FFTs with DC in (1,1) so do not use
% fftshift.
%
%    output = dftregistration(fft2(f),fft2(g),usfac);
%
% The images are registered to within 1/usfac of a pixel.
%
% output(1) is the normalized root-mean-squared error (NRMSE) [1] between f and
% g. 
%
% output(2) is the global phase difference between the two images (should be
% zero if images are real-valued and non-negative).
%
% output(3) and output(4) are the row and column shifts between f and g respectively. 
%
%    [output Greg] = dftregistration(fft2(f),fft2(g),usfac);
%
% Greg is an optional output, it returns the Fourier transform of the registered version of g,
% where the global phase difference [output(2)] is also compensated.

% clear variables;
% close all;

%%
figure(1)
close 1
figure(2)
close 2
%% Obtain a reference and shifted images
% To illustrate the use of the algorithm, lets obtain a reference and a
% shifted image. First we read the reference image f(x,y)
[fname, path] = uigetfile('*.tif*', 'Load tiff.');
multiframe_data = loadtiff(strcat(path,fname));
size_data = size(multiframe_data);
num_frames = size_data(end);

if (length(size_data)==4)
    for frame_num=1:num_frames
        multiframe_data_gray(:,:,frame_num) = rgb2gray(multiframe_data(:,:,1:3,frame_num));
    end
else
    multiframe_data_gray = multiframe_data;
end

clear multiframe_data;

f = im2double(multiframe_data_gray(:,:,1));
figure(1);
[I1, rect] = imcrop(f);
rect = round(rect);

clear apex_rise;
apex_rise(1)=0;
%%
warning('off');
for frame_num=2:num_frames
    %display(rect);
    CurrentFrame = multiframe_data_gray(:,:,frame_num-1);
    I1 = im2double(CurrentFrame(rect(2):rect(2)+rect(4), rect(1):rect(1)+rect(3)));
    
    NextFrame = multiframe_data_gray(:,:,frame_num);
    I2 = im2double(NextFrame(rect(2):rect(2)+rect(4), rect(1):rect(1)+rect(3)));


% Sample Image Registration
% dftregistration.m receives the FT of f and g and the upsampling factor. 
% The code expects DC of the FTs at (1,1) so don't use fftshift. 
%
% We now use the image registration code to register f and g within 0.01
% pixels by specifying an upsampling parameter of 100
    usfac = 100;
    [output, I2reg] = dftregistration(fft2(I1),fft2(I2),usfac);
    %display(output);
    
    apex_rise(frame_num) = apex_rise(frame_num-1) + output(3);

    figure(2);
    subplot(1,2,1);
    imshow(abs(I1));
    title('Reference image, f(x,y)')
    subplot(1,2,2);
    imshow(abs(ifft2(I2reg)));
    title('Registered image, gr(x,y)')
    
    %update search region
    rect_old = rect;
    %rect_new = round([rect_old(1)-output(4), rect_old(2)-output(3), rect_old(3), rect_old(4)]);
    rect_new = [rect_old(1)-output(4), rect_old(2)-output(3), rect_old(3), rect_old(4)];
    %rect = [rect_new(1), max(rect_new(2),1), rect_new(3), rect_new(4)+min(1,rect_new(2))-1];
    rect = [rect_new(1), max(rect_new(2),1), rect_new(3), min(rect_new(4)+min(1,rect_new(2))-1, size(CurrentFrame,1)-max(rect_new(2),1))];
end

apex_rise_metric = apex_rise' * 2.65/size(multiframe_data_gray,1);

figure(3), plot(apex_rise_metric)
warning('on');

%% Save [Apex Rise, Pressure] pairs to CSV file

filename = strcat(fname(1:end-5), '_Apex_raw.csv'); % .tiff 5 letters 
writematrix([apex_rise_metric], cat(2,path,filename));