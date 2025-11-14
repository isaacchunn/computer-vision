%% Lab 2 - 3D Stereo Vision Implementation
% This script implements disparity map computation for stereo image pairs
% using SSD (Sum of Squared Differences) matching

clear; close all; clc;

%% Configuration Parameters
TEMPLATE_HEIGHT = 11;
TEMPLATE_WIDTH = 11;
MAX_DISP = 15;  % Maximum disparity search range
ASSET_PATH = 'assets/';

%% Part (b): Load Synthetic Stereo Pair
fprintf('Loading synthetic stereo pair...\n');

% Load corridor images
imgLeftPath = fullfile(ASSET_PATH, 'corridorl.jpg');
imgRightPath = fullfile(ASSET_PATH, 'corridorr.jpg');
groundTruthPath = fullfile(ASSET_PATH, 'corridor_disp.jpg');

imgLeft = imread(imgLeftPath);
imgRight = imread(imgRightPath);
groundTruthDisp = imread(groundTruthPath);

% Convert to grayscale if needed
if ndims(imgLeft) == 3
    imgLeftGray = rgb2gray(imgLeft);
else
    imgLeftGray = imgLeft;
end

if ndims(imgRight) == 3
    imgRightGray = rgb2gray(imgRight);
else
    imgRightGray = imgRight;
end

if ndims(groundTruthDisp) == 3
    groundTruthDisp = rgb2gray(groundTruthDisp);
end

% Convert ground truth to double for proper display
groundTruthDisp = double(groundTruthDisp);

fprintf('Synthetic images loaded successfully.\n');
fprintf('Image dimensions: %d x %d\n', size(imgLeftGray, 1), size(imgLeftGray, 2));
fprintf('Ground truth range: [%.2f, %.2f]\n', min(groundTruthDisp(:)), max(groundTruthDisp(:)));

%% Part (c): Compute Disparity Map for Synthetic Images
fprintf('\nComputing disparity map for synthetic stereo pair...\n');

tic;
disparityMapSynthetic = computeDisparityMapSSD(imgLeftGray, imgRightGray, ...
                                               TEMPLATE_HEIGHT, TEMPLATE_WIDTH, ...
                                               MAX_DISP);
elapsedTime = toc;

fprintf('Disparity map computed in %.2f seconds.\n', elapsedTime);
fprintf('Disparity range: [%.2f, %.2f]\n', min(disparityMapSynthetic(:)), max(disparityMapSynthetic(:)));

% Display results
figure('Name', 'Synthetic Stereo - Corridor', 'NumberTitle', 'off', 'Position', [100, 100, 1600, 400]);
subplot(1, 4, 1);
imshow(imgLeftGray);
title('Left Image (Reference)');

subplot(1, 4, 2);
imshow(imgRightGray);
title('Right Image');

subplot(1, 4, 3);
imshow(disparityMapSynthetic, [-15 15]);
title('Computed Disparity Map (SSD)');
colormap(gca, 'gray');
colorbar;

subplot(1, 4, 4);
imshow(groundTruthDisp, []);
title('Ground Truth');
colormap(gca, 'gray');
colorbar;

%% Part (d): Compute Disparity Map for Real Stereo Images
fprintf('\n--- Processing Real Stereo Images ---\n');

% Load triclops images
imgLeftRealPath = fullfile(ASSET_PATH, 'triclopsi2l.jpg');
imgRightRealPath = fullfile(ASSET_PATH, 'triclopsi2r.jpg');
groundTruthRealPath = fullfile(ASSET_PATH, 'triclopsid.jpg');

imgLeftReal = imread(imgLeftRealPath);
imgRightReal = imread(imgRightRealPath);
groundTruthReal = imread(groundTruthRealPath);

% Convert to grayscale if needed
if ndims(imgLeftReal) == 3
    imgLeftRealGray = rgb2gray(imgLeftReal);
else
    imgLeftRealGray = imgLeftReal;
end

if ndims(imgRightReal) == 3
    imgRightRealGray = rgb2gray(imgRightReal);
else
    imgRightRealGray = imgRightReal;
end

if ndims(groundTruthReal) == 3
    groundTruthReal = rgb2gray(groundTruthReal);
end

% Convert ground truth to double for proper display
groundTruthReal = double(groundTruthReal);

fprintf('Real stereo images loaded successfully.\n');
fprintf('Image dimensions: %d x %d\n', size(imgLeftRealGray, 1), size(imgLeftRealGray, 2));

% Compute disparity map
fprintf('Computing disparity map for real stereo pair...\n');
tic;
disparityMapReal = computeDisparityMapSSD(imgLeftRealGray, imgRightRealGray, ...
                                          TEMPLATE_HEIGHT, TEMPLATE_WIDTH, ...
                                          MAX_DISP);
elapsedTimeReal = toc;

fprintf('Disparity map computed in %.2f seconds.\n', elapsedTimeReal);
fprintf('Disparity range: [%.2f, %.2f]\n', min(disparityMapReal(:)), max(disparityMapReal(:)));

% Display results
figure('Name', 'Real Stereo - Triclops', 'NumberTitle', 'off', 'Position', [100, 100, 1600, 400]);
subplot(1, 4, 1);
imshow(imgLeftRealGray);
title('Left Image (Reference)');

subplot(1, 4, 2);
imshow(imgRightRealGray);
title('Right Image');

subplot(1, 4, 3);
imshow(disparityMapReal, [-15 15]);
title('Computed Disparity Map (SSD)');
colormap(gca, 'gray');
colorbar;

subplot(1, 4, 4);
imshow(groundTruthReal, []);
title('Ground Truth');
colormap(gca, 'gray');
colorbar;

%% Part (a): Function to Compute Disparity Map using SSD
function dispMap = computeDisparityMapSSD(leftImg, rightImg, templateHeight, templateWidth, maxDisparity)
    % Convert to double for precision
    leftImg = double(leftImg);
    rightImg = double(rightImg);
    
    [rows, cols] = size(leftImg);
    
    % Initialize disparity map and cost volume
    dispMap = zeros(rows, cols);
    costVolume = inf(rows, cols, 2*maxDisparity + 1);
    
    % Create template matching kernel (ones for SSD computation)
    templateKernel = ones(templateHeight, templateWidth);
    
    fprintf('  Computing disparity map...\n');
    
    % Loop only over disparity values
    disparityIdx = 1;
    for d = -maxDisparity : maxDisparity
        % Shift right image by disparity d
        if d < 0
            % Negative disparity: shift right
            shiftedRight = [rightImg(:, -d+1:end), zeros(rows, -d)];
        elseif d > 0
            % Positive disparity: shift left
            shiftedRight = [zeros(rows, d), rightImg(:, 1:end-d)];
        else
            % Zero disparity: no shift
            shiftedRight = rightImg;
        end
        
        % Compute squared differences using vectorization
        sqDiff = (leftImg - shiftedRight) .^ 2;
        
        % Use conv2 to sum over template windows
        ssdMap = conv2(sqDiff, templateKernel, 'same');
        
        % Store in cost volume
        costVolume(:, :, disparityIdx) = ssdMap;
        disparityIdx = disparityIdx + 1;
    end
    
    fprintf('  Finding minimum cost disparities...\n');
    
    % Find disparity with minimum cost at each pixel
    [~, minIdx] = min(costVolume, [], 3);
    
    % Convert indices to actual disparity values
    dispMap = minIdx - (maxDisparity + 1);
    
    % Handle border regions by setting to zero
    halfH = floor(templateHeight / 2);
    halfW = floor(templateWidth / 2);
    dispMap(1:halfH, :) = 0;
    dispMap(end-halfH+1:end, :) = 0;
    dispMap(:, 1:halfW) = 0;
    dispMap(:, end-halfW+1:end) = 0;
end
