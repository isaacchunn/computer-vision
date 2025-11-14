%% Niblack Adaptive Thresholding with Grid Search Optimization
% Implements Niblack's local thresholding with exhaustive grid search
% to find optimal k (sensitivity) and window size parameters

clear; clc; close all;

% File paths
ASSET_PATH = 'assets/';
documents = {'document01.bmp', 'document02.bmp', 'document03.bmp', 'document04.bmp'};
truths = {'document01-GT.tiff', 'document02-GT.tiff', 'document03-GT.tiff', 'document04-GT.tiff'};

% Grid search parameters
PARAM_K_VALUES = -2.0:0.05:1.0;     % k parameter values (61 values)
PARAM_WIN_VALUES = [11, 21, 31, 41, 51, 61, 71, 81, 91, 101, 121, 141, 161, 181, 201];  % Window sizes (15 values)

%% 1. Data Loading
fprintf('1. Loading images and ground truth...\n');

numDocs = length(documents);
imgData = cell(numDocs, 1);
gtData = cell(numDocs, 1);

for i = 1:numDocs
    % Read document
    docImg = imread(fullfile(ASSET_PATH, documents{i}));
    if size(docImg, 3) > 1
        docImg = rgb2gray(docImg);
    end
    imgData{i} = docImg;
    
    % Read ground truth
    gtImg = imread(fullfile(ASSET_PATH, truths{i}));
    if ~islogical(gtImg)
        gtImg = imbinarize(gtImg);
    end
    gtData{i} = gtImg;
    
    fprintf('Loaded: %s\n', documents{i});
end

%% 2. Grid Search Setup
fprintf('\n2. Grid search configuration:\n');
fprintf('Parameter space: k=[%.1f, %.1f] (step %.2f), window sizes: %d values\n', ...
        PARAM_K_VALUES(1), PARAM_K_VALUES(end), PARAM_K_VALUES(2)-PARAM_K_VALUES(1), ...
        length(PARAM_WIN_VALUES));
fprintf('Total combinations per document: %d\n', ...
        length(PARAM_K_VALUES) * length(PARAM_WIN_VALUES));

% Storage for results
results = struct('name', {}, 'kOpt', {}, 'winOpt', {}, 'segmentation', {}, ...
                 'errors', {}, 'accuracy', {}, 'errorRate', {});

%% 3. Grid Search Optimization
fprintf('\n3. Running exhaustive grid search...\n\n');

for docID = 1:numDocs
    fprintf('Processing %s...\n', documents{docID});
    
    currentImage = imgData{docID};
    currentTruth = gtData{docID};
    [rows, cols] = size(currentImage);
    totalPixels = rows * cols;
    
    % Initialize tracking variables
    minError = inf;
    optK = 0;
    optWin = 0;
    optBinary = [];
    
    % Calculate total combinations
    totalTests = length(PARAM_K_VALUES) * length(PARAM_WIN_VALUES);
    currentTest = 0;
    
    fprintf('Testing %d parameter combinations...\n', totalTests);
    
    % Exhaustive grid search
    for k = PARAM_K_VALUES
        for winSize = PARAM_WIN_VALUES
            currentTest = currentTest + 1;
            
            % Apply Niblack segmentation
            binaryResult = performNiblackSegmentation(currentImage, k, winSize);
            
            % Calculate error
            errorMap = xor(binaryResult, currentTruth);
            errorCount = sum(errorMap(:));
            
            % Update if better
            if errorCount < minError
                minError = errorCount;
                optK = k;
                optWin = winSize;
                optBinary = binaryResult;
                fprintf('[%d/%d] New best: k=%.2f, window=%d, errors=%d\n', ...
                        currentTest, totalTests, k, winSize, errorCount);
            end
        end
    end
    
    % Calculate metrics
    accuracy = 100 * (totalPixels - minError) / totalPixels;
    errorRate = 100 * minError / totalPixels;
    
    % Store results
    results(docID).name = documents{docID};
    results(docID).kOpt = optK;
    results(docID).winOpt = optWin;
    results(docID).segmentation = optBinary;
    results(docID).errors = minError;
    results(docID).accuracy = accuracy;
    results(docID).errorRate = errorRate;
    
    fprintf('Best parameters: k=%.2f, window=%d\n', optK, optWin);
    fprintf('Accuracy: %.2f%%, Error rate: %.2f%%\n\n', accuracy, errorRate);
end

%% 4. Results Summary and Visualization
fprintf('4. Generating results summary and visualizations...\n\n');

% Display summary table
fprintf('%-20s  %-10s  %-10s  %-12s  %-12s\n', 'Document', 'k', 'Window', 'Accuracy', 'Error Rate');
fprintf('%s\n', repmat('-', 1, 70));
for i = 1:numDocs
    fprintf('%-20s  %10.2f  %10d  %11.2f%%  %11.2f%%\n', ...
            results(i).name, results(i).kOpt, results(i).winOpt, ...
            results(i).accuracy, results(i).errorRate);
end
fprintf('\n');

% Create detailed visualizations
for i = 1:numDocs
    createVisualization(imgData{i}, gtData{i}, results(i));
end

fprintf('Analysis complete.\n');

%% Helper Functions

function binary = performNiblackSegmentation(img, k, winSize)
    % Niblack's adaptive thresholding implementation
    % T(x,y) = μ(x,y) + k×σ(x,y)
    
    img = double(img);
    radius = floor(winSize / 2);
    
    % Pad image for boundary handling
    paddedImg = padarray(img, [radius, radius], 'symmetric');
    
    % Compute local statistics
    avgKernel = fspecial('average', winSize);
    localMu = imfilter(paddedImg, avgKernel, 'replicate');
    localMuSq = imfilter(paddedImg.^2, avgKernel, 'replicate');
    
    % Extract central region
    localMu = localMu(radius+1:end-radius, radius+1:end-radius);
    localMuSq = localMuSq(radius+1:end-radius, radius+1:end-radius);
    
    % Compute standard deviation
    localVar = max(0, localMuSq - localMu.^2);
    localSigma = sqrt(localVar);
    
    % Apply Niblack formula
    adaptiveThreshold = localMu + k * localSigma;
    
    % Threshold to binary
    binary = img > adaptiveThreshold;
end

function createVisualization(original, truth, result)
    % Create comprehensive analysis figure
    figHandle = figure('Name', ['Grid Search Analysis: ' result.name], ...
                       'Position', [50, 50, 1400, 800]);
    
    errorMap = xor(result.segmentation, truth);
    
    % Original image
    subplot(2, 4, 1);
    imshow(original);
    title('Original Document', 'FontSize', 11, 'FontWeight', 'bold');
    
    % Ground Truth
    subplot(2, 4, 2);
    imshow(truth);
    title('Ground Truth', 'FontSize', 11, 'FontWeight', 'bold');
    
    % Niblack result
    subplot(2, 4, 3);
    imshow(result.segmentation);
    title(sprintf('Niblack Result\n(k=%.2f, w=%d)', result.kOpt, result.winOpt), ...
          'FontSize', 11, 'FontWeight', 'bold');
    
    % Error map
    subplot(2, 4, 4);
    imshow(errorMap);
    title(sprintf('Error Map\n(%.2f%% errors)', result.errorRate), ...
          'FontSize', 11, 'FontWeight', 'bold');
    
    % Color overlay
    subplot(2, 4, 5);
    overlay = cat(3, uint8(result.segmentation)*255, ...
                     uint8(truth)*255, ...
                     uint8(~errorMap)*255);
    imshow(overlay);
    title('Overlay (R=Result, G=Truth)', 'FontSize', 11, 'FontWeight', 'bold');
    
    % Intensity histogram
    subplot(2, 4, 6);
    histogram(original, 100, 'FaceColor', [0.6 0.6 0.6], 'EdgeColor', 'none');
    xlabel('Intensity');
    ylabel('Frequency');
    title('Intensity Distribution', 'FontSize', 11, 'FontWeight', 'bold');
    grid on;
    
    % Metrics panel
    subplot(2, 4, 7);
    axis off;
    metricsText = sprintf(['PERFORMANCE METRICS\n\n' ...
                          'Method: Grid Search\n' ...
                          'Combinations: %d\n\n' ...
                          'Optimal k: %.2f\n' ...
                          'Optimal window: %d px\n\n' ...
                          'Accuracy: %.2f%%\n' ...
                          'Error Rate: %.2f%%\n' ...
                          'Errors: %d pixels'], ...
                          length(-2.0:0.05:1.0) * 15, ...
                          result.kOpt, result.winOpt, ...
                          result.accuracy, result.errorRate, result.errors);
    text(0.1, 0.5, metricsText, 'FontSize', 10, ...
         'VerticalAlignment', 'middle', 'FontName', 'Courier');
    
    % Adaptive threshold map
    subplot(2, 4, 8);
    threshMap = computeThresholdMap(original, result.kOpt, result.winOpt);
    imagesc(threshMap);
    colormap(gca, 'hot');
    colorbar;
    title('Adaptive Threshold Map', 'FontSize', 11, 'FontWeight', 'bold');
    axis image;
end

function threshMap = computeThresholdMap(img, k, winSize)
    % Generate local threshold values for visualization
    img = double(img);
    radius = floor(winSize / 2);
    paddedImg = padarray(img, [radius, radius], 'symmetric');
    
    avgKernel = fspecial('average', winSize);
    localMu = imfilter(paddedImg, avgKernel, 'replicate');
    localMuSq = imfilter(paddedImg.^2, avgKernel, 'replicate');
    
    localMu = localMu(radius+1:end-radius, radius+1:end-radius);
    localMuSq = localMuSq(radius+1:end-radius, radius+1:end-radius);
    
    localVar = max(0, localMuSq - localMu.^2);
    localSigma = sqrt(localVar);
    
    threshMap = localMu + k * localSigma;
end
