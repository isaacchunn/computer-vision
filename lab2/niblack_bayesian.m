%% Niblack Adaptive Thresholding with Bayesian Parameter Optimization
% Implements Niblack's local thresholding with automatic parameter tuning

clear; clc; close all;

% File paths
ASSET_PATH = 'assets/';
documents = {'document01.bmp', 'document02.bmp', 'document03.bmp', 'document04.bmp'};
truths = {'document01-GT.tiff', 'document02-GT.tiff', 'document03-GT.tiff', 'document04-GT.tiff'};

% Optimization parameters
PARAM_K_BOUNDS = [-2.0, 1.0];      % k parameter range
PARAM_WIN_BOUNDS = [1, 501];      % Window size range (pixels)
MAX_EVALUATIONS = 50;              % Number of Bayesian iterations
EXPLORATION_RATE = 0.3;            % Exploration vs exploitation ratio

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

%% 2. Optimization Setup
fprintf('\n2. Bayesian optimization configuration:\n');
fprintf('Parameter space: k=[%.1f, %.1f], window=[%d, %d]\n', ...
        PARAM_K_BOUNDS(1), PARAM_K_BOUNDS(2), PARAM_WIN_BOUNDS(1), PARAM_WIN_BOUNDS(2));
fprintf('Evaluations per document: %d\n', MAX_EVALUATIONS);

% Storage for results
results = struct('name', {}, 'kOpt', {}, 'winOpt', {}, 'segmentation', {}, ...
                 'errors', {}, 'accuracy', {}, 'precision', {}, 'recall', {}, 'f1', {}, 'time', {});

%% 3. Parameter Optimization
fprintf('\n3. Running parameter optimization...\n\n');

for docID = 1:numDocs
    fprintf('Processing %s...\n', documents{docID});
    
    currentImage = imgData{docID};
    currentTruth = gtData{docID};
    [rows, cols] = size(currentImage);
    
    % Define objective function (minimize pixel errors)
    objFunc = @(p) computeSegmentationCost(currentImage, currentTruth, p.k, p.window);
    
    % Setup optimization variables
    varK = optimizableVariable('k', PARAM_K_BOUNDS);
    varWindow = optimizableVariable('window', PARAM_WIN_BOUNDS, 'Type', 'integer');
    
    % Run Bayesian optimization
    tStart = tic;
    
    optResult = bayesopt(objFunc, [varK, varWindow], ...
        'MaxObjectiveEvaluations', MAX_EVALUATIONS, ...
        'AcquisitionFunctionName', 'expected-improvement-plus', ...
        'ExplorationRatio', EXPLORATION_RATE, ...
        'IsObjectiveDeterministic', true, ...
        'Verbose', 1, ...
        'PlotFcn', {@plotObjectiveModel, @plotMinObjective});
    
    timeElapsed = toc(tStart);
    
    % Extract optimized parameters
    optK = optResult.XAtMinObjective.k;
    optWin = optResult.XAtMinObjective.window;
    
    % Ensure odd window size
    if mod(optWin, 2) == 0
        optWin = optWin + 1;
    end
    
    % Apply Niblack with optimal parameters
    finalBinary = performNiblackSegmentation(currentImage, optK, optWin);
    
    % Calculate comprehensive metrics
    TP = sum(finalBinary(:) & currentTruth(:));
    TN = sum(~finalBinary(:) & ~currentTruth(:));
    FP = sum(finalBinary(:) & ~currentTruth(:));
    FN = sum(~finalBinary(:) & currentTruth(:));
    
    precision = TP / (TP + FP + eps);
    recall = TP / (TP + FN + eps);
    f1score = 2 * precision * recall / (precision + recall + eps);
    accuracy = 100 * (TP + TN) / (rows * cols);
    errorCount = FP + FN;
    
    % Store results
    results(docID).name = documents{docID};
    results(docID).kOpt = optK;
    results(docID).winOpt = optWin;
    results(docID).segmentation = finalBinary;
    results(docID).errors = errorCount;
    results(docID).accuracy = accuracy;
    results(docID).precision = precision;
    results(docID).recall = recall;
    results(docID).f1 = f1score;
    results(docID).time = timeElapsed;
    
    % Print summary
    fprintf('Best parameters: k=%.4f, window=%d\n', optK, optWin);
    fprintf('Accuracy: %.2f%%, F1-Score: %.4f\n', accuracy, f1score);
    fprintf('Time: %.1f seconds\n\n', timeElapsed);
    
    % Plot convergence curve
    createConvergencePlot(optResult, documents{docID});
end

%% 4. Results Summary and Visualization
fprintf('4. Generating results summary and visualizations...\n\n');

% Display summary table
fprintf('%-20s  %-10s  %-10s  %-12s  %-12s  %-10s\n', 'Document', 'k', 'Window', 'Diff Sum', 'Accuracy', 'F1-Score');
fprintf('%s\n', repmat('-', 1, 85));
for i = 1:numDocs
    fprintf('%-20s  %10.4f  %10d  %12d  %11.2f%%  %10.4f\n', ...
            results(i).name, results(i).kOpt, results(i).winOpt, ...
            results(i).errors, results(i).accuracy, results(i).f1);
end
fprintf('\n');

% Create detailed visualizations for each document
for i = 1:numDocs
    createDetailedVisualization(imgData{i}, gtData{i}, results(i), MAX_EVALUATIONS);
end

fprintf('Analysis complete.\n');

%% Helper Functions

function cost = computeSegmentationCost(image, groundTruth, k, windowSize)
    % Objective function: returns pixel mismatch count
    if mod(windowSize, 2) == 0
        windowSize = windowSize + 1;
    end
    
    binaryResult = performNiblackSegmentation(image, k, windowSize);
    differenceMap = xor(binaryResult, groundTruth);
    cost = sum(differenceMap(:));
end

function binary = performNiblackSegmentation(img, k, winSize)
    % Niblack's adaptive thresholding implementation
    % T(x,y) = μ(x,y) + k×σ(x,y)
    
    img = double(img);
    radius = floor(winSize / 2);
    
    % Pad image for boundary handling
    paddedImg = padarray(img, [radius, radius], 'symmetric');
    
    % Compute local statistics using convolution
    avgKernel = fspecial('average', winSize);
    localMu = imfilter(paddedImg, avgKernel, 'replicate');
    localMuSq = imfilter(paddedImg.^2, avgKernel, 'replicate');
    
    % Extract central region (remove padding)
    localMu = localMu(radius+1:end-radius, radius+1:end-radius);
    localMuSq = localMuSq(radius+1:end-radius, radius+1:end-radius);
    
    % Compute standard deviation: σ = sqrt(E[X²] - E[X]²)
    localVar = max(0, localMuSq - localMu.^2);
    localSigma = sqrt(localVar);
    
    % Apply Niblack formula
    adaptiveThreshold = localMu + k * localSigma;
    
    % Threshold to binary
    binary = img > adaptiveThreshold;
end

function createConvergencePlot(optResult, docName)
    % Visualize optimization convergence
    figHandle = figure('Name', ['Convergence: ' docName], ...
                       'Position', [100, 100, 900, 500]);
    
    subplot(1, 2, 1);
    plot(1:length(optResult.ObjectiveTrace), optResult.ObjectiveTrace, ...
         'o-', 'Color', [0.2 0.5 0.9], 'LineWidth', 1.5, 'MarkerSize', 4);
    hold on;
    plot(1:length(optResult.ObjectiveMinimumTrace), optResult.ObjectiveMinimumTrace, ...
         '-', 'Color', [0.9 0.2 0.2], 'LineWidth', 2.5);
    xlabel('Iteration', 'FontSize', 11);
    ylabel('Objective Value (Pixel Errors)', 'FontSize', 11);
    title('Optimization Trajectory', 'FontSize', 12, 'FontWeight', 'bold');
    legend({'Evaluated Points', 'Best Found'}, 'Location', 'northeast');
    grid on; box on;
    
    subplot(1, 2, 2);
    improvements = diff([inf; optResult.ObjectiveMinimumTrace]);
    stem(find(improvements < 0), abs(improvements(improvements < 0)), ...
         'filled', 'Color', [0.2 0.7 0.3], 'LineWidth', 1.5);
    xlabel('Iteration', 'FontSize', 11);
    ylabel('Improvement Magnitude', 'FontSize', 11);
    title('Optimization Improvements', 'FontSize', 12, 'FontWeight', 'bold');
    grid on; box on;
end

function createDetailedVisualization(original, truth, result, maxEval)
    % Create comprehensive analysis figure
    figHandle = figure('Name', ['Analysis: ' result.name], ...
                       'Position', [50, 50, 1500, 900]);
    
    diffMap = xor(result.segmentation, truth);
    
    % Subplot 1: Original
    subplot(2, 4, 1);
    imshow(original);
    title('Original Document', 'FontSize', 11, 'FontWeight', 'bold');
    
    % Subplot 2: Ground Truth
    subplot(2, 4, 2);
    imshow(truth);
    title('Ground Truth', 'FontSize', 11, 'FontWeight', 'bold');
    
    % Subplot 3: Segmentation Result
    subplot(2, 4, 3);
    imshow(result.segmentation);
    title(sprintf('Niblack Result\n(k=%.3f, w=%d)', result.kOpt, result.winOpt), ...
          'FontSize', 11, 'FontWeight', 'bold');
    
    % Subplot 4: Error Map
    subplot(2, 4, 4);
    imshow(diffMap);
    title(sprintf('Error Map\n(%.2f%% errors)', 100*(1-result.accuracy/100)), ...
          'FontSize', 11, 'FontWeight', 'bold');
    
    % Subplot 5: Color Overlay
    subplot(2, 4, 5);
    overlay = cat(3, uint8(result.segmentation)*255, ...
                     uint8(truth)*255, ...
                     uint8(~diffMap)*255);
    imshow(overlay);
    title('Overlay (R=Result, G=Truth)', 'FontSize', 11, 'FontWeight', 'bold');
    
    % Subplot 6: Intensity Histogram
    subplot(2, 4, 6);
    histogram(original, 100, 'FaceColor', [0.6 0.6 0.6], 'EdgeColor', 'none');
    xlabel('Intensity');
    ylabel('Frequency');
    title('Intensity Distribution', 'FontSize', 11, 'FontWeight', 'bold');
    grid on;
    
    % Subplot 7: Metrics Panel
    subplot(2, 4, 7);
    axis off;
    textContent = sprintf(['PERFORMANCE METRICS\n\n' ...
                          'Method: Bayesian Optimization\n' ...
                          'Iterations: %d\n\n' ...
                          'Optimal k: %.4f\n' ...
                          'Optimal window: %d px\n\n' ...
                          'Accuracy: %.2f%%\n' ...
                          'Precision: %.4f\n' ...
                          'Recall: %.4f\n' ...
                          'F1-Score: %.4f\n\n' ...
                          'Errors: %d pixels\n' ...
                          'Runtime: %.2f sec'], ...
                          maxEval, result.kOpt, result.winOpt, ...
                          result.accuracy, result.precision, result.recall, ...
                          result.f1, result.errors, result.time);
    text(0.1, 0.5, textContent, 'FontSize', 10, ...
         'VerticalAlignment', 'middle', 'FontName', 'Courier');
    
    % Subplot 8: Adaptive Threshold Map
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
