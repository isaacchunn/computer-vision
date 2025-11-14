%% Part 3.1a) Otsu's Global Thresholding
%% 1. Loading of images and ground truths
fprintf('1. Loading images and ground truths...\n');

% Define image folder that stores the assets from NTULearn
imageFolder = 'assets/';

% Define image files and their relevant ground truths
imageFiles = {'document01.bmp', 'document02.bmp', 'document03.bmp', 'document04.bmp'};
groundTruthFiles = {'document01-GT.tiff', 'document02-GT.tiff', 'document03-GT.tiff', 'document04-GT.tiff'};

% Preallocate cell arrays to store loaded images and ground truths
images = cell(length(imageFiles), 1);
groundTruths = cell(length(imageFiles), 1);

% Load all images and ground truths (each should have its own ground truth
for i = 1:length(imageFiles)
    % Read image
    img = imread(fullfile(imageFolder, imageFiles{i}));
    
    % Ensure grayscale if in RGB
    if size(img, 3) == 3
        img = rgb2gray(img);
    end
    images{i} = img;
    
    % Read ground truth
    gt = imread(fullfile(imageFolder, groundTruthFiles{i}));
    
    % Ensure ground truth is binary
    if ~islogical(gt)
        gt = imbinarize(gt);
    end
    groundTruths{i} = gt;
    
    fprintf('  Loaded: %s\n', imageFiles{i});
end

%% 2. Store some results structure for Otsu and best results in subsequent experiments.
fprintf('2. Initializing results storage...\n');

% Structure to store Otsu results
otsuResults = struct('imageName', {}, 'threshold', {}, 'binaryImg', {}, ...
                     'diffSum', {}, 'accuracy', {}, 'errorRate', {});

% Structure to store best threshold results
bestResults = struct('imageName', {}, 'threshold', {}, 'binaryImg', {}, ...
                     'diffSum', {}, 'accuracy', {}, 'errorRate', {});

% Define threshold range for testing
thresholdTests = 0:0.01:1; % Test from 0 to 1 with step 0.01 (i.e 0.01, 0.02...)

%% 3. For each image, process and compute metrics while storing our Otsu results.
fprintf('3. Processing images with Otsu and storing results...\n');

for i = 1:length(imageFiles)
    fprintf('Processing %s...\n', imageFiles{i});
    
    img = images{i};
    gt = groundTruths{i};
    totalPixels = numel(gt);
    
    % Calculate and apply Otsu's thresholding
    otsuLevel = graythresh(img); % Computation of optimal threshold
    binaryImg = imbinarize(img, otsuLevel); % Apply threshold
    
    % Compute difference with ground truth
    diffImage = xor(binaryImg, gt);
    diffSum = sum(diffImage(:)); % Total number of different pixels
    
    % Compute metrics
    correctPixels = totalPixels - diffSum;
    accuracy = (correctPixels / totalPixels) * 100;
    errorRate = (diffSum / totalPixels) * 100;
    
    % Store Otsu results
    otsuResults(i).imageName = imageFiles{i};
    otsuResults(i).threshold = otsuLevel;
    otsuResults(i).binaryImg = binaryImg;
    otsuResults(i).diffSum = diffSum;
    otsuResults(i).accuracy = accuracy;
    otsuResults(i).errorRate = errorRate;
    
    fprintf('    Otsu Threshold: %.3f | Accuracy: %.2f%% | Error Rate: %.2f%%\n', ...
            otsuLevel, accuracy, errorRate);
end

%% 4. Test different thresholds and find the best one
fprintf('4. Testing different thresholds to find optimal values...\n');

for i = 1:length(imageFiles)
    fprintf('\n  Testing thresholds for %s...\n', imageFiles{i});
    
    img = images{i};
    gt = groundTruths{i};
    totalPixels = numel(gt);
    
    % Initialize tracking variables for best threshold
    minDiffSum = inf;
    bestThreshold = 0; % we can also assume the best threshold starts from Otsu level, but this is not guaranteed.
    bestBinaryImg = []; % store empty list
    
    % Test each threshold
    for t = thresholdTests
        % Apply threshold
        testBinaryImg = imbinarize(img, t);
        
        % Compute difference
        testDiffImage = xor(testBinaryImg, gt);
        testDiffSum = sum(testDiffImage(:));
        
        % Check if this is the best threshold so far
        if testDiffSum < minDiffSum
            minDiffSum = testDiffSum;
            bestThreshold = t;
            bestBinaryImg = testBinaryImg;
        end
    end
    
    % Compute metrics for best threshold
    bestCorrectPixels = totalPixels - minDiffSum;
    bestAccuracy = (bestCorrectPixels / totalPixels) * 100;
    bestErrorRate = (minDiffSum / totalPixels) * 100;
    
    % Store best threshold results
    bestResults(i).imageName = imageFiles{i};
    bestResults(i).threshold = bestThreshold;
    bestResults(i).binaryImg = bestBinaryImg;
    bestResults(i).diffSum = minDiffSum;
    bestResults(i).accuracy = bestAccuracy;
    bestResults(i).errorRate = bestErrorRate;
    
    fprintf('Best Threshold: %.3f | Accuracy: %.2f%% | Error Rate: %.2f%%\n', ...
            bestThreshold, bestAccuracy, bestErrorRate);
    fprintf('Improvement over Otsu: %.2f%% reduction in error\n', ...
            ((otsuResults(i).errorRate - bestErrorRate) / otsuResults(i).errorRate) * 100);
end


%% 5. Plot Otsu Results and Best Results in Table
fprintf('5. Displaying results tables...\n\n');

% Display Otsu results table
fprintf('=== Otsu Threshold Results ===\n');
%% Set some length limits and formattingd...
fprintf('%-20s %-12s %-12s %-12s %-12s\n', 'Image', 'Threshold', 'Diff Sum', 'Accuracy', 'Error Rate');
fprintf('%-20s %-12s %-12s %-12s %-12s\n', repmat('-', 1, 20), repmat('-', 1, 12), ...
        repmat('-', 1, 12), repmat('-', 1, 12), repmat('-', 1, 12));
for i = 1:length(otsuResults)
    fprintf('%-20s %-12.3f %-12d %-11.2f%% %-11.2f%%\n', ...
        otsuResults(i).imageName, otsuResults(i).threshold, otsuResults(i).diffSum, ...
        otsuResults(i).accuracy, otsuResults(i).errorRate);
end

fprintf('\n=== Best Threshold Results ===\n');
fprintf('%-20s %-12s %-12s %-12s %-12s\n', 'Image', 'Threshold', 'Diff Sum', 'Accuracy', 'Error Rate');
fprintf('%-20s %-12s %-12s %-12s %-12s\n', repmat('-', 1, 20), repmat('-', 1, 12), ...
        repmat('-', 1, 12), repmat('-', 1, 12), repmat('-', 1, 12));
for i = 1:length(bestResults)
    fprintf('%-20s %-12.3f %-12d %-11.2f%% %-11.2f%%\n', ...
        bestResults(i).imageName, bestResults(i).threshold, bestResults(i).diffSum, ...
        bestResults(i).accuracy, bestResults(i).errorRate);
end

fprintf('\n=== Comparison: Otsu vs Best ===\n');
fprintf('%-20s %-15s %-15s %-15s\n', 'Image', 'Otsu Error', 'Best Error', 'Improvement');
fprintf('%-20s %-15s %-15s %-15s\n', repmat('-', 1, 20), repmat('-', 1, 15), ...
        repmat('-', 1, 15), repmat('-', 1, 15));
for i = 1:length(otsuResults)
    improvement = otsuResults(i).errorRate - bestResults(i).errorRate;
    fprintf('%-20s %-14.2f%% %-14.2f%% %-14.2f%%\n', ...
        otsuResults(i).imageName, otsuResults(i).errorRate, ...
        bestResults(i).errorRate, improvement);
end


%% 6. For my report: plot histograms with otsu level and best level 
for i = 1:length(imageFiles)
    img = images{i};
    gt = groundTruths{i};
    
    % Create comprehensive figure for each image
    figure('Name', sprintf('Analysis - %s', imageFiles{i}), ...
           'Position', [50, 50, 1400, 900]);
    
    % Original image
    subplot(3, 4, 1);
    imshow(img);
    title('Original Image', 'FontSize', 10, 'FontWeight', 'bold');
    
    % Histogram with both Otsu and Best thresholds
    subplot(3, 4, 2);
    histogram(img, 256, 'EdgeColor', 'none', 'FaceColor', [0.7 0.7 0.7]);
    hold on;
    otsuValue = otsuResults(i).threshold * 255;
    bestValue = bestResults(i).threshold * 255;
    xline(otsuValue, 'r--', 'LineWidth', 2, 'Label', sprintf('Otsu: %.1f', otsuValue));
    xline(bestValue, 'g--', 'LineWidth', 2, 'Label', sprintf('Best: %.1f', bestValue));
    hold off;
    title('Histogram with Thresholds', 'FontSize', 10, 'FontWeight', 'bold');
    xlabel('Intensity');
    ylabel('Frequency');
    legend('Pixel Distribution', 'Otsu Threshold', 'Best Threshold', 'Location', 'best');
    grid on;
    
    % Ground Truth
    subplot(3, 4, 3);
    imshow(gt);
    title('Ground Truth', 'FontSize', 10, 'FontWeight', 'bold');
    
    % Metrics comparison text
    subplot(3, 4, 4);
    axis off;
    textStr = sprintf('METRICS COMPARISON\n\n');
    textStr = [textStr sprintf('Otsu Threshold: %.3f\n', otsuResults(i).threshold)];
    textStr = [textStr sprintf('Otsu Accuracy: %.2f%%\n', otsuResults(i).accuracy)];
    textStr = [textStr sprintf('Otsu Error: %.2f%%\n\n', otsuResults(i).errorRate)];
    textStr = [textStr sprintf('Best Threshold: %.3f\n', bestResults(i).threshold)];
    textStr = [textStr sprintf('Best Accuracy: %.2f%%\n', bestResults(i).accuracy)];
    textStr = [textStr sprintf('Best Error: %.2f%%\n\n', bestResults(i).errorRate)];
    improvement = otsuResults(i).errorRate - bestResults(i).errorRate;
    textStr = [textStr sprintf('Improvement: %.2f%%', improvement)];
    text(0.1, 0.5, textStr, 'FontSize', 9, 'VerticalAlignment', 'middle', 'FontName', 'FixedWidth');
    
    % Otsu binary result
    subplot(3, 4, 5);
    imshow(otsuResults(i).binaryImg);
    title(sprintf('Otsu Result (T=%.3f)', otsuResults(i).threshold), 'FontSize', 10, 'FontWeight', 'bold');
    
    % Otsu difference image
    subplot(3, 4, 6);
    otsuDiff = xor(otsuResults(i).binaryImg, gt);
    imshow(otsuDiff);
    title(sprintf('Otsu Difference (Error=%.2f%%)', otsuResults(i).errorRate), ...
          'FontSize', 10, 'FontWeight', 'bold');
    
    % Otsu overlay
    subplot(3, 4, 7);
    otsuOverlay = cat(3, uint8(otsuResults(i).binaryImg)*255, uint8(gt)*255, uint8(~otsuDiff)*255);
    imshow(otsuOverlay);
    title('Otsu Overlay (R=Result, G=GT)', 'FontSize', 10, 'FontWeight', 'bold');
    
    % Best binary result
    subplot(3, 4, 9);
    imshow(bestResults(i).binaryImg);
    title(sprintf('Best Result (T=%.3f)', bestResults(i).threshold), 'FontSize', 10, 'FontWeight', 'bold');
    
    % Best difference image
    subplot(3, 4, 10);
    bestDiff = xor(bestResults(i).binaryImg, gt);
    imshow(bestDiff);
    title(sprintf('Best Difference (Error=%.2f%%)', bestResults(i).errorRate), ...
          'FontSize', 10, 'FontWeight', 'bold');
    
    % Best overlay
    subplot(3, 4, 11);
    bestOverlay = cat(3, uint8(bestResults(i).binaryImg)*255, uint8(gt)*255, uint8(~bestDiff)*255);
    imshow(bestOverlay);
    title('Best Overlay (R=Result, G=GT)', 'FontSize', 10, 'FontWeight', 'bold');
    
    % Error comparison
    subplot(3, 4, 12);
    errorComparison = [otsuResults(i).errorRate, bestResults(i).errorRate];
    bar(errorComparison);
    set(gca, 'XTickLabel', {'Otsu', 'Best'});
    ylabel('Error Rate (%)');
    title('Error Rate Comparison', 'FontSize', 10, 'FontWeight', 'bold');
    grid on;
    ylim([0 max(errorComparison)*1.2]);
    for j = 1:2
        text(j, errorComparison(j), sprintf('%.2f%%', errorComparison(j)), ...
             'HorizontalAlignment', 'center', 'VerticalAlignment', 'bottom');
    end
    
    fprintf('Generated figure for %s\n', imageFiles{i});
end