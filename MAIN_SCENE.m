 
addpath('models');

% Initialize webcam
WCAM = webcam(1);
disp('Camera initialized. Press Ctrl+C to stop...');

% Get classnames of COCO dataset
classNames = helper.getCOCOClassNames;
numClasses = size(classNames, 1                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                       );

% Load YOLO v8 network
modelName = 'yolov8n';
data = load([modelName, '.mat']);
detector = data.yolov8Net;

% Scene context keywords for better descriptions
sceneKeywords = containers.Map();
sceneKeywords('person') = 'people';
sceneKeywords('car') = 'vehicles';
sceneKeywords('dog') = 'animals';
sceneKeywords('cat') = 'animals';
sceneKeywords('chair') = 'furniture';
sceneKeywords('dining table') = 'furniture';
sceneKeywords('tv') = 'electronics';
sceneKeywords('laptop') = 'electronics';
sceneKeywords('cell phone') = 'electronics';
sceneKeywords('bottle') = 'containers';
sceneKeywords('cup') = 'containers';
sceneKeywords('bowl') = 'containers';

% Temporal tracking variables
prevLabels = {};
frameCount = 0;
sceneStableCount = 0;

% Main loop for continuous processing
while true
    try
        % Capture frame
        I = snapshot(WCAM);
        frameCount = frameCount + 1;
        
        % Perform detection
        executionEnvironment = 'auto';
        [bboxes, scores, labelIds] = detectYOLOv8(detector, I, numClasses, executionEnvironment);
        
        % Map labelIds back to labels
        labels = classNames(labelIds);
        
        % Filter low confidence detections (optional)
        confidenceThreshold = 0.5;
        highConfidence = scores > confidenceThreshold;
        bboxes = bboxes(highConfidence, :);
        scores = scores(highConfidence);
        labels = labels(highConfidence);
        
        % Visualize detection results
        if ~isempty(labels)
            annotations = string(labels) + ': ' + string(round(scores*100,1)) + '%';
            Iout = insertObjectAnnotation(I, 'rectangle', bboxes, annotations, ...
                'FontSize', 12, 'LineWidth', 2);
            
            % Add scene description overlay
            sceneDesc = generateSceneDescription(labels, scores, sceneKeywords);
            Iout = insertText(Iout, [10 10], sceneDesc, ...
                'FontSize', 18, 'BoxColor', 'black', 'TextColor', 'white');
        else
            Iout = insertText(I, [10 10], 'No objects detected', ...
                'FontSize', 18, 'BoxColor', 'black', 'TextColor', 'white');
        end
        
        % Display frame
        imshow(Iout);
        title(sprintf('Frame: %d | Real-time Scene Understanding', frameCount));
        drawnow;
        
        % Text-to-Speech for significant scene changes
        if mod(frameCount, 30) == 0 || isSceneChanged(prevLabels, labels)
            if ~isempty(labels)
                % Count unique objects for better description
                [uniqueLabels, ~, ic] = unique(labels);
                counts = accumarray(ic, 1);
                
                % Create natural language description
                spokenText = generateSpokenDescription(uniqueLabels, counts, sceneKeywords);
                
                % Speak the description (uncomment to enable)
                % tts(char(spokenText));
                
                disp(['Scene: ', spokenText]);
                 tts(char(spokenText));
            else
                % tts('No objects detected');
                disp('Scene: No objects detected');
            end
            
            % Update tracking
            prevLabels = labels;
        end
        
        % Small delay to control frame rate
        pause(0.05);
        
    catch ME
        disp(['Error: ', ME.message]);
        break;
    end
end

% Clean up
clear WCAM;
disp('Camera released.');

% Helper function to generate scene description
function desc = generateSceneDescription(labels, scores, sceneKeywords)
    if isempty(labels)
        desc = "Empty scene";
        return;
    end
    
    % Get unique objects and counts
    [uniqueLabels, ~, ic] = unique(labels);
    counts = accumarray(ic, 1);
    
    % Build description
    descParts = {};
    
    % Group by categories
    categories = containers.Map();
    for i = 1:length(uniqueLabels)
        label = char(uniqueLabels(i));
        if isKey(sceneKeywords, label)
            category = sceneKeywords(label);
        else
            category = 'objects';
        end
        
        if isKey(categories, category)
            categories(category) = categories(category) + counts(i);
        else
            categories(category) = counts(i);
        end
    end
    
    % Generate description
    catKeys = keys(categories);
    for i = 1:length(catKeys)
        cat = catKeys{i};
        count = categories(cat);
        if count == 1
            descParts{end+1} = sprintf('1 %s', cat(1:end-1)); % Remove 's' for singular
        else
            descParts{end+1} = sprintf('%d %s', count, cat);
        end
    end
    
    % Combine parts
    if length(descParts) == 1
        desc = sprintf("Scene contains %s", descParts{1});
    elseif length(descParts) == 2
        desc = sprintf("Scene contains %s and %s", descParts{1}, descParts{2});
    else
        lastPart = descParts{end};
        descParts(end) = [];
        desc = sprintf("Scene contains %s, and %s", ...
            strjoin(descParts, ', '), lastPart);
    end
end

% Helper function to generate spoken description
function spoken = generateSpokenDescription(uniqueLabels, counts, sceneKeywords)
    if isempty(uniqueLabels)
        spoken = "No objects detected";
        return;
    end
    
    % Create more natural spoken description
    items = {};
    for i = 1:length(uniqueLabels)
        label = char(uniqueLabels(i));
        count = counts(i);
        
        if count == 1
            % Handle a/an based on vowel sound
            if any(strcmpi(label(1), {'a','e','i','o','u'}))
                items{end+1} = sprintf('an %s', label);
            else
                items{end+1} = sprintf('a %s', label);
            end
        else
            items{end+1} = sprintf('%d %ss', count, label);
        end
    end
    
    % Join naturally
    if length(items) == 1
        spoken = sprintf('I can see %s', items{1});
    elseif length(items) == 2
        spoken = sprintf('I can see %s and %s', items{1}, items{2});
    else
        lastItem = items{end};
        items(end) = [];
        spoken = sprintf('I can see %s, and %s', strjoin(items, ', '), lastItem);
    end
end

% Helper function to check if scene changed significantly
function changed = isSceneChanged(prevLabels, currLabels)
    if isempty(prevLabels) && isempty(currLabels)
        changed = false;
    elseif isempty(prevLabels) || isempty(currLabels)
        changed = true;
    else
        % Check if object types changed significantly
        prevUnique = unique(prevLabels);
        currUnique = unique(currLabels);
        
        % Scene changes if different objects appear/disappear
        changed = ~isequal(sort(prevUnique), sort(currUnique));
    end
end