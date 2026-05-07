%% Enhanced Real-time Scene Understanding with YOLOv8
% Main application class for real-time object detection and scene understanding
% with object logging to text files

classdef SceneUnderstandingApp < handle
    properties
        % Camera and detector
        camera
        detector
        classNames
        numClasses
        
        % Configuration
        confidenceThreshold = 0.5
        frameRate = 20 % FPS
        speechEnabled = true
        recordingEnabled = false
        loggingEnabled = true % Enable/disable logging
        
        % Scene understanding
        sceneKeywords
        contextMemory
        sceneHistory
        objectTracker
        
        % Visualization
        figureHandle
        videoWriter
        statsDisplay
        frameSize = [480 640] % Default frame size [height width]
        
        % Temporal tracking
        frameCount = 0
        prevLabels = {}
        sceneStableCount = 0
        fps = 0
        fpsTimer
        
        % Logging properties
        logFileHandle      % File handle for current log
        logFileName        % Name of current log file
        logSessionID       % Unique session identifier
        detectionLog       % Store detections for current session
        objectCountHistory % Track object counts over time
    end
    
    methods
        %% Constructor
        function obj = SceneUnderstandingApp(modelName, cameraID)
            % Initialize the application
            if nargin < 1
                modelName = 'yolov8n';
            end
            if nargin < 2
                cameraID = 1;
            end
            
            % Add paths (with error handling)
            obj.addPaths();
            
            % Initialize components in correct order
            obj.initializeCamera(cameraID);
            obj.initializeDetector(modelName);
            obj.initializeSceneKeywords();
            obj.initializeTracking();
            obj.initializeVisualization();
            
            % Initialize logging
            obj.initializeLogging();
            
            disp('=================================');
            disp('Scene Understanding App initialized successfully!');
            disp('=================================');
            disp('Controls:');
            disp('  - Press ''s'' to toggle speech');
            disp('  - Press ''r'' to start/stop recording');
            disp('  - Press ''l'' to toggle logging');
            disp('  - Press ''q'' to quit');
            disp('=================================');
        end
        
        %% Initialize logging
        function initializeLogging(obj)
            % Create logs directory if it doesn't exist
            if ~exist('logs', 'dir')
                mkdir('logs');
                disp('Created logs directory');
            end
            
            % Generate unique session ID
            obj.logSessionID = datestr(now, 'yyyymmdd_HHMMSS');
            
            % Create log filename with timestamp
            obj.logFileName = fullfile('logs', sprintf('detection_log_%s.txt', obj.logSessionID));
            
            % Initialize detection log structure
            obj.detectionLog = struct(...
                'timestamp', {}, ...
                'frame', {}, ...
                'objects', {}, ...
                'counts', {}, ...
                'scene_description', {});
            
            % Initialize object count history
            obj.objectCountHistory = {};
            
            % Open log file for writing
            try
                obj.logFileHandle = fopen(obj.logFileName, 'w');
                if obj.logFileHandle == -1
                    error('Cannot create log file');
                end
                
                % Write header to log file
                obj.writeLogHeader();
                disp(['Logging enabled. Log file: ', obj.logFileName]);
            catch ME
                warning('Failed to create log file: %s', ME.message);
                obj.loggingEnabled = false;
                obj.logFileHandle = -1;
            end
        end
        
        %% Write log header
        function writeLogHeader(obj)
            if obj.logFileHandle == -1
                return;
            end
            
            fprintf(obj.logFileHandle, '========================================\n');
            fprintf(obj.logFileHandle, 'Object Detection Log\n');
            fprintf(obj.logFileHandle, 'Session ID: %s\n', obj.logSessionID);
            fprintf(obj.logFileHandle, 'Start Time: %s\n', datestr(now, 'yyyy-mm-dd HH:MM:SS'));
            fprintf(obj.logFileHandle, 'Model: %s\n', 'YOLOv8n');
            fprintf(obj.logFileHandle, 'Confidence Threshold: %.2f\n', obj.confidenceThreshold);
            fprintf(obj.logFileHandle, '========================================\n\n');
            
            % Create table header
            fprintf(obj.logFileHandle, '%-12s | %-8s | %-40s | %-20s | %s\n', ...
                'Timestamp', 'Frame', 'Detected Objects', 'Counts', 'Scene Description');
            fprintf(obj.logFileHandle, '%s\n', repmat('-', 1, 120));
        end
        
        %% Log detection results
        function logDetection(obj, labels, scores, sceneDesc)
            if ~obj.loggingEnabled || obj.logFileHandle == -1
                return;
            end
            
            % Get current timestamp
            timestamp = datestr(now, 'HH:MM:SS');
            
            % Process detected objects
            if ~isempty(labels)
                % Count unique objects
                [uniqueLabels, ~, ic] = unique(labels);
                counts = accumarray(ic, 1);
                
                % Create objects string
                objectsStr = strjoin(uniqueLabels, ', ');
                if length(objectsStr) > 40
                    objectsStr = [objectsStr(1:37), '...'];
                end
                
                % Create counts string
                countsStr = sprintf('%d ', counts);
                
                % Log to file
                fprintf(obj.logFileHandle, '%-12s | %-8d | %-40s | %-20s | %s\n', ...
                    timestamp, obj.frameCount, ...
                    objectsStr, countsStr, sceneDesc);
                
                % Also log detailed information
                fprintf(obj.logFileHandle, '  Detailed: ');
                for i = 1:length(uniqueLabels)
                    % Get average confidence for this object type
                    objIndices = find(ic == i);
                    if ~isempty(objIndices)
                        avgConfidence = mean(scores(objIndices));
                        fprintf(obj.logFileHandle, '%s(%d,%.1f%%) ', ...
                            uniqueLabels{i}, counts(i), avgConfidence*100);
                    end
                end
                fprintf(obj.logFileHandle, '\n');
                
                % Store in detection log structure
                obj.detectionLog{end+1} = struct(...
                    'timestamp', timestamp, ...
                    'frame', obj.frameCount, ...
                    'objects', {uniqueLabels}, ...
                    'counts', counts, ...
                    'scene_description', sceneDesc);
                
                % Update object count history
                obj.updateObjectCountHistory(uniqueLabels, counts);
                
                % Periodic summary logging (every 100 frames)
                if mod(obj.frameCount, 100) == 0 && obj.frameCount > 0
                    obj.writePeriodicSummary();
                end
            else
                % Log empty detection
                fprintf(obj.logFileHandle, '%-12s | %-8d | %-40s | %-20s | %s\n', ...
                    timestamp, obj.frameCount, 'None', '0', 'No objects detected');
            end
            
            % Flush the file buffer
            fprintf(obj.logFileHandle, '\n');
        end
        
        %% Update object count history
        function updateObjectCountHistory(obj, uniqueLabels, counts)
            % Track counts over time for analysis
            if isempty(obj.objectCountHistory)
                obj.objectCountHistory = {};
            end
            
            for i = 1:length(uniqueLabels)
                label = uniqueLabels{i};
                count = counts(i);
                
                % Find or create entry for this object type
                found = false;
                for j = 1:size(obj.objectCountHistory, 1)
                    if j <= size(obj.objectCountHistory, 1) && ...
                       ~isempty(obj.objectCountHistory{j, 1}) && ...
                       strcmp(obj.objectCountHistory{j, 1}, label)
                        obj.objectCountHistory{j, 2} = [obj.objectCountHistory{j, 2}, count];
                        obj.objectCountHistory{j, 3} = [obj.objectCountHistory{j, 3}, obj.frameCount];
                        found = true;
                        break;
                    end
                end
                
                if ~found
                    obj.objectCountHistory(end+1, :) = {label, [count], [obj.frameCount]};
                end
            end
        end
        
        %% Write periodic summary
        function writePeriodicSummary(obj)
            if obj.logFileHandle == -1
                return;
            end
            
            fprintf(obj.logFileHandle, '\n--- Periodic Summary at Frame %d ---\n', obj.frameCount);
            fprintf(obj.logFileHandle, 'Time Elapsed: %.1f seconds\n', obj.frameCount / obj.frameRate);
            
            % Calculate statistics for each object type
            fprintf(obj.logFileHandle, 'Object Statistics:\n');
            if ~isempty(obj.objectCountHistory)
                for i = 1:size(obj.objectCountHistory, 1)
                    label = obj.objectCountHistory{i, 1};
                    counts = obj.objectCountHistory{i, 2};
                    
                    if length(counts) > 1
                        avgCount = mean(counts);
                        maxCount = max(counts);
                        minCount = min(counts);
                        fprintf(obj.logFileHandle, '  %s: avg=%.1f, max=%d, min=%d, appearances=%d\n', ...
                            label, avgCount, maxCount, minCount, length(counts));
                    end
                end
            end
            fprintf(obj.logFileHandle, '------------------------------------\n\n');
        end
        
        %% Export detection log to CSV
        function exportToCSV(obj)
            if isempty(obj.detectionLog)
                disp('No detections to export');
                return;
            end
            
            % Create CSV filename
            csvFileName = fullfile('logs', sprintf('detection_log_%s.csv', obj.logSessionID));
            
            try
                % Open CSV file
                csvHandle = fopen(csvFileName, 'w');
                
                % Write header
                fprintf(csvHandle, 'Timestamp,Frame,Object,Count,SceneDescription\n');
                
                % Write data
                for i = 1:length(obj.detectionLog)
                    entry = obj.detectionLog{i};
                    for j = 1:length(entry.objects)
                        fprintf(csvHandle, '%s,%d,%s,%d,%s\n', ...
                            entry.timestamp, entry.frame, ...
                            entry.objects{j}, entry.counts(j), ...
                            entry.scene_description);
                    end
                end
                
                fclose(csvHandle);
                disp(['CSV exported: ', csvFileName]);
                
            catch ME
                warning('Failed to export CSV: %s', ME.message);
            end
        end
        
        %% Generate detection report
        function generateReport(obj)
            % Create a comprehensive report
            reportFileName = fullfile('logs', sprintf('detection_report_%s.txt', obj.logSessionID));
            
            try
                reportHandle = fopen(reportFileName, 'w');
                
                % Write report header
                fprintf(reportHandle, 'DETECTION REPORT\n');
                fprintf(reportHandle, '================\n\n');
                fprintf(reportHandle, 'Session: %s\n', obj.logSessionID);
                fprintf(reportHandle, 'Date: %s\n', datestr(now, 'yyyy-mm-dd'));
                fprintf(reportHandle, 'Total Frames Processed: %d\n', obj.frameCount);
                fprintf(reportHandle, 'Detections Logged: %d\n\n', length(obj.detectionLog));
                
                % Object frequency analysis
                fprintf(reportHandle, 'OBJECT FREQUENCY ANALYSIS\n');
                fprintf(reportHandle, '--------------------------\n');
                
                % Count total appearances per object
                objectFrequency = containers.Map();
                for i = 1:length(obj.detectionLog)
                    entry = obj.detectionLog{i};
                    for j = 1:length(entry.objects)
                        objName = entry.objects{j};
                        if isKey(objectFrequency, objName)
                            objectFrequency(objName) = objectFrequency(objName) + 1;
                        else
                            objectFrequency(objName) = 1;
                        end
                    end
                end
                
                % Sort and display frequencies
                if objectFrequency.Count > 0
                    objNames = keys(objectFrequency);
                    frequencies = cell2mat(values(objectFrequency));
                    [sortedFreq, sortIdx] = sort(frequencies, 'descend');
                    
                    for i = 1:length(sortedFreq)
                        fprintf(reportHandle, '%s: %d appearances (%.1f%% of frames)\n', ...
                            objNames{sortIdx(i)}, sortedFreq(i), ...
                            (sortedFreq(i)/length(obj.detectionLog))*100);
                    end
                end
                
                % Scene description analysis
                fprintf(reportHandle, '\nSCENE DESCRIPTION SAMPLES\n');
                fprintf(reportHandle, '-------------------------\n');
                
                % Show unique scene descriptions
                if ~isempty(obj.detectionLog)
                    step = max(1, floor(length(obj.detectionLog)/10));
                    for i = 1:step:length(obj.detectionLog)
                        fprintf(reportHandle, 'Frame %d: %s\n', ...
                            obj.detectionLog{i}.frame, ...
                            obj.detectionLog{i}.scene_description);
                    end
                end
                
                fclose(reportHandle);
                disp(['Report generated: ', reportFileName]);
                
            catch ME
                warning('Failed to generate report: %s', ME.message);
            end
        end
        
        %% Main run loop
        function run(obj)
            % Main processing loop
            obj.fpsTimer = tic;
            
            % Set up figure callbacks
            set(obj.figureHandle, 'KeyPressFcn', @obj.keyPressCallback);
            
            while ishandle(obj.figureHandle)
                try
                    % Capture and process frame
                    obj.processFrame();
                    
                    % Update display
                    obj.updateDisplay();
                    
                    % Control frame rate
                    obj.controlFrameRate();
                    
                catch ME
                    disp(['Error in main loop: ', ME.message]);
                    break;
                end
            end
            
            % Cleanup
            obj.cleanup();
        end
        
        %% Frame processing
        function processFrame(obj)
            % Capture frame
            I = snapshot(obj.camera);
            obj.frameCount = obj.frameCount + 1;
            
            % Update frame size based on actual image
            obj.frameSize = size(I);
            
            % Perform detection (you need to implement or have detectYOLOv8 function)
            [bboxes, scores, labelIds] = obj.detectYOLOv8(I);
            
            % Map labelIds to labels
            if ~isempty(labelIds) && ~isempty(obj.classNames)
                labels = obj.classNames(labelIds);
            else
                labels = {};
            end
            
            % Filter low confidence detections
            if ~isempty(scores)
                highConfidence = scores > obj.confidenceThreshold;
                bboxes = bboxes(highConfidence, :);
                scores = scores(highConfidence);
                if ~isempty(labels)
                    labels = labels(highConfidence);
                end
            end
            
            % Update object tracker
            obj.updateTracking(bboxes, scores, labels);
            
            % Generate scene understanding
            sceneDesc = obj.generateSceneDescription(labels, scores);
            
            % Log detection results
            obj.logDetection(labels, scores, sceneDesc);
            
            % Store in history
            obj.updateHistory(labels, sceneDesc);
            
            % Check for significant scene changes
            if obj.isSceneChanged(obj.prevLabels, labels)
                obj.handleSceneChange(labels, sceneDesc);
            end
            
            % Update previous labels
            obj.prevLabels = labels;
            
            % Store processed frame for display
            obj.contextMemory.currentFrame = I;
            obj.contextMemory.currentBBoxes = bboxes;
            obj.contextMemory.currentScores = scores;
            obj.contextMemory.currentLabels = labels;
            obj.contextMemory.currentSceneDesc = sceneDesc;
            
            % Update FPS
            obj.updateFPS();
        end
        
        %% Dummy detection function - replace with actual YOLOv8 detection
        function [bboxes, scores, labelIds] = detectYOLOv8(obj, I)
            % This is a placeholder. You need to implement actual YOLOv8 detection
            % or load your trained model here
            
            % For testing purposes, return empty detections
            bboxes = [];
            scores = [];
            labelIds = [];
            
            % Uncomment below when you have the actual detection function
            % [bboxes, scores, labelIds] = detectYOLOv8(obj.detector, I, obj.numClasses, 'auto');
        end
        
        %% Scene description generation
        function desc = generateSceneDescription(obj, labels, scores)
            if isempty(labels)
                desc = "Empty scene";
                return;
            end
            
            % Get unique objects and counts
            [uniqueLabels, ~, ic] = unique(labels);
            counts = accumarray(ic, 1);
            
            % Group by categories
            categories = containers.Map();
            for i = 1:length(uniqueLabels)
                label = char(uniqueLabels(i));
                if isKey(obj.sceneKeywords, label)
                    category = obj.sceneKeywords(label);
                else
                    category = 'objects';
                end
                
                if isKey(categories, category)
                    categories(category) = categories(category) + counts(i);
                else
                    categories(category) = counts(i);
                end
            end
            
            % Generate contextual description
            descParts = {};
            catKeys = keys(categories);
            
            for i = 1:length(catKeys)
                cat = catKeys{i};
                count = categories(cat);
                
                % Add context-aware descriptions
                descParts{end+1} = obj.getContextualCount(cat, count);
            end
            
            % Combine parts with natural language
            desc = obj.combineDescriptionParts(descParts);
            
            % Add scene type inference
            sceneType = obj.inferSceneType(labels);
            if ~isempty(sceneType)
                desc = sprintf("%s. This appears to be a %s scene.", ...
                    desc, sceneType);
            end
        end
        
        %% Contextual count description
        function desc = getContextualCount(obj, category, count)
            switch category
                case 'people'
                    if count == 1
                        desc = '1 person';
                    else
                        desc = sprintf('%d people', count);
                    end
                case 'vehicles'
                    if count == 1
                        desc = '1 vehicle';
                    else
                        desc = sprintf('%d vehicles', count);
                    end
                case 'animals'
                    if count == 1
                        desc = '1 animal';
                    else
                        desc = sprintf('%d animals', count);
                    end
                otherwise
                    if count == 1
                        % Remove 's' for singular
                        if ~isempty(category) && length(category) > 1
                            singular = category(1:end-1);
                        else
                            singular = category;
                        end
                        desc = sprintf('1 %s', singular);
                    else
                        desc = sprintf('%d %s', count, category);
                    end
            end
        end
        
        %% Combine description parts
        function desc = combineDescriptionParts(obj, parts)
            if isempty(parts)
                desc = "Empty scene";
                return;
            end
            
            if length(parts) == 1
                desc = sprintf("Scene contains %s", parts{1});
            elseif length(parts) == 2
                desc = sprintf("Scene contains %s and %s", parts{1}, parts{2});
            else
                lastPart = parts{end};
                parts(end) = [];
                desc = sprintf("Scene contains %s, and %s", ...
                    strjoin(parts, ', '), lastPart);
            end
        end
        
        %% Infer scene type
        function sceneType = inferSceneType(obj, labels)
            % Simple scene type inference based on detected objects
            if isempty(labels)
                sceneType = '';
                return;
            end
            
            labelSet = unique(labels);
            
            if any(ismember(labelSet, {'dining table', 'chair', 'bottle'}))
                sceneType = 'indoor dining';
            elseif any(ismember(labelSet, {'tv', 'laptop', 'remote'}))
                sceneType = 'living room';
            elseif any(ismember(labelSet, {'car', 'truck', 'bus'}))
                sceneType = 'traffic';
            elseif any(ismember(labelSet, {'person'})) && length(labelSet) > 3
                sceneType = 'crowded';
            else
                sceneType = '';
            end
        end
        
        %% Handle scene change
        function handleSceneChange(obj, labels, sceneDesc)
            % Significant scene change detected
            if ~isempty(labels) && obj.speechEnabled
                % Create spoken description
                [uniqueLabels, ~, ic] = unique(labels);
                counts = accumarray(ic, 1);
                spokenText = obj.generateSpokenDescription(uniqueLabels, counts);
                
                % Speak description
                obj.speakDescription(spokenText);
                disp(['Scene changed: ', spokenText]);
            end
        end
        
        %% Generate spoken description
        function spoken = generateSpokenDescription(obj, uniqueLabels, counts)
            if isempty(uniqueLabels)
                spoken = "No objects detected";
                return;
            end
            
            items = {};
            for i = 1:length(uniqueLabels)
                label = char(uniqueLabels(i));
                count = counts(i);
                
                if count == 1
                    % Handle a/an based on vowel sound
                    if ~isempty(label) && any(strcmpi(label(1), {'a','e','i','o','u'}))
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
                spoken = sprintf('I can see %s, and %s', ...
                    strjoin(items, ', '), lastItem);
            end
        end
        
        %% Text-to-speech
        function speakDescription(obj, text)
            % Use system TTS
            try
                if ispc
                    % Windows
                    NET.addAssembly('System.Speech');
                    tts = System.Speech.Synthesis.SpeechSynthesizer();
                    tts.Speak(char(text));
                elseif ismac
                    % Mac
                    system(['say "', char(text), '" &']);
                elseif isunix
                    % Linux (requires espeak)
                    system(['espeak "', char(text), '" &']);
                end
            catch
                % TTS failed, just continue
            end
        end
        
        %% Update tracking
        function updateTracking(obj, bboxes, scores, labels)
            % Simple tracking by position
            if isempty(obj.objectTracker)
                obj.objectTracker = struct('bboxes', {}, 'labels', {}, 'age', {});
            end
            
            % Update existing tracks or create new ones
            if ~isempty(bboxes)
                for i = 1:size(bboxes, 1)
                    % Check if similar bbox exists from previous frame
                    matched = false;
                    for j = 1:length(obj.objectTracker)
                        if obj.objectTracker(j).age < 10 % Consider recent tracks
                            prevBbox = obj.objectTracker(j).bboxes(end, :);
                            if obj.bboxOverlap(bboxes(i,:), prevBbox) > 0.3
                                % Update existing track
                                obj.objectTracker(j).bboxes(end+1, :) = bboxes(i,:);
                                if ~isempty(labels)
                                    obj.objectTracker(j).labels{end+1} = labels{i};
                                end
                                obj.objectTracker(j).age = obj.objectTracker(j).age + 1;
                                matched = true;
                                break;
                            end
                        end
                    end
                    
                    if ~matched && ~isempty(labels)
                        % Create new track
                        newTrack.bboxes = bboxes(i,:);
                        newTrack.labels = {labels{i}};
                        newTrack.age = 1;
                        if isempty(obj.objectTracker)
                            obj.objectTracker = newTrack;
                        else
                            obj.objectTracker(end+1) = newTrack;
                        end
                    end
                end
            end
            
            % Remove old tracks
            if ~isempty(obj.objectTracker)
                obj.objectTracker = obj.objectTracker([obj.objectTracker.age] < 30);
            end
        end
        
        %% Bounding box overlap
        function overlap = bboxOverlap(obj, bbox1, bbox2)
            % Calculate intersection over union
            x1 = max(bbox1(1), bbox2(1));
            y1 = max(bbox1(2), bbox2(2));
            x2 = min(bbox1(1)+bbox1(3), bbox2(1)+bbox2(3));
            y2 = min(bbox1(2)+bbox1(4), bbox2(2)+bbox2(4));
            
            intersection = max(0, x2-x1) * max(0, y2-y1);
            area1 = bbox1(3) * bbox1(4);
            area2 = bbox2(3) * bbox2(4);
            union = area1 + area2 - intersection;
            
            overlap = intersection / (union + eps);
        end
        
        %% Update history
        function updateHistory(obj, labels, sceneDesc)
            % Store scene history
            obj.sceneHistory{end+1} = struct(...
                'frame', obj.frameCount, ...
                'labels', {labels}, ...
                'description', sceneDesc, ...
                'timestamp', datetime('now'));
            
            % Keep only last 100 frames
            if length(obj.sceneHistory) > 100
                obj.sceneHistory(1) = [];
            end
        end
        
        %% Scene change detection
        function changed = isSceneChanged(obj, prevLabels, currLabels)
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
                
                % Also check if number of objects changed significantly
                if ~changed && length(prevLabels) ~= length(currLabels)
                    changed = abs(length(prevLabels) - length(currLabels)) > 2;
                end
            end
        end
        
        %% Update FPS calculation
        function updateFPS(obj)
            elapsed = toc(obj.fpsTimer);
            if elapsed > 0
                obj.fps = 1 / elapsed;
            end
            obj.fpsTimer = tic;
        end
        
        %% Control frame rate
        function controlFrameRate(obj)
            % Simple frame rate control
            elapsed = toc(obj.fpsTimer);
            targetTime = 1/obj.frameRate;
            if elapsed < targetTime
                pause(targetTime - elapsed);
            end
        end
        
        %% Update display
        function updateDisplay(obj)
            I = obj.contextMemory.currentFrame;
            bboxes = obj.contextMemory.currentBBoxes;
            scores = obj.contextMemory.currentScores;
            labels = obj.contextMemory.currentLabels;
            sceneDesc = obj.contextMemory.currentSceneDesc;
            
            % Create annotated image
            if ~isempty(labels) && ~isempty(bboxes)
                % Create annotations for each detection
                annotations = arrayfun(@(i) ...
                    sprintf('%s: %.1f%%', labels{i}, scores(i)*100), ...
                    1:length(labels), 'UniformOutput', false);
                
                Iout = insertObjectAnnotation(I, 'rectangle', bboxes, ...
                    annotations, 'FontSize', 12, 'LineWidth', 2, ...
                    'Color', 'yellow', 'TextColor', 'black');
                
                % Add scene description
                Iout = insertText(Iout, [10 30], sceneDesc, ...
                    'FontSize', 16, 'BoxColor', 'black', ...
                    'TextColor', 'white', 'BoxOpacity', 0.7);
            else
                Iout = insertText(I, [10 30], 'No objects detected', ...
                    'FontSize', 16, 'BoxColor', 'black', ...
                    'TextColor', 'white', 'BoxOpacity', 0.7);
            end
            
            % Add HUD overlay
            Iout = obj.addHUDOverlay(Iout);
            
            % Display
            if ishandle(obj.statsDisplay.axes)
                imshow(Iout, 'Parent', obj.statsDisplay.axes);
                title(obj.statsDisplay.axes, ...
                    sprintf('Frame: %d | FPS: %.1f | Scene Understanding', ...
                    obj.frameCount, obj.fps));
            end
            
            % Update statistics panel
            obj.updateStatsPanel();
            
            drawnow;
            
            % Record if enabled
            if obj.recordingEnabled && ~isempty(obj.videoWriter)
                writeVideo(obj.videoWriter, Iout);
            end
        end
        
        %% HUD overlay
        function Iout = addHUDOverlay(obj, I)
            % Add status information
            statusText = sprintf(...
                'Speech: %s | Recording: %s | Logging: %s | Press ''q'' to quit', ...
                obj.getYesNo(obj.speechEnabled), ...
                obj.getYesNo(obj.recordingEnabled), ...
                obj.getYesNo(obj.loggingEnabled));
            
            Iout = insertText(Iout, [10 10], statusText, ...
                'FontSize', 12, 'BoxColor', 'black', ...
                'TextColor', 'white', 'BoxOpacity', 0.5);
            
            % Add object count
            if ~isempty(obj.contextMemory.currentLabels)
                [uniqueLabels, ~, idx] = unique(obj.contextMemory.currentLabels);
                counts = accumarray(idx, 1);
                
                countText = '';
                for i = 1:min(5, length(uniqueLabels)) % Show max 5 items
                    countText = sprintf('%s%s: %d  ', countText, ...
                        uniqueLabels{i}, counts(i));
                end
                
                if ~isempty(countText)
                    Iout = insertText(Iout, [10 size(I,1)-30], countText, ...
                        'FontSize', 12, 'BoxColor', 'black', ...
                        'TextColor', 'white', 'BoxOpacity', 0.5);
                end
            end
            
            % Add logging indicator
            if obj.loggingEnabled
                Iout = insertText(Iout, [size(I,2)-150 10], 'LOGGING', ...
                    'FontSize', 14, 'BoxColor', 'red', ...
                    'TextColor', 'white', 'BoxOpacity', 0.8);
            end
        end
        
        %% Update statistics panel
        function updateStatsPanel(obj)
            if isempty(obj.statsDisplay.panel) || ~ishandle(obj.statsDisplay.panel) || ...
               ~isvalid(obj.statsDisplay.axes2)
                return;
            end
            
            % Clear panel
            cla(obj.statsDisplay.axes2);
            
            % Display statistics
            if ~isempty(obj.contextMemory.currentLabels)
                [uniqueLabels, ~, idx] = unique(obj.contextMemory.currentLabels);
                counts = accumarray(idx, 1);
                
                % Create bar chart
                if length(uniqueLabels) <= 10 % Only show if not too many
                    bar(obj.statsDisplay.axes2, counts);
                    set(obj.statsDisplay.axes2, 'XTickLabel', uniqueLabels);
                    title(obj.statsDisplay.axes2, 'Object Counts');
                    xlabel(obj.statsDisplay.axes2, 'Object Type');
                    ylabel(obj.statsDisplay.axes2, 'Count');
                    xtickangle(obj.statsDisplay.axes2, 45);
                end
            end
        end
        
        %% Initialize camera
        function initializeCamera(obj, cameraID)
            try
                obj.camera = webcam(cameraID);
                % Get a test frame to determine frame size
                testFrame = snapshot(obj.camera);
                obj.frameSize = size(testFrame);
                disp(['Camera initialized: ', obj.camera.Name]);
                disp(['Frame size: ', num2str(obj.frameSize(2)), 'x', num2str(obj.frameSize(1))]);
            catch ME
                error('Failed to initialize camera: %s', ME.message);
            end
        end
        
        %% Initialize detector
        function initializeDetector(obj, modelName)
            try
                % Check if models directory exists
                if ~exist('models', 'dir')
                    warning('Models directory not found. Creating it...');
                    mkdir('models');
                end
                
                % Check if model file exists
                modelPath = fullfile('models', [modelName, '.mat']);
                if ~exist(modelPath, 'file')
                    warning('Model file %s not found. Using placeholder detector.', modelPath);
                    obj.detector = [];
                    % Load class names from helper or use default
                    try
                        obj.classNames = helper.getCOCOClassNames();
                    catch
                        % Default COCO class names (first 20 for testing)
                        obj.classNames = {'person', 'bicycle', 'car', 'motorcycle', 'airplane', ...
                            'bus', 'train', 'truck', 'boat', 'traffic light', ...
                            'fire hydrant', 'stop sign', 'parking meter', 'bench', 'bird', ...
                            'cat', 'dog', 'horse', 'sheep', 'cow'};
                    end
                else
                    data = load(modelPath);
                    obj.detector = data.yolov8Net;
                    obj.classNames = helper.getCOCOClassNames();
                end
                
                obj.numClasses = length(obj.classNames);
                disp(['Detector initialized: ', modelName]);
                disp(['Number of classes: ', num2str(obj.numClasses)]);
            catch ME
                warning('Failed to initialize detector: %s', ME.message);
                disp('Using placeholder detector with default settings.');
                obj.detector = [];
                % Default COCO class names
                obj.classNames = {'person', 'bicycle', 'car', 'motorcycle', 'airplane', ...
                    'bus', 'train', 'truck', 'boat', 'traffic light'};
                obj.numClasses = length(obj.classNames);
            end
        end
        
        %% Initialize scene keywords
        function initializeSceneKeywords(obj)
            obj.sceneKeywords = containers.Map();
            obj.sceneKeywords('person') = 'people';
            obj.sceneKeywords('car') = 'vehicles';
            obj.sceneKeywords('truck') = 'vehicles';
            obj.sceneKeywords('bus') = 'vehicles';
            obj.sceneKeywords('dog') = 'animals';
            obj.sceneKeywords('cat') = 'animals';
            obj.sceneKeywords('chair') = 'furniture';
            obj.sceneKeywords('couch') = 'furniture';
            obj.sceneKeywords('dining table') = 'furniture';
            obj.sceneKeywords('tv') = 'electronics';
            obj.sceneKeywords('laptop') = 'electronics';
            obj.sceneKeywords('cell phone') = 'electronics';
            obj.sceneKeywords('bottle') = 'containers';
            obj.sceneKeywords('cup') = 'containers';
            obj.sceneKeywords('bowl') = 'containers';
            obj.sceneKeywords('book') = 'reading material';
            obj.sceneKeywords('clock') = 'timepieces';
            obj.sceneKeywords('motorcycle') = 'vehicles';
            obj.sceneKeywords('airplane') = 'vehicles';
            obj.sceneKeywords('train') = 'vehicles';
            obj.sceneKeywords('bird') = 'animals';
        end
        
        %% Initialize tracking
        function initializeTracking(obj)
            obj.objectTracker = [];
            obj.contextMemory = struct(...
                'currentFrame', [], ...
                'currentBBoxes', [], ...
                'currentScores', [], ...
                'currentLabels', [], ...
                'currentSceneDesc', '');
            obj.sceneHistory = {};
        end
        
        %% Initialize visualization
        function initializeVisualization(obj)
            % Create main figure
            obj.figureHandle = figure('Name', 'Real-time Scene Understanding', ...
                'NumberTitle', 'off', 'Position', [100 100 1200 600], ...
                'Color', 'black', 'CloseRequestFcn', @obj.closeCallback);
            
            % Create axes for video
            obj.statsDisplay.axes = axes('Parent', obj.figureHandle, ...
                'Position', [0.05 0.05 0.6 0.9]);
            
            % Create axes for statistics
            obj.statsDisplay.axes2 = axes('Parent', obj.figureHandle, ...
                'Position', [0.7 0.55 0.25 0.4]);
            
            % Create panel for controls
            obj.statsDisplay.panel = uipanel('Parent', obj.figureHandle, ...
                'Position', [0.7 0.05 0.25 0.45], 'Title', 'Controls', ...
                'BackgroundColor', 'white');
            
            % Add buttons
            uicontrol('Parent', obj.statsDisplay.panel, 'Style', 'pushbutton', ...
                'String', 'Toggle Speech', 'Position', [10 150 120 30], ...
                'Callback', @(src, evt) obj.toggleSpeech());
            
            uicontrol('Parent', obj.statsDisplay.panel, 'Style', 'pushbutton', ...
                'String', 'Toggle Recording', 'Position', [140 150 120 30], ...
                'Callback', @(src, evt) obj.toggleRecording());
            
            uicontrol('Parent', obj.statsDisplay.panel, 'Style', 'pushbutton', ...
                'String', 'Toggle Logging', 'Position', [10 110 120 30], ...
                'Callback', @(src, evt) obj.toggleLogging());
            
            uicontrol('Parent', obj.statsDisplay.panel, 'Style', 'pushbutton', ...
                'String', 'Export CSV', 'Position', [140 110 120 30], ...
                'Callback', @(src, evt) obj.exportCSV());
            
            uicontrol('Parent', obj.statsDisplay.panel, 'Style', 'pushbutton', ...
                'String', 'Generate Report', 'Position', [10 70 120 30], ...
                'Callback', @(src, evt) obj.generateReportCallback());
            
            uicontrol('Parent', obj.statsDisplay.panel, 'Style', 'pushbutton', ...
                'String', 'Take Snapshot', 'Position', [140 70 120 30], ...
                'Callback', @(src, evt) obj.takeSnapshot());
            
            uicontrol('Parent', obj.statsDisplay.panel, 'Style', 'pushbutton', ...
                'String', 'Quit', 'Position', [75 30 120 30], ...
                'Callback', @(src, evt) obj.quit());
            
            % Add confidence slider
            uicontrol('Parent', obj.statsDisplay.panel, 'Style', 'text', ...
                'String', 'Confidence Threshold:', 'Position', [10 190 150 20], ...
                'HorizontalAlignment', 'left');
            
            uicontrol('Parent', obj.statsDisplay.panel, 'Style', 'slider', ...
                'Position', [10 170 250 20], 'Min', 0, 'Max', 1, 'Value', 0.5, ...
                'Callback', @(src, evt) obj.updateThreshold(src));
        end
        
        %% Helper functions
        function str = getYesNo(obj, value)
            if value
                str = 'ON';
            else
                str = 'OFF';
            end
        end
        
        %% Add paths
        function addPaths(obj)
            % Add necessary paths with error handling
            try
                % Check if models directory exists, create if not
                if ~exist('models', 'dir')
                    mkdir('models');
                    disp('Created models directory');
                end
                
                % Add to path
                addpath('models');
                
                % Check if logs directory exists, create if not
                if ~exist('logs', 'dir')
                    mkdir('logs');
                    disp('Created logs directory');
                end
            catch ME
                warning('Path initialization issue: %s', ME.message);
            end
        end
        
        %% Callback functions
        function keyPressCallback(obj, ~, event)
            if ~isempty(event.Key)
                switch event.Key
                    case 's'
                        obj.toggleSpeech();
                    case 'r'
                        obj.toggleRecording();
                    case 'l'
                        obj.toggleLogging();
                    case 'q'
                        obj.quit();
                end
            end
        end
        
        function toggleSpeech(obj)
            obj.speechEnabled = ~obj.speechEnabled;
            disp(['Speech ', obj.getYesNo(obj.speechEnabled)]);
        end
        
        function toggleRecording(obj)
            obj.recordingEnabled = ~obj.recordingEnabled;
            if obj.recordingEnabled
                % Start recording
                filename = fullfile('logs', sprintf('recording_%s.avi', ...
                    datestr(now, 'yyyy-mm-dd_HH-MM-SS')));
                try
                    obj.videoWriter = VideoWriter(filename);
                    open(obj.videoWriter);
                    disp(['Recording started: ', filename]);
                catch ME
                    warning('Failed to start recording: %s', ME.message);
                    obj.recordingEnabled = false;
                end
            else
                % Stop recording
                if ~isempty(obj.videoWriter)
                    try
                        close(obj.videoWriter);
                    catch
                    end
                    obj.videoWriter = [];
                    disp('Recording stopped');
                end
            end
        end
        
        function toggleLogging(obj)
            obj.loggingEnabled = ~obj.loggingEnabled;
            if obj.loggingEnabled
                obj.initializeLogging();
            else
                % Close log file if open
                if obj.logFileHandle ~= -1
                    obj.writeSessionSummary();
                    fclose(obj.logFileHandle);
                    obj.logFileHandle = -1;
                end
            end
            disp(['Logging ', obj.getYesNo(obj.loggingEnabled)]);
        end
        
        function exportCSV(obj)
            obj.exportToCSV();
        end
        
        function generateReportCallback(obj)
            obj.generateReport();
        end
        
        function takeSnapshot(obj)
            % Save current frame
            if ~isempty(obj.contextMemory) && ~isempty(obj.contextMemory.currentFrame)
                filename = fullfile('logs', sprintf('snapshot_%s.png', ...
                    datestr(now, 'yyyy-mm-dd_HH-MM-SS')));
                try
                    imwrite(obj.contextMemory.currentFrame, filename);
                    disp(['Snapshot saved: ', filename]);
                catch ME
                    warning('Failed to save snapshot: %s', ME.message);
                end
            else
                disp('No frame available for snapshot');
            end
        end
        
        function updateThreshold(obj, source)
            obj.confidenceThreshold = source.Value;
            disp(['Confidence threshold updated to: ', num2str(obj.confidenceThreshold)]);
        end
        
        function quit(obj)
            delete(obj.figureHandle);
        end
        
        function closeCallback(obj, ~, ~)
            obj.cleanup();
            delete(obj.figureHandle);
        end
        
        %% Write session summary
        function writeSessionSummary(obj)
            if obj.logFileHandle == -1
                return;
            end
            
            fprintf(obj.logFileHandle, '\n========================================\n');
            fprintf(obj.logFileHandle, 'Session Summary\n');
            fprintf(obj.logFileHandle, 'End Time: %s\n', datestr(now, 'yyyy-mm-dd HH:MM:SS'));
            fprintf(obj.logFileHandle, 'Total Frames Processed: %d\n', obj.frameCount);
            fprintf(obj.logFileHandle, 'Total Detections Logged: %d\n', length(obj.detectionLog));
            
            % Calculate unique objects detected
            allObjects = {};
            for i = 1:length(obj.detectionLog)
                entry = obj.detectionLog{i};
                if ~isempty(entry.objects)
                    allObjects = [allObjects; entry.objects(:)];
                end
            end
            if ~isempty(allObjects)
                uniqueObjects = unique(allObjects);
                fprintf(obj.logFileHandle, 'Unique Objects Detected: %d\n', length(uniqueObjects));
                if length(uniqueObjects) <= 20
                    fprintf(obj.logFileHandle, 'Objects: %s\n', strjoin(uniqueObjects, ', '));
                else
                    fprintf(obj.logFileHandle, 'First 20 objects: %s\n', strjoin(uniqueObjects(1:20), ', '));
                end
            end
            fprintf(obj.logFileHandle, '========================================\n');
        end
        
        %% Cleanup
        function cleanup(obj)
            % Stop recording if active
            if obj.recordingEnabled && ~isempty(obj.videoWriter)
                try
                    close(obj.videoWriter);
                catch
                end
            end
            
            % Close log file
            if obj.loggingEnabled && obj.logFileHandle ~= -1
                try
                    obj.writeSessionSummary();
                    fclose(obj.logFileHandle);
                    disp(['Log file saved: ', obj.logFileName]);
                    
                    % Generate final report and CSV
                    obj.generateReport();
                    obj.exportToCSV();
                catch ME
                    warning('Error during log cleanup: %s', ME.message);
                end
            end
            
            % Clear camera
            if ~isempty(obj.camera)
                try
                    clear obj.camera;
                catch
                end
            end
            
            disp('Cleanup completed. Goodbye!');
        end
    end
end