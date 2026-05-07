% Add path containing the pretrained models.
addpath('models');

WCAM = webcam(1);

% Get classnames of COCO dataset.
classNames = helper.getCOCOClassNames;
numClasses = size(classNames,1);

modelName = 'yolov8n';
% Load YOLO v8 network with custom split layers.
data = load([modelName,'.mat']);
detector = data.yolov8Net;

executionEnvironment = 'auto';

objectDetected = false;

while ~objectDetected
    % Capture image
    while(1)
    I = snapshot(WCAM);

    % Perform detection
    [bboxes, scores, labelIds] = detectYOLOv8(detector, I, numClasses, executionEnvironment);

    % Map labelIds back to labels
    labels = classNames(labelIds);

    if ~isempty(labels)
        objectDetected = true;

        % Visualize detection results
        annotations = string(labels) + ': ' + string(scores);
        Iout = insertObjectAnnotation(I, 'rectangle', bboxes, annotations);
        figure, imshow(Iout);

        % -------- Text-to-Speech Section --------
        spokenText = "Detected objects are: " + join(string(labels), ', ');
        tts(char(spokenText));

    else
        % If no object detected, speak and scan again
        tts('No objects detected. Scanning again.');
        pause(2); % small delay before next scan
    end
    end
end

% Release camera after detection
clear WCAM;
