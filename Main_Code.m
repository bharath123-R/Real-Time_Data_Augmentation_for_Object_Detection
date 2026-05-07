% Add path containing the pretrained models.
addpath('models');
WCAM=webcam(1);
I=snapshot(WCAM);
% Read test image.
% I = imread(fullfile('data','inputTeam.jpg'));

% Get classnames of COCO dataset.
classNames = helper.getCOCOClassNames;
numClasses = size(classNames,1);

modelName = 'yolov8n';
% Load YOLO v8 network with custom split layers.
data = load([modelName,'.mat']);
detector = data.yolov8Net;

% Perform detection using pretrained model.
executionEnvironment = 'auto';
[bboxes, scores, labelIds] = detectYOLOv8(detector, I, numClasses, executionEnvironment);

% Map labelIds back to labels.
labels = classNames(labelIds);

% Visualize detection results.
annotations = string(labels) + ': ' + string(scores);
Iout = insertObjectAnnotation(I, 'rectangle', bboxes, annotations);
figure, imshow(Iout);

% -------- Text-to-Speech Section --------
if ~isempty(labels)
    % Convert labels into a readable sentence
    spokenText = "Detected objects are: " + join(string(labels), ', ');
    tts(char(spokenText));
else
    tts('No objects detected');
end