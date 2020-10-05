%This section deals with creating an image database
%Name of the person to be added to database is entered and 100 pictures are
%taken at various face angles.

name = input('Please Enter Name: ', 's');
filename = sprintf('%s', name);
mkdir('C:\Users\Hari\Desktop\\Stuff\FaceRecog\Data', filename);
setPath = sprintf('C:\\Users\\Hari\\Desktop\\Stuff\\FaceRecog\\Data\\%s', filename);
folder = setPath;
cam = webcam('Lenovo EasyCamera');
cam.resolution = '320x240';
preview(cam);
detector = vision.CascadeObjectDetector;
rgbImage = snapshot(cam);
grayImage = rgb2gray(rgbImage);
figure;
subplot(2,1,1);
imshow(rgbImage);
subplot(2,1,2);
imshow(grayImage);
bbox = detector(grayImage);
for j = 1 : size(bbox, 1) 
        J = imcrop(grayImage, bbox(j, :));
        J = imresize(J,[250 250]);
end
for i = 1:100
    % Acquire a single image.
    rgbImage = snapshot(cam);
    % Convert RGB to grayscale.
    grayImage = rgb2gray(rgbImage);
    for j = 1 : size(bbox, 1) 
        J = imcrop(rgbImage, bbox(j, :));
        J = imresize(J,[100 100]);
    end
    baseFilename = sprintf('#%d.png', i+80);
    fullFilename = fullfile(folder, baseFilename);
    imwrite(J, fullFilename);  
end
[singleFeature, visualization] = extractHOGFeatures(J);
figure;
subplot(2,1,1);
imshow(J);
subplot(2,1,2);
plot(visualization);
clear('cam');

%% 
%This section loads the faces in the Data folder into faceDatabase
%variable. Also displays all the faces side by side and all the images of
%the first person in the Data directory.

faceDatabase = imageSet('C:\Users\Hari\Desktop\Stuff\FaceRecog\data','recursive');
figure;
montage(faceDatabase(1).ImageLocation);
title('Images of Single Face');
figure;
for i=1:size(faceDatabase,2)
    imageList(i) = faceDatabase(i).ImageLocation(5);
end
montage(imageList);

%% 
%This section first gives the entire faceDatabase an 80-20 split.
%It then extracts the features of one image of one person in the 
%faceDatabase through the Histogram of Oriented Gradients and displays it.

[training,test] = partition(faceDatabase,[0.8 0.2]);  
p = 1;
[hogF, visualization]= ...
    extractHOGFeatures(read(training(p),1));
figure;
subplot(2,1,1);imshow(read(training(p),1));title('Input Face');
subplot(2,1,2);plot(visualization);title('HoG Feature');

%% 
%This section creates the trainingFeatures matrix which is a matrix with
%the features of one image in one row spanning 4356 columns for the 100x100
%resolution of each image. It also creates the corresponding trainingLabel
%and personIndex, one to label the images and other to count the number of
%people in the database.

featureCount = 1;
trainingFeatures = zeros((size(training,2))*(training(1).Count),4356);
for i=1:size(training,2)
    for j = 1:training(i).Count
        trainingFeatures(featureCount,:) = ...
            extractHOGFeatures(read(training(i),j));
        trainingLabel{featureCount} = training(i).Description;    
        featureCount = featureCount + 1;
    end
personIndex{i} = training(i).Description;
end

%% 
%This section creates the classifier. The input are the trainingFeatures
%and trainingLabel created in the previous section. Uses the
%[fit][c][ecoc] command. [ecoc] or Error Correcting Output Codes
%is the method through which the classifier is made.

faceClassifier = fitcecoc(trainingFeatures, trainingLabel);
pause;

%% 
%This section loads the queryImageData which are the queryImages to be
%classified into the people in the main database. It follows the same
%procedure to extract features from the image(Input->Detect->Crop->Resize)
%and then predict command is used to find the best match to the image.

queryImageData = imageSet('C:\Users\Hari\Desktop\FaceRecog\queryImage','recursive');
qPath = 'C:\Users\Hari\Desktop\FaceRecog\queryImage\';
files=dir([qPath, '*.jpeg']);
detector = vision.CascadeObjectDetector;
for index = 1:(queryImageData.Count)
    this = files(index).name;
    rgbIm = imread(this);
    grayIm = rgb2gray(rgbIm);
    bbox = detector(grayIm);
    for j = 1 : size(bbox, 1) 
        K = imcrop(grayIm, bbox(j, :));
        K = imresize(K,[100 100]);
    end
    queryFeatures = extractHOGFeatures(K);
    personLabel = predict(faceClassifier,queryFeatures);
    booleanIndex = strcmp(personLabel, personIndex);
    integerIndex = find(booleanIndex);
    figure;
    subplot(2,1,1);imshow(rgbIm);title('Query Face');
    subplot(2,1,2);imshow(imresize(read(training(integerIndex),1),3));
    title('Matched Class');
end

%% 
%This section takes another image from webcam as input and extracts
%features from it to predict the person in the image.

queryCam = webcam('Lenovo EasyCamera');
queryCam.resolution = '320x240';
preview(queryCam);
queryDetector = vision.CascadeObjectDetector;
rgbIm2 = snapshot(queryCam);
grayIm2 = rgb2gray(rgbIm2);
bbox = queryDetector(grayIm2);
for in = 1 : size(bbox, 1) 
        Q = imcrop(grayIm2, bbox(in, :));
        Q = imresize(Q,[100 100]);
end
queryFeatures2 = extractHOGFeatures(Q);
personLabel = predict(faceClassifier,queryFeatures2);
booleanIndex = strcmp(personLabel, personIndex);
integerIndex = find(booleanIndex);
figure;
subplot(2,1,1);imshow(rgbIm2);title('Query Face');
subplot(2,1,2);imshow(imresize(read(training(integerIndex),1),3));title('Matched Class');
clear('queryCam');
