load dataset-letters
features = dataset.images; %Load the 26,000 1*784 image vectors
labels = dataset.labels; %Load the 26,000 label column vectors
key = dataset.key; %Alphabet for translating label nums to letters

%2: Dataset -
figure(1), colormap gray;
for i = 1:12     %For loop for 12 Images
    subplot(3, 4, i);
    im = reshape(features(i, :), [28,28]);  %Reshape 1*784 to 28*28 vector
    imagesc(im), axis off;
    title(key(labels(i, :)));     %Subtitle each subplot
end

fname = './Results';
if ~exist(fname, 'dir') 
    mkdir(fname)
    disp(['Folder "', fname, '" created.']);
else
    disp(['Folder "', fname, '" already exists.']);
end
saveas(gcf, fullfile(fname, "Dataset.png"));


%3: Data Preparation -
imf = double(features);
iml = categorical(labels); %Categorical array for labels
randp = randperm(26000); %Randomise EMNIST dataset
trfeatures = imf(randp(1:13000), :); %Training Set
tefeatures = imf(randp(13001:end), :); %Testing Set
trlabels = iml(randp(1:13000), :);
telabels = iml(randp(13001:end), :);

%4.1: Custom KNN (K-Nearest Neighbour) -
prediction1 = categorical.empty(size(tefeatures, 1), 0); %Array for prediction output of Euclidean Distance
prediction2 = categorical.empty(size(tefeatures, 1), 0); %Array for prediction output of other distance
k = 28; %Set K Parameter

for i = 1:size(tefeatures, 1) %Loop for Euclidean Distance
    comp1 = trfeatures;
    comp2 = repmat(tefeatures(i, :), [size(trfeatures, 1), 1]);
    l2 = sum((comp1-comp2).^2, 2); %Euclidean Distance
    [~, ind] = sort(l2);
    ind = ind(1:k);
    labs = trlabels(ind);
    prediction1(i, 1) = mode(labs);
    i
end
for i = 1:size(tefeatures, 1) %For loop for 
    comp1 = trfeatures;
    comp2 = repmat(tefeatures(i, :), [size(trfeatures, 1), 1]);
    l1 = sum(abs(comp1-comp2), 2); %L1 Distance Model
    [~, ind] = sort(l1);
    ind = ind(1: k);
    labs = trlabels(ind);
    prediction2(i, 1) = mode(labs);
    i
end


%4.2: Model Training with existing models
knnmodel = fitcknn(trfeatures, trlabels);
predictedknn = predict(knnmodel, tefeatures);
svmmodel = fitcecoc(trfeatures, trlabels);
predictedsvm = predict(svmmodel, tefeatures);

%4.3: Evaluation
%Sum of Correct Predictions:
correct_knn1 = sum(telabels == prediction1);
correct_knn2 = sum(telabels == prediction2);
correct_knnmodel = sum(telabels == predictedknn);
correct_svmmodel = sum(telabels == predictedsvm);

%Accuracy %
accuracy_knn1 = correct_knn1/size(telabels, 1);
accuracy_knn2 = correct_knn2/size(telabels, 1);
accuracy_knnmodel = correct_knnmodel/size(telabels, 1);
accuracy_svmmodel = correct_svmmodel/size(telabels, 1);

%Confusion Matrix
figure(2)
subplot(2, 2, 1);
knn1CM = confusionchart(telabels, prediction1);
title("KNN with Euclidean Distance");
subplot(2, 2, 2);
knn2CM = confusionchart(telabels, prediction2); 
title("KNN with L1 Distance");
subplot(2, 2, 3);
knnmodelCM = confusionchart(telabels, predictedknn);
title("MatLab KNN");
subplot(2, 2, 4);
svmmodelCM = confusionchart(telabels, predictedsvm);
title("MatLab SVM");


fname = './Results';
if ~exist(fname, 'dir') 
    mkdir(fname)
    disp(['Folder "', fname, '" created.']);
else
    disp(['Folder "', fname, '" already exists.']);
end
saveas(gcf, fullfile(fname, "Confusion.png"));
