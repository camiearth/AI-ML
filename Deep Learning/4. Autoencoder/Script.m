% Load the dataset
[XTrainImages, yTrain] = digittrain_dataset;
[XTestImages, yTest] = digittest_dataset;

% Task 1: Train an autoencoder on a set of input images
hidden = 4; % Number of hidden nodes
autoenc = trainAutoencoder(XTrainImages, hidden, 'L2WeightRegularization', 0.004, 'SparsityRegularization', 4, 'SparsityProportion', 0.15, 'ScaleData', false);

% Task 2: Train a second autoencoder whose input is the output of the first autoencoder
features = encode(autoenc, XTrainImages);
autoenc2 = trainAutoencoder(features, hidden, 'L2WeightRegularization', 0.004, 'SparsityRegularization', 4, 'SparsityProportion', 0.15, 'ScaleData', false);

% Task 3: Train a softmax layer
SomeFeatures = encode(autoenc2, features);
softnet = trainSoftmaxLayer(SomeFeatures, yTrain, 'MaxEpochs', 400);

% Task 4: Construct a deep neural network
deepnet = stack(autoenc, autoenc2, softnet);

% Task 5: Preprocess the test set and evaluate the performance
XTest = zeros(28 * 28, numel(XTestImages));
for i = 1:numel(XTestImages)
    XTest(:, i) = XTestImages{i}(:);
end
yPredict = deepnet(XTest);

% Calculate performance metrics (e.g., accuracy)
performance = perform(deepnet, yTest, yPredict);
fprintf('Performance of the deep network: %f\n', performance);

% Task 6: Retrain the deep network and evaluate performance again
XTrain = zeros(28 * 28, numel(XTrainImages));
for i = 1:numel(XTrainImages)
    XTrain(:, i) = XTrainImages{i}(:);
end

deepnet = train(deepnet, XTrain, yTrain);
yPredict2 = deepnet(XTest);

% Calculate performance metrics after retraining
performance2 = perform(deepnet, yTest, yPredict2);
fprintf('Performance of the deep network after retraining: %f\n', performance2);