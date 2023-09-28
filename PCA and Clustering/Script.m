%% 1) Principal Component Analysis
%--------------------------------------------------------

%% 1.1 Preprocessing 
% 1.1.1 Loading of the dataset 

data = csvread('chronic_kidney_disease_ad.csv'); 

% 1.1.2
% Replacement of missing values (NaN) with the mean value 

cols = size(data, 2); % number of columns in data
for i=1:cols
    f = data(:,i);                                             % selection of the i-th column of data that represents a single feature f
    index_nan = isnan(f);                              % logical vector containing 1 in the positions of NaN elements contained in f
    mean_value = mean(f(~index_nan));     % mean value of the non-NaN elements contained in f
    f(index_nan) = mean_value;                  % assignment of the mean_value to all the NaN elements contained in f
    data(:,i) = f;                                             % assignment of f to the i-th columns of D
end

% 1.1.3
% Division of the dataset in a feature set and a label vector
feature_set = data(:,1:end-1); % columns 1 to 24 of data
label_vector = data(:,end);      % last column (25) of data

% Scatter plots between some numerical features
% Numerical features: age, blood pressure, blood glucose random, blood
% urea, serum creatinine, sodium, potassium, hemoglobin, packed cell
% volume, red blood cell count

figure
sgtitle('Scatter plot between some numerical features colored according to class label')
subplot(2, 2, 1)
scatter(feature_set(:, 11), feature_set(:, 12), 50, label_vector, 'filled'), xlabel('blood urea'), ylabel('serum creatinine')
subplot(2, 2, 2)
scatter(feature_set(:, 1), feature_set(:, 16), 50, label_vector, 'filled'), xlabel('age'), ylabel('packed cell volume')
subplot(2, 2, 3)
scatter(feature_set(:, 15), feature_set(:, 18), 50, label_vector, 'filled'), xlabel('hemoglobin'), ylabel('red blood cell count')
subplot(2, 2, 4)
scatter(feature_set(:, 2), feature_set(:, 13), 50, label_vector, 'filled'), xlabel('blood pressure'), ylabel('sodium')

%% 1.2 Implementing PCA
%--------------------------------------------------------

% 1.2.2
% Manually standardize the features to have zero mean and unit variance
feature_means = mean(feature_set);
feature_stds = std(feature_set);
feature_set_standard = (feature_set - feature_means) ./ feature_stds;

% Compute the covariance matrix
covariance_matrix = cov(feature_set_standard);

% Perform SVD
[U, S, ~] = svd(covariance_matrix);

% Sort the singular values and eigenvectors
[eigenvalues, idx] = sort(diag(S), 'descend');
eigenvectors = U(:, idx);

% Project the data onto the principal components
new_feature_set = feature_set_standard * eigenvectors;

% Visualizing the first two dimensions of the transformed data
figure
hold on
scatter(new_feature_set(label_vector==0, 1), new_feature_set(label_vector==0, 2), 'g', 'o', 'filled', 'DisplayName', 'Healthy');
scatter(new_feature_set(label_vector==1, 1), new_feature_set(label_vector==1, 2), 'r', 'o', 'filled', 'DisplayName', 'Pathological');
xlabel('Dimension 1');
ylabel('Dimension 2');
title('Scatter plot with dimensionality reduction');
legend('Location', 'Best');
hold off

% 1.2.3
% Recover of the data in the high dimensional space by using 10 components
recovered_feature_set = new_feature_set(:, 1:10);
reconstruction_error = 1 - sum(eigenvalues(1:10)) / sum(eigenvalues);
% The reconstruction error is calculated as the fraction of unexplained variance

%% 1.3 Application to face image data
%--------------------------------------------------------
% 1.3.1
% Loading the dataset 'faces.mat' and visualize the first 100 faces

load('faces.mat')
figure()
[h, display_array] = displayImageData(X(1:100,:), 32);

% 1.3.2
% Normalization of the dataset to have zero mean
mean_values = mean(X);
X_norm = (X-mean_values);

% PCA
C = cov(X_norm);
[V,D] = eigs(C,100);
% Justify whether standardization (i.e. rescaling to have unit variance) is
% needed.
% We don't need to standardize because the columns of X have a paragonable
% range of values (the grayscale).

% 1.3.3
% Visualization of the first 36 principal components that describe the 
% largest variations
new_X = X_norm*V(:,1:36);
new_X = fliplr(new_X);
figure()
[h, display_array] = displayImageData(new_X(1:100,:), 6);

% 1.3.4
% Project the face dataset onto the first 100 principal components
new_X = X_norm*V(:,1:100);
new_X = fliplr(new_X);

% 1.3.5
% Visualization the first 100 reconstructed faces
[recover_dataset, perc_variance] = pca_function(X_norm');
recover_dataset = recover_dataset';
figure()
[h, display_array] = displayImageData(recover_dataset(1:100,:), 32);

%% 2) Clustering: application to cardiac data
%--------------------------------------------------------

%% 2.1 Extracting features

% 2.1.1
% Loading of the dataset 'cardiacShapes.mat'
load('cardiacShapes.mat')

% 2.1.2
% Visualization of the myocardial shapes for all subjects
% The first 71 acquisitions are in class 1, while the remaining 35
% acquisitions are in class 2

% Division of x coordinate and y coordinate for each acquisition
x_shape=zeros(106,67);
y_shape=zeros(106,67);
for p=1:106
    for i=1:67
        x_shape(p,i) = shapeData(p,1,i);
        y_shape(p,i) = shapeData(p,2,i);
    end
end

figure
plot(x_shape(1:71,:)',y_shape(1:71,:)','b') % plot of the first 71 acquisitions (healthy)
hold on
plot(x_shape(72:end,:)',y_shape(72:end,:)','r') % plot of the remaining 35 acquisitions (pathological)
title('Healthy acquisitions in BLUE, pathological acquisitions in RED'), xlabel('Cardiac shapes')

% Pathological shapes appear to be more irregular than healthy ones. 
% As a feature, in addition to those already present, the linearity of the curve could be used.

% 2.1.3
% Computation and visualization of some basic features (curve length, shape
% area, curvature at the apex)

% Estimation of the curve length by doing the sum of the distances between
% pairs of points alog each curve 
for i=1:106
    for j=2:67
        dist(j-1) = sqrt((x_shape(i,j)-x_shape(i,j))^2+(y_shape(i,j-1)-y_shape(i,j))^2);
    end
    curve_length(i) = sum(dist);
end
% The implemented method is only an approximation of the curve length (it is an underestimation)

% Estimation of the shape area using the function polyarea
shape_area = polyarea(x_shape,y_shape,2);

% Alternative: loading of the features 'cardiacFeatures.mat'
load('cardiacFeatures.mat')

% 2.1.4
% Plot one feature against another and check if you can have an intuition of
% data clusters

% Define colors for the two classes
healthyColor = 'b'; % Blue for healthy
pathologicalColor = 'r'; % Red for pathological

figure
sgtitle('Scatter plot between the features colored according to class label')

% Scatter plot for Myocardial length vs. Cavity area
subplot(3,1,1)
scatter(features{1}(labelTnorm == 0), features{2}(labelTnorm == 0), [], healthyColor, 'filled', 'DisplayName', 'Healthy')
hold on
scatter(features{1}(labelTnorm == 1), features{2}(labelTnorm == 1), [], pathologicalColor, 'filled', 'DisplayName', 'Pathological')
xlabel('Myocardial length')
ylabel('Cavity area')
legend('Location', 'Best')

% Scatter plot for Myocardial length vs. Apical curvature
subplot(3,1,2)
scatter(features{1}(labelTnorm == 0), features{3}(labelTnorm == 0), [], healthyColor, 'filled', 'DisplayName', 'Healthy')
hold on
scatter(features{1}(labelTnorm == 1), features{3}(labelTnorm == 1), [], pathologicalColor, 'filled', 'DisplayName', 'Pathological')
xlabel('Myocardial length')
ylabel('Apical curvature')
legend('Location', 'Best')

% Scatter plot for Cavity area vs. Apical curvature
subplot(3,1,3)
scatter(features{2}(labelTnorm == 0), features{3}(labelTnorm == 0), [], healthyColor, 'filled', 'DisplayName', 'Healthy')
hold on
scatter(features{2}(labelTnorm == 1), features{3}(labelTnorm == 1), [], pathologicalColor, 'filled', 'DisplayName', 'Pathological')
xlabel('Cavity area')
ylabel('Apical curvature')
legend('Location', 'Best')

hold off


%% 2.2 Clustering

% 2.2.1
% Implementation of k-means to group the data into two clusters
X = [x_shape, y_shape];
K = 2; % 2 clusters
num_iterations = 10; % max number of iterarions (stop condition); this value can be changed
centroids = initCentroids(X, K); % initialization of k centroids
for i=1:num_iterations
	idx = getClosestCentroids(X, centroids); % index of the centroid assigned to data point i
	centroids = computeCentroids(X, idx, K); % compute means based on centroid assignments
end
% Visualization of the calculated centroids
x_centroids = centroids(:,1:67);          % x coordinates of centroids
y_centroids = centroids(:,68:end);      % y coordinates of centroids
figure, plot(x_centroids',y_centroids') % plot of the two curves of centroids
title('Centroids')

% Computation of the distances between the i-th acquisition and each of the centroids
for i=1:106
    distances_from_1(i,:) = sqrt((x_shape(i,:)-x_centroids(1,:)).^2+(y_shape(i,:)-y_centroids(1,:)).^2);
    distances_from_2(i,:) = sqrt((x_shape(i,:)-x_centroids(2,:)).^2+(y_shape(i,:)-y_centroids(2,:)).^2);
end
% Computation of the predicted classes (assign each data point to the closest
% centroid) and of the committed error
comparison_distances = distances_from_1 < distances_from_2;
for i=1:106
    num_0 = length(find(comparison_distances(i,:)==0)); % if the result of the cell is 0 it means that distances_from_1 > distances_from_2 and therefore class=1
    num_1 = length(find(comparison_distances(i,:)==1)); % if the result of the cell is 1 it means that distances_from_1 < distances_from_2 and therefore class=0
    if num_0 > num_1
        label_predict(i,1) = 0;
    else
        label_predict(i,1) = 1;
    end
end
% Sometimes the classes have to be changed because of the order of the centroids' definition
ind_0 = find(label_predict==0);
ind_1 = find(label_predict==1);
if length(ind_0)<length(ind_1)
    label_predict = zeros(106,1);
    label_predict(ind_0)=1;
end
perc_error = length(find(label_predict~=labelTnorm))/length(labelTnorm)*100; % percentage error on class predictions
