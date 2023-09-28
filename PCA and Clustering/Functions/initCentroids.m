function centroids = initCentroids(X, K)
% Calculate the initialization of K centroids
	centroids = zeros(K,size(X,2)); % initialization of a zeros matrix of K rows and the same number of columns as X
    rand_idx = randperm(size(X,1)); % random permutation of rows indexes of X
    centroids = X(rand_idx(1:K), :); % selection of the first K random rows and of all the columns of X
end