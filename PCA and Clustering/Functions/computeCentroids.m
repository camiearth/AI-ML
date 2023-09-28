function centroids = computeCentroids(X, idx, K)
% Compute the values of centroids
    [m, n] = size(X); % number of rows and of columns of X
    centroids = zeros(K, n);  % initialization of a zeros matrix of K rows and the same number of columns as X
    for i=1:K
        xi = X(idx==i,:); % select the rows of X that have idx=i and all the columns of X
        ck = size(xi,1); % number of rows of xi
        centroids(i, :) = (1/ck)*sum(xi); % compute the mean of each column of xi and put this vector into the i-th row of centroids
    end
end