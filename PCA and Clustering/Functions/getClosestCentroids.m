function indices = getClosestCentroids(X, centroids)
% Compute the indices of the closest centroids
    K = size(centroids, 1); % K is the number of rows of centroids
	indices = zeros(size(X,1), 1);  % initialization of a zeros vector of the same number of rows as X
	m = size(X,1); % number of rows of X
    for i=1:m
    	k = 1; % initialize k to 1
        min_dist = sum((X(i,:)-centroids(1,:)).^ 2); % distance between the i-th row of X and the first row of centroids
        for j=2:K
        	dist = sum((X(i,:)-centroids(j,:)).^ 2); % distance between the i-th row of X and the j-th row of centroids
            if(dist < min_dist) % if the new distance < min distance
            	min_dist = dist; % update the value of min_dist with the value of dist
            	k = j; % update the value of k with the value of j
            end
        end
        indices(i) = k; % save the index k into indices
    end
end