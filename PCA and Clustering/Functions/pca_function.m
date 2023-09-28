function [new_data, perc_variance] = pca_function(data)
% Implementation of PCA
% Input:
%   - data --> standardized feature set 
% Output:
%   - new_data --> reduced feature set
%   - perc_variance --> relative variance in % for each principal component


C = cov(data); % covariance matrix of the data; it contains the information to rotate the coordinate system
% The rotation helps to create new variables which are uncorrelated, so the
% covariance is zero for all pairs of the new variables

[V,D] = eig(C);
% The eigenvectors V belonging to the diagonalized covariance matrix are a 
% linear combination of the old base vectors, thus expressing the 
% correlation between the old and the new series
% The eigenvalues D of the diagonalized covariance matrix gives the variance
% within the new coordinate axes (the principal components)

new_data = V' * data';
new_data = new_data';
new_data = fliplr(new_data);
% Calculate the data set in the new coordinate system
% We need to change the order of the new variables in newdata after the 
% transformation, because the eigenvalues are in ascendent order

perc_variance = var(new_data)./sum(var(new_data))*100;
% The eigenvalues of the covariance matrix indicate the variance in the new 
% coordinate direction
% We can use this information to calculate the relative variance for each 
% new variable by dividing the variances according to the eigenvectors by 
% the sum of the variances
% From the values in rel_variance we can note that the 1st and 2nd principal
% components contain 30.07 and 7.70 percent of the total variance in data
% We optain the same values by normalizing the eigenvalues:
perc_variance2 = D/sum(D(:))*100;

end