function [U, S] = pca(X)
%PCA Run principal component analysis on the dataset X
%   [U, S, X] = pca(X) computes eigenvectors of the covariance matrix of X
%   Returns the eigenvectors U, the eigenvalues (on diagonal) in S
%

% Useful values
[m, n] = size(X);

% You need to return the following variables correctly.
U = zeros(n);
S = zeros(n);

% ====================== YOUR CODE HERE ======================
% Instructions: You should first compute the covariance matrix. Then, you
%               should use the "svd" function to compute the eigenvectors
%               and eigenvalues of the covariance matrix. 
%
% Note: When computing the covariance matrix, remember to divide by m (the
%       number of examples).
%

% the covariance matrix is called the 'sigma' matrix
sigma = (1/m).*(X'*X);      % nxm * mxn --> nxn
% eigenvalues can be gotten with eig(), but our instructor
%    uses svd() when the input is a covariance matrix 
[U, S, V] = svd(sigma);     % U is nxn.  The columns are the vectors for reduction
% U holds the vectors for reducing the dimensions.  Just choose
%    the first k U vectors to reduce to k dimensions
% Ureduce = U(:, 1:k);
% Then calculate the new values, z as follows
% z = Ureduce'*X




% =========================================================================

end
