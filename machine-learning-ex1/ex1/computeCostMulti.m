function J = computeCostMulti(X, y, theta)
%COMPUTECOSTMULTI Compute cost for linear regression with multiple variables
%   J = COMPUTECOSTMULTI(X, y, theta) computes the cost of using theta as the
%   parameter for linear regression to fit the data points in X and y

% Initialize some useful values
m = length(y); % number of training examples

% You need to return the following variables correctly 
J = 0;

% ====================== YOUR CODE HERE ======================
% Instructions: Compute the cost of a particular choice of theta
%               You should set J to the cost.
m = size(X, 1);                    % number of training examples
predictions = X*theta;
sqrErrors = (predictions - y).^2; 
J = 1/(2*m) * sum(sqrErrors);

%note, if we add regularization into this, then the cost becomes the following:
% J = 1/(2*m) * (sum(sqrErrors) + (lambda * sum(theta(2:end).^2)))



% =========================================================================

end
