function J = costFunctionJ(X, y, theta)

% X is the "design matrix" with training examples and X_0 is 1 for each
% y is the class labels (i.e. the true values for the training examples)

m = size(X, 1);   % number of training examples)
predictions = X*theta; %pred of hypothesis on all training examples
sqrErrors = (predictions - y).^2;  % squared errors (note .^ is cell-wise power)

J = 1/(2*m) * sum(sqrErrors);