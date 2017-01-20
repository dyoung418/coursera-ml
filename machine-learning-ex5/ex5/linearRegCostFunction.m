function [J, grad] = linearRegCostFunction(X, y, theta, lambda)
%LINEARREGCOSTFUNCTION Compute cost and gradient for regularized linear 
%regression with multiple variables
%   [J, grad] = LINEARREGCOSTFUNCTION(X, y, theta, lambda) computes the 
%   cost of using theta as the parameter for linear regression to fit the 
%   data points in X and y. Returns the cost in J and the gradient in grad

% Initialize some useful values
m = length(y); % number of training examples -- y is mx1

% You need to return the following variables correctly 
J = 0;
grad = zeros(size(theta));

% ====================== YOUR CODE HERE ======================
% Instructions: Compute the cost and gradient of regularized linear 
%               regression for a particular choice of theta.
%
%               You should set J to the cost and grad to the gradient.
%

predictions = X*theta;                % mxn * nx1 --> mx1
%h = 1.0 ./ (1.0 + exp(-predictions)); % (sigmoid)   --> mx1
h = predictions;
J_unregularized = (1/(2*m)) * sum((h-y).^2); % 1 * sum((mx1 - mx1).^2) = 1
J = J_unregularized + ((lambda/(2*m)) .* sum(theta(2:end).^2));

errors = h - y;             % mx1 - mx1 --> mx1
delta = (1/m) * X' * errors; % 1 * nxm * mx1 --> nx1
grad(1) = delta(1); % theta_0 (the "y-intercept") doesn't get regularized
grad(2:end) = delta(2:end) + ((lambda/m)*theta(2:end));










% =========================================================================

grad = grad(:);

end
