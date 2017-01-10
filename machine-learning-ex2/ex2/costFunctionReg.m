function [J, grad] = costFunctionReg(theta, X, y, lambda)
%COSTFUNCTIONREG Compute cost and gradient for logistic regression with regularization
%   J = COSTFUNCTIONREG(theta, X, y, lambda) computes the cost of using
%   theta as the parameter for regularized logistic regression and the
%   gradient of the cost w.r.t. to the parameters. 

% Initialize some useful values
m = length(y); % number of training examples

% You need to return the following variables correctly 
J = 0;
grad = zeros(size(theta));

% ====================== YOUR CODE HERE ======================
% Instructions: Compute the cost of a particular choice of theta.
%               You should set J to the cost.
%               Compute the partial derivatives and set grad to the partial
%               derivatives of the cost w.r.t. each parameter in theta
%
% Note: grad should have the same dimensions as theta
%

pre_sigmoid_predictions = X*theta;                % mxn * nx1 --> mx1
h = sigmoid(pre_sigmoid_predictions);             % --> mx1
J_unregularized = (1/m) * ((-y' * log(h)) - ((1-y)' * log(1-h))); % 1 * (1xm * mx1 - 1xm * mx1)
J = J_unregularized + ((lambda/(2*m)) .* sum(theta(2:end).^2));

errors = h - y;             % mx1 - mx1 --> mx1
delta = (1/m) * X' * errors; % 1 * nxm * mx1 --> nx1
grad(1) = delta(1); % theta_0 (the "y-intercept") doesn't get regularized
grad(2:end) = delta(2:end) + ((lambda/m)*theta(2:end));


% =============================================================

end
