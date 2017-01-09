function [theta, J_history] = gradientDescentMulti(X, y, theta, alpha, num_iters)
%GRADIENTDESCENTMULTI Performs gradient descent to learn theta
%   theta = GRADIENTDESCENTMULTI(x, y, theta, alpha, num_iters) updates theta by
%   taking num_iters gradient steps with learning rate alpha

% Initialize some useful values
m = length(y); % number of training examples
J_history = zeros(num_iters, 1);

for iter = 1:num_iters

    % ====================== YOUR CODE HERE ======================
    % Instructions: Perform a single gradient step on the parameter vector
    %               theta. 
    %
    % Hint: While debugging, it can be useful to print out the values
    %       of the cost function (computeCostMulti) and gradient here.
    %

    % My vectorization is theta := theta - alpha * delta
    %   where delta is 1/m sum_j(h_theta(x_i) - y_i) * x_i
    % note that X has been initialized with initial column of 1s so it 
    #   is m x n.
    n = length(theta);
    predictions = X * theta;  % aka h_theta(x_i) mxn * nx1 -- result is m x 1
    errors = predictions - y; % mx1 - mx1 -- result is m x 1
    delta = (1/m) * errors' * X;       % 1xm * mxn  -- result is 1 x n  
    %disp(sprintf('predictions - y: %d', predictions - y));
    theta = theta - (alpha * delta)'; % nx1 - (1 * 1xn)'

    % ============================================================

    % Save the cost J in every iteration    
    J_history(iter) = computeCost(X, y, theta);
    %plot(J_history);
    %pause;

end

end
