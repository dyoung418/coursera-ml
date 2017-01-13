function [J grad] = nnCostFunction(nn_params, ...
                                   input_layer_size, ...
                                   hidden_layer_size, ...
                                   num_labels, ...
                                   X, y, lambda)
%NNCOSTFUNCTION Implements the neural network cost function for a two layer
%neural network which performs classification
%   [J grad] = NNCOSTFUNCTON(nn_params, hidden_layer_size, num_labels, ...
%   X, y, lambda) computes the cost and gradient of the neural network. The
%   parameters for the neural network are "unrolled" into the vector
%   nn_params and need to be converted back into the weight matrices. 
% 
%   The returned parameter grad should be a "unrolled" vector of the
%   partial derivatives of the neural network.
%

% Reshape nn_params back into the parameters Theta1 and Theta2, the weight matrices
% for our 2 layer neural network
Theta1 = reshape(nn_params(1:hidden_layer_size * (input_layer_size + 1)), ...
                 hidden_layer_size, (input_layer_size + 1));

Theta2 = reshape(nn_params((1 + (hidden_layer_size * (input_layer_size + 1))):end), ...
                 num_labels, (hidden_layer_size + 1));

% Setup some useful variables
m = size(X, 1);
         
% You need to return the following variables correctly 
J = 0;
Theta1_grad = zeros(size(Theta1)); % 25x401  -- hidden x input+1
Theta2_grad = zeros(size(Theta2)); % 10x26   -- output x hidden+1

% ====================== YOUR CODE HERE ======================
% Instructions: You should complete the code by working through the
%               following parts.
%
% Part 1: Feedforward the neural network and return the cost in the
%         variable J. After implementing Part 1, you can verify that your
%         cost function computation is correct by verifying the cost
%         computed in ex4.m
%
% Part 2: Implement the backpropagation algorithm to compute the gradients
%         Theta1_grad and Theta2_grad. You should return the partial derivatives of
%         the cost function with respect to Theta1 and Theta2 in Theta1_grad and
%         Theta2_grad, respectively. After implementing Part 2, you can check
%         that your implementation is correct by running checkNNGradients
%
%         Note: The vector y passed into the function is a vector of labels
%               containing values from 1..K. You need to map this vector into a 
%               binary vector of 1's and 0's to be used with the neural network
%               cost function.
%
%         Hint: We recommend implementing backpropagation using a for-loop
%               over the training examples if you are implementing it for the 
%               first time.
%
% Part 3: Implement regularization with the cost function and gradients.
%
%         Hint: You can implement this around the code for
%               backpropagation. That is, you can compute the gradients for
%               the regularization separately and then add them to Theta1_grad
%               and Theta2_grad from Part 2.
%

% Part 1 -- Feedforward and compute cost without regularization
%disp(size(y));           % 5000x1

% Convert y to one-hot encoding
y_onehot = double([1:num_labels]==y);
%disp(size(y_onehot));           % 5000x1

J_unregularized = 0;

% we need to portion out the error attributable to every single weight in the
% graph.  (Remember there are more weights than their are nodes in the graph 
% because each node gets weights from *all* of the nodes from the previous
% layers.  So in each layer, l, there are num_nodes_l-1 * num_nodes_l weights.
% We need that many error terms (plus error terms for the bias that is added to 
% each node).  
% The error of each weight is the weighted average of the errors further up the
% graph (starting with the output error).
% If we were doing all example data at once, the error for each theta/bias (weights/bias)
% would be delta_l+1 * a_l' (note the transpose on a_l).  However, we are doing
% this one example at a time (for i = 1:m), so we need an accumulator to add
% up the terms from each example.
% bigDelta_l is the accumulator for the weights between level l and l+1.
bigDelta1 = zeros(size(Theta1)); % l=2, i=nodes in layer2=25, j= nodes in layer1=400 -> 25x401
%bigDelta1 = [ones(1, size(bigDelta1,2)); bigDelta1]; % add bias row (but not at end delta) -> 26x401
%disp(size(bigDelta1));
bigDelta2 = zeros(size(Theta2)); % 10x26
%disp(size(bigDelta2));
for i = 1:m,
    % Forward Propogation
    x_i = X(i,:);            % 1x400
    y_i = y_onehot(i,:);     % 1x10
    a1 = [ones(1,1); x_i'];  % 401x1
    z2 = Theta1*a1;          % 25x401 * 401x1 --> 25x1
    a2 = sigmoid(z2);        % 25x1
    a2 = [ones(1,1); a2]; % add ones column for bias (aka theta_0) --> 26x1
    z3 = Theta2*a2;          % 10x26 * 26x1 --> 10x1
    a3 = sigmoid(z3);        % 10x1
    h = a3;  % should be mx1 but I haven't exhaustively checked
    
    % Cost Accumulation
    J_i_unregularized = ((-y_i * log(h)) ...
                        - ((1-y_i) * log(1-h))); % (1x10 * 10x1 - 1x10 * 10x1)
    J_unregularized = J_unregularized + J_i_unregularized;
    
    % Back Propogation
    delta3 = a3 - y_i';      % 10x1 - 10x1 --> 10x1
    delta2 = (Theta2' * delta3) .* (a2.*(1-a2)); % 2nd half is same as sigmaGradient(z2) 26x10 * 10x1 -> 26x1
    bigDelta1 = bigDelta1 + delta2(2:end) * a1'; % 25x401 + ((26-1)x1 * 1x401) --> 25x401 (note that I removed delta_0)
    bigDelta2 = bigDelta2 + delta3 * a2'; % 10x26 + (10x1 * 1x26) --> 10x26
    
end
J_unregularized = (1/m) .* J_unregularized;
J = J_unregularized + ((lambda/(2*m)) .* ...
                (sum(sum(Theta1(:,2:end).^2)') + ...
                 sum(sum(Theta2(:,2:end).^2)') ) );

Theta1_grad_unregularized = (1/m) .* bigDelta1;
Theta2_grad_unregularized = (1/m) .* bigDelta2;

% don't want to regularize the biases, so do those separately
Theta1_grad(:,1) = Theta1_grad_unregularized(:,1);
Theta1_grad(:,2:end) = Theta1_grad_unregularized(:,2:end) + ((lambda/m).*Theta1(:,2:end));
Theta2_grad(:,1) = Theta2_grad_unregularized(:,1);
Theta2_grad(:,2:end) = Theta2_grad_unregularized(:,2:end) + ((lambda/m).*Theta2(:,2:end));


% -------------------------------------------------------------

% =========================================================================

% Unroll gradients
grad = [Theta1_grad(:) ; Theta2_grad(:)];


end
