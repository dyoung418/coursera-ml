function p = predict(Theta1, Theta2, X)
%PREDICT Predict the label of an input given a trained neural network
%   p = PREDICT(Theta1, Theta2, X) outputs the predicted label of X given the
%   trained weights of a neural network (Theta1, Theta2)

% Useful values
m = size(X, 1);
num_labels = size(Theta2, 1);

% You need to return the following variables correctly 
p = zeros(size(X, 1), 1);

% ====================== YOUR CODE HERE ======================
% Instructions: Complete the following code to make predictions using
%               your learned neural network. You should set p to a 
%               vector containing labels between 1 to num_labels.
%
% Hint: The max function might come in useful. In particular, the max
%       function can also return the index of the max element, for more
%       information see 'help max'. If your examples are in rows, then, you
%       can use max(A, [], 2) to obtain the max for each row.
%

% The nn has 400 input nodes (n), 25 hidden nodes and 10 output nodes (num_labels)
% Theta1 = hidden_nodes_X_input_nodes+1  (25x401)
% Theta2 = output_nodes_X_hidden_nodes+1 (10x26)
% X = m_X_input_nodes

% Add 1's column to X to represent theta_0 or the bias terminal_size
X = [ones(m, 1) X]; % add the bias units as the first column

a2 = sigmoid(X * Theta1');  %  mx401 * 401x25 --> mx25
a2 = [ones(m,1) a2];   % add the bias unit, now a2 is mx26
a3 = sigmoid(a2 * Theta2'); %  mx25 * 26x10 
[max_value, max_index] = max(a3, [], 2);
p = max_index;








% =========================================================================


end
