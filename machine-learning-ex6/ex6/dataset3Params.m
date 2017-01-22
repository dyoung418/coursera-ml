function [C, sigma] = dataset3Params(X, y, Xval, yval)
%DATASET3PARAMS returns your choice of C and sigma for Part 3 of the exercise
%where you select the optimal (C, sigma) learning parameters to use for SVM
%with RBF kernel
%   [C, sigma] = DATASET3PARAMS(X, y, Xval, yval) returns your choice of C and 
%   sigma. You should complete this function to return the optimal C and 
%   sigma based on a cross-validation set.
%

% You need to return the following variables correctly.
C = 1;
sigma = 0.3;

% ====================== YOUR CODE HERE ======================
% Instructions: Fill in this function to return the optimal C and sigma
%               learning parameters found using the cross validation set.
%               You can use svmPredict to predict the labels on the cross
%               validation set. For example, 
%                   predictions = svmPredict(model, Xval);
%               will return the predictions on the cross validation set.
%
%  Note: You can compute the prediction error using 
%        mean(double(predictions ~= yval))
%

C = 1.0;      % these are the optimal values I found
sigma = 0.1;

%C_trials = [0.01; 0.03; 0.1; 0.3; 1; 3; 10; 30];
%sigma_trials = [0.01; 0.03; 0.1; 0.3; 1; 3; 10; 30];
C_trials = [0.1; 1];
sigma_trials = [0.1; 1];
min_error = 100000;
min_error_C = -1000;
min_error_sigma = -1000;

for i = 1:length(C_trials)
    for j = 1:length(sigma_trials)
        cTrial = C_trials(i,:);
        sigTrial = sigma_trials(j,:);
        %fprintf(['top of for: cTrial=%f, sigTrial=%f\n'], cTrial, sigTrial);
        model= svmTrain(X, y, cTrial, @(xx1, xx2) gaussianKernel(xx1, xx2, sigTrial));
        predictions = svmPredict(model, Xval);
        valid_error = mean(double(predictions ~= yval));
        fprintf(['Trial C=%f, sigma=%f:  Error=%f\n'], cTrial, sigTrial, valid_error);
        if valid_error < min_error
            min_error = valid_error;
            min_error_C = cTrial;
            min_error_sigma = sigTrial;
        end
    end
end

C = min_error_C;
sigma = min_error_sigma;
fprintf(['\nReturning optimal values:  C=%f, sigma=%f which has error=%f'],  C, sigma, min_error);






% =========================================================================

end
