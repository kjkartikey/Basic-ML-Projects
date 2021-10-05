function p = predictOneVsAll(all_theta, X)
%PREDICT Predict the label for a trained one-vs-all classifier. The labels 
%are in the range 1..K, where K = size(all_theta, 1). 
%  p = PREDICTONEVSALL(all_theta, X) will return a vector of predictions
%  for each example in the matrix X. Note that X contains the examples in
%  rows. all_theta is a matrix where the i-th row is a trained logistic
%  regression theta vector for the i-th class.

m = size(X, 1);
num_labels = size(all_theta, 1); % 10

p = zeros(size(X, 1), 1); % 5000 * 1

% Add ones to the X data matrix
X = [ones(m, 1) X]; % 5000 * 401

predict = sigmoid(X*all_theta'); % 5000 * 401 by 401*10
[~, p] = max(predict, [], 2);

% M = max(A,[],dim) returns the largest elements along dimension dim.
end
