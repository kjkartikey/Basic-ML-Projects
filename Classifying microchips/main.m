%Data
rawdata = load("MicrochipScores_Test1-Test2-Functional.txt");
X = rawdata(:,[1,2]);
y = rawdata(:,3);
m = length(y);
%%
%initializations
X = [ones(m,1), X, X.*X, X.*X.*X, X.*X.*X.*X];
alpha = 1;
theta = zeros(size(X,2),1);
num_iter = 2000;
J = zeros(num_iter,1);
%%
%Gradient Descent
for i = 1:num_iter
    theta = theta - alpha*(1/m)*X'*(sigmoid(X*theta) - y);
    J(i) = cost(X,y,theta);
end

plot(1:num_iter,J);
%%
%Prediction
y_predict = zeros(m,1);
for j = 1:m
    y_predict(j) = sigmoid(X(j,:)*theta);
    if y_predict(j) > 0.5
        y_predict(j) = 1;
    else
        y_predict(j) = 0;
    end
end
%%
%visualiztions
plot(X(y==0,2),X(y==0,3),'ro');
hold on
plot(X(y==1,2),X(y==1,3),'g+');
hold off
%%
%error
error = zeros(m,1);
for k = 1:m
    error(k) = abs(y_predict(k) - y(k));
end
accuracy = 100*(1 - length(find(error==1))/m)
histogram(error)
%%
function S = sigmoid(z) %Sigmoid Function
S = 1./(1+exp(-z));
end


%Cost Function
function J = cost(X,y,theta)
m = length(y);
J = (1/m)*sum(-y.*log(sigmoid(X*theta)) - (1-y).*log(1 - sigmoid(X*theta)));
end