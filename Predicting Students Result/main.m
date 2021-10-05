%data
rawdata = load('StudentMarks_Test1-Test2-Admission.txt');
X = rawdata(:,[1,2]);
y = rawdata(:,3);
m = length(y);
%%
X = [ones(m,1) X];
num_iter = 1000;
alpha = 10^-5;
theta = zeros(size(X,2),1);
J = zeros(num_iter,1);
%%
%Gradient Descent
for i = 1:num_iter
    theta = theta - alpha*(1/m)*X'*(sigmoid(X*theta) - y);
    J(i) = cost(X,y,theta);
end
plot(1:num_iter,J);
%%
%data visualization
plot(X(y==0,2),X(y==0,3),'ro');
hold on
plot(X(y==1,2),X(y==1,3),'g+');

%boundary plot
plot_x = [min(X(:,2))-2,  max(X(:,2))+2];
plot_y = (-1./theta(3)).*(theta(2).*plot_x + theta(1));
plot(plot_x, plot_y);

hold off
%%
y_predict = zeros(m,1);
for j = 1:m
    y_predict(j) = sigmoid(X(j,:)*theta);
    if y_predict(j) > 0.5
        y_predict(j) = 1;
    else
        y_predict(j) = 0;
    end
end

error = zeros(m,1);
for k = 1:m
    error(k) = abs(y_predict(k) - y(k));
end
accuracy = 100*(1 - length(find(error==1))/m)
%%
function S = sigmoid(X)  %Sigmoid Function
S = 1./(1+exp(-X));
end

function J = cost(X,y,theta)
m = length(y);
h = sigmoid(X*theta);
J = (1/m)*sum(-y.*log(h) - (1-y).*log(1 - h));
end