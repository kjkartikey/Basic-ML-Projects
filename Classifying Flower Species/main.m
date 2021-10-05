%load data
rawdata = readtable('Iris.csv');
rawdata.Species = categorical(rawdata.Species);
X = rawdata{:,[2:5]};

%converting y categories to integers
a=find(rawdata.Species == 'Iris-setosa');
b=find(rawdata.Species == 'Iris-versicolor');
c=find(rawdata.Species == "Iris-virginica");
y = zeros(150,1);
for i = 1:a(end,1)
    y(i) = 1;
end
for i = i:b(end,1)
    y(i) = 2;
end
for i = i:c(end,1)
    y(i) = 3;
end

y1 = zeros(150,1);
for i = 1:150
    if y(i) == 1
        y1(i) = 1;
    else
        y1(i) = 0;
    end
end

y2 = zeros(150,1);
for i = 1:150
    if y(i) == 2
        y2(i) = 1;
    else
        y2(i) = 0;
    end
end

y3 = zeros(150,1);
for i = 1:150
    if y(i) == 3
        y3(i) = 1;
    else
        y3(i) = 0;
    end
end
%%
%initializations
m = length(y);
X = [ones(m,1),X];
theta1 = zeros(size(X,2),1);
theta2 = zeros(size(X,2),1);
theta3 = zeros(size(X,2),1);
alpha = 10^-1;
num_iter = 10000;
J1 = zeros(num_iter,1);
J2 = zeros(num_iter,1);
J3 = zeros(num_iter,1);
%%
plot(X(y==1,4),X(y==1,5),'r*');
hold on
plot(X(y==2,4),X(y==2,5),'bo');
plot(X(y==3,4),X(y==3,5),'g*');
hold off
%%
%gradient descent to identify '1' from others
for i = 1:num_iter
    theta1 = theta1 - alpha*(1/m)*X'*(sigmoid(X*theta1)-y1);
    J1(i) = cost(X,y1,theta1);
end
plot(1:num_iter,J1,'r*');

%gradient descent to identify '2' from others
for i = 1:num_iter
    theta2 = theta2 - alpha*(1/m)*X'*(sigmoid(X*theta2)-y2);
    J2(i) = cost(X,y2,theta2);
end
plot(1:num_iter,J2,'r*');

%gradient descent to identify '3' from others
for i = 1:num_iter
    theta3 = theta3 - alpha*(1/m)*X'*(sigmoid(X*theta3)-y3);
    J3(i) = cost(X,y3,theta3);
end
plot(1:num_iter,J3,'r*');
%%
%prediction
y_predicted = zeros(m,1);
for i = 1:m
    yy1 = sigmoid(X(i,:)*theta1);
    yy2 = sigmoid(X(i,:)*theta2);
    yy3 = sigmoid(X(i,:)*theta3);
    a = [yy1 yy2 yy3];
    [maximum,y_predicted(i)] = max(a);
end
%%
accuracy = 100*length(find((y_predicted-y)==0))/length(y)
%%
function S = sigmoid(z) %sigmoid function
S = 1./(1+exp(-z));
end

function J = cost(X,y,theta) %cost function
m = length(y);
J = (1/m)*sum(-y.*log(sigmoid(X*theta)) - (1-y).*log(1-sigmoid(X*theta)));
end