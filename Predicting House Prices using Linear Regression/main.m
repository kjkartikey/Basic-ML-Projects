%load data
data = load("house_data.txt");
X = data(:,1:2);
y = data(:,3);
m = length(y);
n = size(X,2) + 1;

%%
%normalisation of data
mu = mean(X);
sigma = std(X);

X = (X-mu)./sigma;
X = [ones(m,1) X];
%%
%initialising some variables
theta = zeros(n,1);
alpha = 1;
num_iteration = 4000;
J = zeros(num_iteration,1);
%%
%gradient descent
for i = 1:4000
    theta = theta - (alpha/m)*(X'*(X*theta - y));
    
    J(i) = (1/2*m)*sum((X*theta-y).^2); %computing cost
end
plot(1:num_iteration,J);
%%
%error calculations
percent_error = zeros(m,1);
y_predicted = zeros(m,1);
for j = 1:m
    y_predicted(j) = theta'*[1;(data(j,1)-mu(1))./sigma(1);(data(j,2)-mu(2))./sigma(2)]; 
    percent_error(j) = abs((theta'*[1;(data(j,1)-mu(1))./sigma(1);(data(j,2)-mu(2))./sigma(2)]-y(j))/y(j));
end
avg_percentage_error = mean(percent_error)
%%
scatter(X(:,2),y);
hold on
scatter(X(:,2),y_predicted,'black',"filled");