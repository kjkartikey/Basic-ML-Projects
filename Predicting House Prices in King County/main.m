raw_data = readtable("kc_house_data.csv");
X = raw_data{:,4:21};
y = raw_data{:,3};
%%
X_train = X(1:10805,:);
y_train = y(1:10805);

X_test = X(10806:end,:);
y_test = y(10806:end);
%%
mu = mean(X_train);
sigma = std(X_train);
mf = ones(length(y_train),1);
X_train = (X_train - mf*mu)./(mf*sigma);

mu1 = mean(X_test);
sigma1 = std(X_test);
mf1 = ones(length(y_test),1);
X_test = (X_test - mf1*mu1)./(mf1*sigma1);


X_test = [ones(length(y_test),1), X_test];

m = length(y_train);

X_train = [ones(m,1), X_train];

n = size(X_train,2);

theta = zeros(n,1);
%%
num_itr = 1000;
J = zeros(num_itr,1);
%%
alpha = 10^-2;
for i = 1:num_itr
    theta = theta - alpha*(1/m)*(X_train'*(X_train*theta - y_train));
    J(i) = cost(X_train,y_train,theta);
end
%%
plot(1:num_itr,J);

error_train = zeros(length(y_train),1);
for j = 1:(length(y_train))
    error_train(j) = sqrt((100*(X_train(j,:)*theta - y_train(j))./y_train(j)).^2);
end
mean(error_train)
%%
error_test = zeros(length(y_test),1);
for j = 1:length(y_test)
    error_test(j) = sqrt((100*(X_test(j,:)*theta - y_test(j))./y_test(j)).^2);
end
mean(error_test)
%%
range_alpha = linspace(10^-1,10^-15,15);
mean_error = zeros(15,1);
k = 1;
for alpha = range_alpha
    for i = 1:num_itr
        theta = theta - alpha*(1/m)*(X_train'*(X_train*theta - y_train));
    end
    error_test = zeros(length(y_test),1);
    for j = 1:length(y_test)
        error_test(j) = sqrt((100*(X_test(j,:)*theta - y_test(j))./y_test(j)).^2);
    end
    mean_error(k) = mean(error_test);
    k = k+1;
end

plot(range_alpha,mean_error);