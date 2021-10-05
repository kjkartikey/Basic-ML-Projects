function J = cost(X,y,theta)
m = size(X,1);
J = (1/2*m)*sum((X*theta - y).^2);
end