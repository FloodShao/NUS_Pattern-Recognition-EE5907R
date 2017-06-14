%Training, Testing and Error Calculation
function [error_train, error_test] = learning(Xtrain_p, ytrain_p, Xtest_p, ytest_p, a)
    
% a is the hyperparameter of beta distribution
%%
% Training
% calculate the MAP for each feature in the training data
[n_train, D] = size(Xtrain_p);
%calculate binomial parameter for each feature
theta_1 = zeros(1,D);
theta_0 = zeros(1,D);

for i = 1:D
    N_1 = sum(ytrain_p);
    N1_1 = sum(Xtrain_p(find(ytrain_p == 1),i));
    theta_1(i) = (N1_1 + a - 1) / (N_1 + 2*a - 2); %MAP of feature i
    N_0 = sum(ytrain_p == 0);
    N1_0 = sum(Xtrain_p(find(ytrain_p == 0), i));
    theta_0(i) = (N1_0 + a - 1) / (N_0 + 2*a - 2);
end

P_1 = Xtrain_p .* theta_1 + (1-Xtrain_p) .* (1-theta_1);
P_0 = Xtrain_p .* theta_0 + (1-Xtrain_p) .* (1-theta_0);

Pri_1 = sum(ytrain_p)/n_train;
Pri_0 = sum(ytrain_p == 0)/n_train;

Py_1 = prod(P_1,2) * Pri_1; %the probability of label 1
Py_0 = prod(P_0,2) * Pri_0; %the probability of label 0, where sum(ytrain_p)/n_train is the priori probability

Estimate_ytrain = zeros(n_train,1);
Estimate_ytrain(find(Py_1 > Py_0)) = 1;


%%
%Testing

%find the priori in the test dataset
[n_test, D] = size(Xtest_p);
pi_ML_1 = sum(ytest_p) / n_test; %this is the priori probability
pi_ML_0 = sum(ytest_p == 0) / n_test;

P_1 = Xtest_p .* theta_1 + (1-Xtest_p) .* (1-theta_1);
P_0 = Xtest_p .* theta_0 + (1-Xtest_p) .* (1-theta_0);
Py_1 = prod(P_1,2) * Pri_1;
Py_0 = prod(P_0,2) * Pri_0;
% mu_y = mean(Py);
Estimate_ytest = zeros(n_test,1);
Estimate_ytest(find(Py_1 >= Py_0)) = 1;

%%

%calculate error
error_test = sum(abs(Estimate_ytest - ytest_p)) / n_test;
error_train = sum(abs(Estimate_ytrain - ytrain_p)) / n_train;

end