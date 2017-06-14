%Guassian Learning
%use LDA linear boundary of each class
function [error_train, error_test] = GuassianLearning_LDA(Xtrain_p, ytrain_p, Xtest_p, ytest_p)

    [Ntrain, D] = size(Xtrain_p); 
    [Ntest, D] = size(Xtest_p);
    
    %%
    %ML estimation of mu and sigma for each feature in training data set
    
    Xtemp_1 = Xtrain_p(find(ytrain_p == 1), :);
    Xtemp_0 = Xtrain_p(find(ytrain_p == 0), :);
       
    mu_1 = mean(Xtemp_1);
    mu_0 = mean(Xtemp_0);
    
    %LDA assumes all classes have the same covariance
    sigma = cov(Xtrain_p);
    
% this command diagonal the covariance of features    
%     sigma = sigma .* eye(D);
    
    %use LDA
    %%Training Data test
    Prior1_train = sum(ytrain_p)/Ntrain;
    Prior0_train = 1 - Prior1_train;

    
    w = (mu_1 - mu_0) * inv(sigma);
    x0=  0.5*(mu_1 - mu_0) + (mu_1 - mu_0) * log(Prior1_train / Prior0_train) / ((mu_1 - mu_0) * inv(sigma) * (mu_1 - mu_0)');
    
    yestimate_train = (w * (Xtrain_p' - x0'))';
    yestimate_train(yestimate_train > 0) = 1;
    yestimate_train(yestimate_train <= 0) = 0;
    
    error_train = abs(yestimate_train - ytrain_p);
    
    %%Testing Data test
%     Prior1_test = sum(ytest_p) / Ntest;
%     Prior0_test = 1 - Prior1_test;
    
    x0_test=  0.5*(mu_1 - mu_0) + (mu_1 - mu_0) * log(Prior1_train / Prior0_train) / ((mu_1 - mu_0) * inv(sigma) * (mu_1 - mu_0)');
    
    yestimate_test = (w * (Xtest_p' - x0_test'))';
    yestimate_test(yestimate_test > 0) = 1;
    yestimate_test(yestimate_test <= 0) = 0;

    error_test = abs(yestimate_test - ytest_p);
    
end