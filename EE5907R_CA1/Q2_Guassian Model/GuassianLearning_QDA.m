%Guassian Naive Bayes Learning
function [error_train, error_test] = GuassianLearning_QDA(Xtrain_p, ytrain_p, Xtest_p, ytest_p)

    [Ntrain, D] = size(Xtrain_p); 
    [Ntest, D] = size(Xtest_p);
    
    %%
    %ML estimation of mu and sigma for each feature in training data set
    
    Xtemp_1 = Xtrain_p(find(ytrain_p == 1), :);
    Xtemp_0 = Xtrain_p(find(ytrain_p == 0), :);
       
    mu_1 = mean(Xtemp_1);
    sigma_1 = cov(Xtemp_1);
    mu_0 = mean(Xtemp_0);
    sigma_0 = cov(Xtemp_0);
    
    %Naive Bayes Assumption
    sigma_1 = sigma_1 .* eye(D);
    sigma_0 = sigma_0 .* eye(D);
    
    %use QDA
    %%Training Data test
    Prior1_train = sum(ytrain_p)/Ntrain;
    Prior0_train = 1 - Prior1_train;
    P_1 = zeros(Ntrain, 1);
    P_0 = zeros(Ntrain, 1);
    yestimate_train = zeros(Ntrain, 1);
    
    detsigma_1 = det(sigma_1);
    detsigma_0 = det(sigma_0);
    invsigma_1 = inv(sigma_1);
    invsigma_0 = inv(sigma_0);
    
    for i = 1:Ntrain
        P_1(i) = Prior1_train * (sqrt(detsigma_1))^-1 * exp(-0.5 * (Xtrain_p(i) - mu_1) * invsigma_1 * (Xtrain_p(i) - mu_1)');
        P_0(i) = Prior0_train * (sqrt(detsigma_0))^-1 * exp(-0.5 * (Xtrain_p(i) - mu_0) * invsigma_0 * (Xtrain_p(i) - mu_0)');
    end
    
    yestimate_train(find(P_1 > P_0 )) = 1;
    
    error_train = abs(yestimate_train - ytrain_p);
    
    %%Testing Data test
%     Prior1_test = sum(ytest_p) / Ntest;
%     Prior0_test = 1 - Prior1_test;
    P_1 = zeros(Ntest, 1);
    P_0 = zeros(Ntest, 1);
    yestimate_test = zeros(Ntest, 1);
    
    for i = 1:Ntest
       P_1(i) = Prior1_train * (sqrt(detsigma_1))^-1 * exp(-0.5 * (Xtest_p(i) - mu_1) * invsigma_1 * (Xtest_p(i) - mu_1)');
       P_0(i) = Prior0_train * (sqrt(detsigma_0))^-1 * exp(-0.5 * (Xtest_p(i) - mu_0) * invsigma_0 * (Xtest_p(i) - mu_0)');
    end
    
    yestimate_test(find(P_1 > P_0 )) = 1;
    
    error_test = abs(yestimate_test - ytest_p);
    
end