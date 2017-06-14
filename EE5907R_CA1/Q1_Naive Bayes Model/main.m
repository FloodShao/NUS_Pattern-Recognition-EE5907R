%Main fuction
%Naive Bayes Model - Beta-Bernoulli Model 
close all; clear all; clc

%%
%data processing
load('spamData.mat');
Xtrain_p = zeros(size(Xtrain));
Xtest_p = zeros(size(Xtest));
Xtrain_p(Xtrain == 0) = 0;
Xtrain_p(Xtrain ~= 0) = 1;
Xtest_p(Xtest == 0) = 0;
Xtest_p(Xtest ~= 0) = 1;
ytrain_p = ytrain;
ytest_p = ytest;


%%
%set hyperparameter for Beta Function, two hyperparameters are set equal
a = 1:0.5:100;
% error = learning(Xtrain_p, ytrain_p, Xtest_p, ytest_p, a);
% Error_train = zeros(1,size(a,2));
% Error_test = zeros(1,size(a,2));

fprintf('Start Training!\n');
for i = 1:size(a,2)
    [Error_train(i), Error_test(i)] = learning(Xtrain_p, ytrain_p, Xtest_p, ytest_p, a(i));
end

fprintf('Complete Training!\n');
figure(1);
hold on;
plot(a,Error_train,'k');
plot(a,Error_test,'r');
title('Error funtion of \alpha');
xlabel('\alpha');
ylabel('Error');
legend('error of trainning','error of testing');
grid on;