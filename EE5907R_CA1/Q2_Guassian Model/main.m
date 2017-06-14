%Main fuction
close all; clear all; clc
%%
%data processing
dataFile = 'spamData.mat';
[Xtrain_log, ytrain_log, Xtest_log, ytest_log] = DataProcessing(dataFile, 'log');
[Xtrain_norm, ytrain_norm, Xtest_norm, ytest_norm] = DataProcessing(dataFile, 'Znorm');

%%
%GuassianLearning process 
%use LDA 
%this function contains naive bayes LDA(diagonal covariance) and general LDA 
[Error_train, Error_test] = GuassianLearning_LDA(Xtrain_log, ytrain_log, Xtest_log, ytest_log);
Error_train_rate = sum(Error_train) / size(Error_train,1);
Error_test_rate = sum(Error_test) / size(Error_test,1);
fprintf('LDA Training Results--log:\n  Error_train_rate = %f\n  Error_test_rate = %f\n', Error_train_rate, Error_test_rate);

[Error_train, Error_test] = GuassianLearning_LDA(Xtrain_norm, ytrain_norm, Xtest_norm, ytest_norm);
Error_train_rate = sum(Error_train) / size(Error_train,1);
Error_test_rate = sum(Error_test) / size(Error_test,1);
fprintf('LDA Training Results--Znorm:\n  Error_train_rate = %f\n  Error_test_rate = %f\n', Error_train_rate, Error_test_rate);

%use QDA
%this function contains naive bayes QDA(diagonal covariance) and general
%QDA
[Error_train, Error_test] = GuassianLearning_QDA(Xtrain_log, ytrain_log, Xtest_log, ytest_log);
Error_train_rate = sum(Error_train) / size(Error_train,1);
Error_test_rate = sum(Error_test) / size(Error_test,1);
fprintf('QDA Training Results--log:\n  Error_train_rate = %f\n  Error_test_rate = %f\n', Error_train_rate, Error_test_rate);

[Error_train, Error_test] = GuassianLearning_QDA(Xtrain_norm, ytrain_norm, Xtest_norm, ytest_norm);
Error_train_rate = sum(Error_train) / size(Error_train,1);
Error_test_rate = sum(Error_test) / size(Error_test,1);
fprintf('QDA Training Results--Znorm:\n  Error_train_rate = %f\n  Error_test_rate = %f\n', Error_train_rate, Error_test_rate);