%KNN classifier

clear all; close all; clc

%%
%data processing
%DataProcessing improvement: add a parameter to choose the processing type
datafile = 'spamData.mat';
[Xtrain_bin, ytrain_bin, Xtest_bin, ytest_bin] = DataProcessing(datafile, 'bin');
[Xtrain_log, ytrain_log, Xtest_log, ytest_log] = DataProcessing(datafile, 'log');
[Xtrain_norm, ytrain_norm, Xtest_norm, ytest_norm] = DataProcessing(datafile, 'Znorm');

%%
%KNN Parameter initiate
K = [1:9,10:5:100];
N = size(K,2);
%log file
LogFile = zeros(N,7);
LogFile(:,1) = K';
%%
%bin processing KNN classifier
[error_test, error_train] = KNN(Xtrain_bin, ytrain_bin, Xtest_bin, ytest_bin, K, 'hamming');
LogFile(:,2:3) = [error_test, error_train];

%%
%Log Data KNN classifier
[error_test, error_train] = KNN(Xtrain_log, ytrain_log, Xtest_log, ytest_log, K, 'euclidean');
LogFile(:,4:5) = [error_test, error_train];

%%
%Znormalize Data KNN classifier
[error_test, error_train] = KNN(Xtrain_norm, ytrain_norm, Xtest_norm, ytest_norm, K, 'euclidean');
LogFile(:,6:7) = [error_test, error_train];

save('KNN_Log.mat', 'LogFile');
