%Logistic Regression
%Use Newton Method And l2 regularization
%After the data processing, all the data sets are transformed to binary
%The Logistic Regression is based on Bernouli distribution
clear all; close all; clc
%%
%load the datafile and procede the pre-processing
%add one bias feature in the pre-processing procedure
datafile = 'spamData.mat';

[Xtrain_bin, ytrain_bin, Xtest_bin, ytest_bin] = DataProcessing(datafile, 'bin');
[Xtrain_log, ytrain_log, Xtest_log, ytest_log] = DataProcessing(datafile, 'log');
[Xtrain_norm, ytrain_norm, Xtest_norm, ytest_norm] = DataProcessing(datafile, 'Znorm');

lambda = [1:9, 10:5:100]; %regularized term
logfile_bin = Training(Xtrain_bin, ytrain_bin, Xtest_bin, ytest_bin, lambda);
logfile_log = Training(Xtrain_log, ytrain_log, Xtest_log, ytest_log, lambda);
logfile_norm = Training(Xtrain_norm, ytrain_norm, Xtest_norm, ytest_norm, lambda);

Logfile = [lambda', logfile_bin(:,1:3), logfile_log(:,1:3), logfile_norm(:,1:3)];

save('Logfile.mat', 'Logfile');

