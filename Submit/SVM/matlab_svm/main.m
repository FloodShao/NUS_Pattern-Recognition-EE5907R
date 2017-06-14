%main
clear all; close all; clc
%%
%load data
[Training_data, Training_label, N_TrainImages, Testing_data, Testing_label, N_TestImages, Size_image] = loadData();
Training_data = reshape(Training_data, Size_image(1) * Size_image(2), N_TrainImages);
Testing_data = reshape(Testing_data, Size_image(1) * Size_image(2), N_TestImages);

Training_data = Training_data'; % n*d
Testing_data = Testing_data'; % n*d

Mean_training_data = mean(Training_data, 1);
Mean_testing_data = mean(Testing_data, 1);

%%
%proceed PCA
train_data = Training_data - repmat(Mean_training_data, [N_TrainImages, 1]);
test_data = Testing_data - repmat(Mean_testing_data, [N_TestImages, 1]);

%calculate the covariance of the data set 784*784, and do the eigenvector
%decomposition
S = cov(train_data);
[eigvector, eigvalue] = svd(S, 'econ');

%reduce the dimension and rescaling
%change the eigvector to tune the input data dimension
X_train_40 = train_data * eigvector(:,1:40) / 1e3;
X_test_40 = test_data * eigvector(:,1:40) / 1e3;

%%
%proceed SVM training
%-t 2for radial base kernel, 0 for linear kernel
model_40_c10 = svmtrain(Training_label, X_train_40, '-s 0 -t 0 -c 10');
model_40_c1 = svmtrain(Training_label, X_train_40, '-s 0 -t 0 -c 1');
model_40_cn1 = svmtrain(Training_label, X_train_40, '-s 0 -t 0 -c 1e-1');
model_40_cn2 = svmtrain(Training_label, X_train_40, '-s 0 -t 0 -c 1e-2');

save = [];

fprintf('40 PCA, penalty 10:\n');
fprintf('Training Accuracy:\n');
[tr_lbl, tr_acc, tr_val] = svmpredict(Training_label, X_train_40, model_40_c10);
fprintf('Testing Accuracy:\n');
[te_lbl, te_acc, te_val] = svmpredict(Testing_label, X_test_40, model_40_c10);
save = [save, tr_acc', te_acc'];

fprintf('40 PCA, penalty 1:\n');
fprintf('Training Accuracy:\n');
[tr_lbl, tr_acc, tr_val] = svmpredict(Training_label, X_train_40, model_40_c1);
fprintf('Testing Accuracy:\n');
[te_lbl, te_acc, te_val] = svmpredict(Testing_label, X_test_40, model_40_c1);
save = [save, tr_acc', te_acc'];

fprintf('40 PCA, penalty 1e-1:\n');
fprintf('Training Accuracy:\n');
[tr_lbl, tr_acc, tr_val] = svmpredict(Training_label, X_train_40, model_40_cn1);
fprintf('Testing Accuracy:\n');
[te_lbl, te_acc, te_val] = svmpredict(Testing_label, X_test_40, model_40_cn1);
save = [save, tr_acc', te_acc'];

fprintf('40 PCA, penalty 1e-2:\n');
fprintf('Training Accuracy:\n');
[tr_lbl, tr_acc, tr_val] = svmpredict(Training_label, X_train_40, model_40_cn2);
fprintf('Testing Accuracy:\n');
[te_lbl, te_acc, te_val] = svmpredict(Testing_label, X_test_40, model_40_cn2);
save = [save, tr_acc', te_acc'];


