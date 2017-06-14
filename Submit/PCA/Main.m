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

train_data = Training_data - repmat(Mean_training_data, [N_TrainImages, 1]);
test_data = Testing_data - repmat(Mean_testing_data, [N_TestImages, 1]);

%%
%calculate the covariance of the data set 784*784, and do the eigenvector
%decomposition
S = cov(train_data);
[eigvector, eigvalue] = svd(S, 'econ');


%%
%plot the projection data
Y3 = train_data * eigvector(:,1:3);
figure;
scatter3(Y3(find(Training_label == 0),1), Y3(find(Training_label == 0),2), Y3(find(Training_label == 0),3) , 'r', 'o');
hold on;
scatter3(Y3(find(Training_label == 1),1), Y3(find(Training_label == 1),2), Y3(find(Training_label == 1),3) , 'g', 'o');
scatter3(Y3(find(Training_label == 2),1), Y3(find(Training_label == 2),2), Y3(find(Training_label == 2),3) , 'b', 'o');
scatter3(Y3(find(Training_label == 3),1), Y3(find(Training_label == 3),2), Y3(find(Training_label == 3),3) , 'r', 'filled');
scatter3(Y3(find(Training_label == 4),1), Y3(find(Training_label == 4),2), Y3(find(Training_label == 4),3) , 'g', 'filled');
scatter3(Y3(find(Training_label == 5),1), Y3(find(Training_label == 5),2), Y3(find(Training_label == 5),3) , 'b', 'filled');
scatter3(Y3(find(Training_label == 6),1), Y3(find(Training_label == 6),2), Y3(find(Training_label == 6),3) , 'r', '+');
scatter3(Y3(find(Training_label == 7),1), Y3(find(Training_label == 7),2), Y3(find(Training_label == 7),3) , 'g', '+');
scatter3(Y3(find(Training_label == 8),1), Y3(find(Training_label == 8),2), Y3(find(Training_label == 8),3) , 'b', '+');
scatter3(Y3(find(Training_label == 9),1), Y3(find(Training_label == 9),2), Y3(find(Training_label == 9),3) , 'y', 'filled');
title('3 features transform');
hold off;

Y2 = train_data * eigvector(:,1:2);
figure;
scatter(Y2(find(Training_label == 0),1), Y2(find(Training_label == 0),2) , 'r', 'o');
hold on;
scatter(Y2(find(Training_label == 1),1), Y2(find(Training_label == 1),2) , 'g', 'o');
scatter(Y2(find(Training_label == 2),1), Y2(find(Training_label == 2),2) , 'b', 'o');
scatter(Y2(find(Training_label == 3),1), Y2(find(Training_label == 3),2) , 'r', 'filled');
scatter(Y2(find(Training_label == 4),1), Y2(find(Training_label == 4),2) , 'g', 'filled');
scatter(Y2(find(Training_label == 5),1), Y2(find(Training_label == 5),2) , 'b', 'filled');
scatter(Y2(find(Training_label == 6),1), Y2(find(Training_label == 6),2) , 'r', '+');
scatter(Y2(find(Training_label == 7),1), Y2(find(Training_label == 7),2) , 'g', '+');
scatter(Y2(find(Training_label == 8),1), Y2(find(Training_label == 8),2) , 'b', '+');
scatter(Y2(find(Training_label == 9),1), Y2(find(Training_label == 9),2) , 'y', 'filled');
title('2 features transform');
hold off;

%plot the principle component
figure;
[h, array] = display_network(eigvector(:,1:2));
figure;
[h, array] = display_network(eigvector(:,1:3));
figure;
[h, array] = display_network(eigvector(:,1:25));

%%
% classfication, use NN
error_rate_40 = classification(train_data, Training_label, test_data, Testing_label, eigvector(:,1:40));
error_rate_80 = classification(train_data, Training_label, test_data, Testing_label, eigvector(:,1:80));
error_rate_200 = classification(train_data, Training_label, test_data, Testing_label, eigvector(:,1:200));

