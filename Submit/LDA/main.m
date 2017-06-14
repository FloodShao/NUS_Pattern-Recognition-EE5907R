%main function
clear all; close all; clc
%%
%load data
[Training_data, Training_label, N_TrainImages, Testing_data, Testing_label, N_TestImages, Size_image] = loadData();
Training_data = reshape(Training_data, Size_image(1) * Size_image(2), N_TrainImages);
Testing_data = reshape(Testing_data, Size_image(1) * Size_image(2), N_TestImages);

Training_data = Training_data';
Testing_data = Testing_data';

classlabel = unique(Training_label);

%%
%calculate the scatter matrix of original data
%total mean vector
mu = mean(Training_data,1);
data = struct;
Sw = zeros(size(Training_data,2), size(Training_data,2));
Sb = zeros(size(Training_data,2), size(Training_data,2));
for i = 1:size(classlabel)
   %class-specific mean vector
   data(i).mu_i = mean(Training_data(find(Training_label == classlabel(i)),:), 1); 
   data(i).N_i = size(find(Training_label == classlabel(i)),1);
   %class-specific scatter matrix
   data(i).Si = cov(Training_data(find(Training_label == classlabel(i)),:) - data(i).mu_i);
   data(i).classlabel = classlabel(i);
   %within class scatter
   Sw = Sw + data(i).N_i/size(Training_label,1)*data(i).Si; 
   %between class scatter matrix
   Sb = Sb + data(i).N_i/size(Training_label,1)*(data(i).mu_i - mu)' * (data(i).mu_i - mu);
end

%%
%solve the generalized eigenvector problem
%the generalized eigenvector problem is (Sb - lambda * Sw) * w = 0
[V, D] = eig(pinv(Sw)*Sb); %the eigvalue is sorted from the maximum to minimum
EigenVal = diag(D);
[sort_Eigenval, sort_index] = sort(EigenVal, 'descend');
%%
%formulate the transformation matrix, the column vector
A_2 = conj(V(:,sort_index(1:2)));
A_3 = conj(V(:,sort_index(1:3)));
A_9 = conj(V(:,sort_index(1:9)));

%dimension reduction
Y_2_train = Training_data * A_2;
Y_2_test = Testing_data * A_2;
Y_3_train = Training_data * A_3;
Y_3_test = Testing_data * A_3;
Y_9_train = Training_data * A_9;
Y_9_test=  Testing_data * A_9;

%%
%plot the projection graph
figure;
scatter(Y_2_train(find(Training_label == 0),1), Y_2_train(find(Training_label == 0),2) , 'r', 'o');
hold on;
scatter(Y_2_train(find(Training_label == 1),1), Y_2_train(find(Training_label == 1),2) , 'g', 'o');
scatter(Y_2_train(find(Training_label == 2),1), Y_2_train(find(Training_label == 2),2) , 'b', 'o');
scatter(Y_2_train(find(Training_label == 3),1), Y_2_train(find(Training_label == 3),2) , 'r', 'filled');
scatter(Y_2_train(find(Training_label == 4),1), Y_2_train(find(Training_label == 4),2) , 'g', 'filled');
scatter(Y_2_train(find(Training_label == 5),1), Y_2_train(find(Training_label == 5),2) , 'b', 'filled');
scatter(Y_2_train(find(Training_label == 6),1), Y_2_train(find(Training_label == 6),2) , 'r', '+');
scatter(Y_2_train(find(Training_label == 7),1), Y_2_train(find(Training_label == 7),2) , 'g', '+');
scatter(Y_2_train(find(Training_label == 8),1), Y_2_train(find(Training_label == 8),2) , 'b', '+');
scatter(Y_2_train(find(Training_label == 9),1), Y_2_train(find(Training_label == 9),2) , 'y', 'filled');
title('2 features transform');
hold off;

figure;
scatter3(Y_3_train(find(Training_label == 0),1), Y_3_train(find(Training_label == 0),2), Y_3_train(find(Training_label == 0),3) , 'r', 'o');
hold on;
scatter3(Y_3_train(find(Training_label == 1),1), Y_3_train(find(Training_label == 1),2), Y_3_train(find(Training_label == 1),3) , 'g', 'o');
scatter3(Y_3_train(find(Training_label == 2),1), Y_3_train(find(Training_label == 2),2), Y_3_train(find(Training_label == 2),3) , 'b', 'o');
scatter3(Y_3_train(find(Training_label == 3),1), Y_3_train(find(Training_label == 3),2), Y_3_train(find(Training_label == 3),3) , 'r', 'filled');
scatter3(Y_3_train(find(Training_label == 4),1), Y_3_train(find(Training_label == 4),2), Y_3_train(find(Training_label == 4),3) , 'g', 'filled');
scatter3(Y_3_train(find(Training_label == 5),1), Y_3_train(find(Training_label == 5),2), Y_3_train(find(Training_label == 5),3) , 'b', 'filled');
scatter3(Y_3_train(find(Training_label == 6),1), Y_3_train(find(Training_label == 6),2), Y_3_train(find(Training_label == 6),3) , 'r', '+');
scatter3(Y_3_train(find(Training_label == 7),1), Y_3_train(find(Training_label == 7),2), Y_3_train(find(Training_label == 7),3) , 'g', '+');
scatter3(Y_3_train(find(Training_label == 8),1), Y_3_train(find(Training_label == 8),2), Y_3_train(find(Training_label == 8),3) , 'b', '+');
scatter3(Y_3_train(find(Training_label == 9),1), Y_3_train(find(Training_label == 9),2), Y_3_train(find(Training_label == 9),3) , 'y', 'filled');
title('3 features transform');
hold off;
%%
%estimate label
label_2 = -ones(size(Testing_label, 1),1);
label_3 = -ones(size(Testing_label, 1),1);
label_9 = -ones(size(Testing_label, 1),1);

for j = 1:size(Testing_label,1)
   d_2 = sqrt(sum((Y_2_test(j,:) - Y_2_train).^2,2));
   d_3 = sqrt(sum((Y_3_test(j,:) - Y_3_train).^2,2));
   d_9 = sqrt(sum((Y_9_test(j,:) - Y_9_train).^2,2));
   
   [v, idx_2] = min(d_2);
   [v, idx_3] = min(d_3);
   [v, idx_9] = min(d_9);
   
   label_2(j) = Training_label(idx_2);
   label_3(j) = Training_label(idx_3);
   label_9(j) = Training_label(idx_9);
   
end
%%
%calculate the error rate
error_2 = 1 -  size(find((label_2 - Testing_label) == 0),1)/ size(Testing_label,1);
error_3 = 1 -  size(find((label_3 - Testing_label) == 0),1)/ size(Testing_label,1);
error_9 = 1 -  size(find((label_9 - Testing_label) == 0),1)/ size(Testing_label,1);
