function [error_test, error_train] = KNN(Xtrain, ytrain, Xtest, ytest, K, parameter)

    [Ntest, D] = size(Xtest);
    [Ntrain, D] = size(Xtrain);
    N_try = size(K,2);

    error_test = zeros(N_try,1);
    error_train = zeros(N_try, 1);
    
    fprintf('Start training!\n');
    
    for j = 1:N_try
        Prediction = - zeros(Ntest,1); %initiate as -1

        for i = 1:Ntest

            d = distance(Xtest(i,:), Xtrain, parameter);
            [sort_d, sort_index] = sort(d);

            temp = ytrain(sort_index(1:K(j)));
            K_1 = sum(temp);
            K_0 = K(j) - K_1;

            I_1 = find(temp == 1);
            I_0 = find(temp == 0);

            V_1 = max(d(I_1));
            V_0 = max(d(I_0));

            if(size(I_1,1) ~= 0 && size(I_0,1)~= 0) %the data in the window contains all classes 
                P_1 = K_1 / V_1;
                P_0 = K_0 / V_0;
                if (P_1 > P_0) Prediction(i) = 1;
                else Prediction(i) = 0;
                end
            else %the data in the window only contains one class
                if (I_1 ~=0) Prediction(i) = 1;
                else Prediction(i) = 0;
                end
            end

        end
        error_test(j) = sum(xor(Prediction, ytest)) / Ntest;

        for i = 1:Ntrain

            d = distance(Xtrain(i,:), Xtrain, parameter);
            [sort_d, sort_index] = sort(d);

            temp = ytrain(sort_index(2:K(j)+1));
            K_1 = sum(temp);
            K_0 = K(j) - K_1;

            I_1 = find(temp == 1);
            I_0 = find(temp == 0);

            V_1 = max(d(I_1));
            V_0 = max(d(I_0));

            if(size(I_1,1) ~= 0 && size(I_0,1)~= 0) %the data in the window contains all classes 
                P_1 = K_1 / V_1;
                P_0 = K_0 / V_0;
                if (P_1 > P_0) Prediction(i) = 1;
                else Prediction(i) = 0;
                end
            else %the data in the window only contains one class
                if (I_1 ~=0) Prediction(i) = 1;
                else Prediction(i) = 0;
                end
            end

        end    

        error_train(j) = sum(xor(Prediction, ytrain)) / Ntrain;

        fprintf('Complete K = %d, error_test = %f, errror_train = %f\n', K(j), error_test(j), error_train(j));
    end
    
    fprintf('Training Complete!\n');
end