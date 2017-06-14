%Do the prediction
function [Logfile] = Training(Xtrain, ytrain, Xtest, ytest, lambda)
%%
    %training session
    [N_train, D] = size(Xtrain);
    [N_test, D] = size(Xtrain);

    %initiate the parameter for newton Training
    w = zeros(1,D); %row vector
    step = 1;
    convergenceE = 1e-6;
    Count = 0;
    Logfile = zeros(size(lambda,2), 3 + D); %store the traning results
    
    fprintf('Start training!\n');

    for i = 1:size(lambda,2)

        fprintf('training for lambda = %d\n', lambda(i));
        [w_final, iterationCount] = NewtonTraining(Xtrain, ytrain, w, step, lambda(i), convergenceE, Count);
        error_test = testError(Xtest, ytest, w_final);
        error_train = testError(Xtrain, ytrain, w_final);
        
        Logfile(i,:) = [error_test, error_train, iterationCount, w_final];
        fprintf('complete training for lambda = %d\n', lambda(i));

    end
    
    fprintf('Complete training!\n');
    
end