%Newton Training Process
% Input:  X, y : training data set
%         w: current parameter, 1-by-D
%         lambada: step size
%         convergenceE : the convergence error
% Output: w: parameter after one iteration

function [w_next, iterationCount] = NewtonTraining(X, y, w, step, lambda, convergenceE, count)

    D = size(X,2);
    u = sigm(w * X')'; %N-by-1 colume
    
    %costFunction is NLL
    
    g = X' * (u - y) + lambda * [0, w(2:end)]'; %D-by-1 colume
    S = eye(size(u,1)) .* (u * (1-u)'); %N-N matrix
    H = X' * S * X + lambda * eye(D); %D-by-D matrix, As S is diagonal, H is positive definite
    
    %calculate the direction and update step, pinv() is more accurate than
    %inv() so the hessian matrix will not be seen as singular matrix
    d = - inv(H)*g; %find the direction of gradient descent, D-by-1
    
    %update
    w_next = w + step .* d';
    iterationCount = count +1;
    NLL = costFunction(X, y, w, lambda);
    NLL_next = costFunction(X, y, w_next, lambda);
    e = abs(NLL - NLL_next)/NLL_next; %use fractional changes
    
    fprintf('%d  %f\n', iterationCount, e);
    
    if (e > convergenceE)
        if (iterationCount < 1000)
            [w_next, iterationCount] = NewtonTraining(X, y, w_next, step, lambda, convergenceE, iterationCount);
        else 
            fprintf('Training time exceed. This Training failed');
        end
    end
    
end