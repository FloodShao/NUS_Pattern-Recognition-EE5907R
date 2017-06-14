%cost function of training w, the NLL(negative Log Likelihood)
function [cost] = costFunction(X, y, w, lambda)
    
    %calculate the log likelihood of each data, sigmoid function
    P_1 = sigm(w * X')';
    P_0 = 1 - P_1;
    Likelihood = P_1 .* y + P_0 .* (1-y); 
    NLL = -sum(log(Likelihood));
    
    cost = NLL + lambda * (w*w'); %regularization
    
end