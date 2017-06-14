%calculate the distance of between two data points
%input: X0: the test data, vector
%       X: all training set for one class label
%       parameter: euclidean, or hamming
%output: d: vector of distance
function [d] = distance(X0, X, parameter)
    
    [N, D] = size(X);
    d = zeros(N,1);
    
    switch parameter
       
        case 'euclidean'
            d = sqrt( sum((X0 - X).^2 , 2));
        case 'hamming' %for binary feature
            temp = xor(X0, X);
            d = sum(temp,2);
        otherwise
            fprintf('unidentified distance type, return d = 0');
    end

end