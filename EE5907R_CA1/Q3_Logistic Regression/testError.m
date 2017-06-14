%test data error function
%use log odds
function [error] = testError(X, y, w)

    prediction = X * w'; %log odds prediction, N-by-1
    prediction(find(prediction > 0)) = 1;
    prediction(find(prediction < 0)) = 0;
    
    error = sum(abs(prediction - y)) / size(y,1);

end