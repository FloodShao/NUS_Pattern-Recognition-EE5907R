%sigmoid function
%the input can be a vector
function [sig_x] = sigm(x)
    sig_x = (1 + exp(-x)).^(-1);
end
