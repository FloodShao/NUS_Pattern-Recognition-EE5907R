A = sum(sum(eigvalue));
for i = 1:length(eigvalue)
    if sum(sum(eigvalue(:,1:i)))>(0.95*A)
        B =i;
        break;
    end
end