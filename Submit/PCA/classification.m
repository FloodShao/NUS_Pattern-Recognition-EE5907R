%PCA classification function
function error_rate = classification(train_data, train_label, test_data, test_label, PC_basis)

    N_TestImages = length(test_label);
    N_TrainImages = length(train_label);
    
    train = train_data * PC_basis;
    test = test_data * PC_basis;
    class_test = zeros(N_TestImages,1);

    for i = 1:N_TestImages
        Sub = train - repmat(test(i,:), [N_TrainImages, 1]);
        L = sqrt(sum(Sub.^2, 2));
        [value, ind]=  min(L);
        class_test(i) = train_label(ind);
    end

    result_40 = class_test - test_label;
    result_40(find(result_40 ~= 0)) = 1;
    error_rate = sum(result_40) / N_TestImages;
    
end