% Data Processing
% there are 57 features of the input data
% 1-48 type word_freq_WORD: percentage
% 49-54 type char_freq_WORD: percentage
% 55: type capital_run_length_average
% 56: type captital_run_length_longest
% 57: type capital_run_length_total
function [Xtrain_p, ytrain_p, Xtest_p, ytest_p] = DataProcessing(dataFile, parameter)

    load(dataFile);

    
    ytrain_p = ytrain;
    ytest_p = ytest;
    
    switch parameter
        case 'bin'
            %     binary feature
            Xtrain_p = Xtrain;
            Xtest_p = Xtest;
            
            Xtrain_p(find(Xtrain > 0)) = 1;
            Xtrain_p(find(Xtrain <= 0)) = 0;

            Xtest_p(find(Xtest > 0)) = 1;
            Xtest_p(find(Xtest <= 0)) = 0;
            fprintf('Bin Processing\n');
       
        case 'log'
            %     log transform for long feature
            Xtrain_p = Xtrain;
            Xtest_p = Xtest;
            
            Xtrain_p = log(Xtrain_p + 0.1);

            Xtest_p = log(Xtest_p + 0.1);
            fprintf('Log Processing\n');
            
        case 'Znorm'
            %     z-normalization
            Xtrain_p = Xtrain;
            Xtest_p = Xtest;
            
            mu_train = mean(Xtrain_p);
            sigma_train = std(Xtrain_p);
            mu_test = mean(Xtest_p);
            sigma_test = std(Xtest_p);

            Xtrain_p = (Xtrain_p - mu_train) ./ sigma_train;
            Xtest_p = (Xtest_p - mu_test) ./ sigma_test;
            fprintf('Znormalize Processing\n');
        otherwise
            fprintf('No Data Processing, use the raw data\n');
     
    end

end