Update 10/02/2017

This code file is the source code for Q2_Gaussian Model. 
To run the code, please follow the steps:

	1. Include the 'spamData.mat' in the file;
	2. Find 'main.m', and run it. 
	3. And you will get the error rate of LDA and QDA in the Command Window.

This Code file contains 4 functions:

(1)DataProcessing.m 
	Use Log, and Z-normalization transform to process the raw data
(2)GuassianLearning_LDA.m
	Use LDA, assuming the variance of all features are the same
(3)GuassianLearning_QDA.m
	Use QDA, only assuming all features are independent
(4)main.m
	The main function

The final results in command window is:
LDA Training Results--log:
  Error_train_rate = 0.085155
  Error_test_rate = 0.074219
LDA Training Results--Znorm:
  Error_train_rate = 0.148124
  Error_test_rate = 0.131510