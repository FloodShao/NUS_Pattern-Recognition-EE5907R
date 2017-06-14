Update 10/02/2017

This code file is the source code for Q3_Logistic Regression. 
To run the code, please follow the steps:

	1. Include the 'spamData.mat' in the file;
	2. Find 'main.m', and run it. 
	3. And the command window will print the training procedure; after complete all training,
	   the main function will create a 'Logfile.mat' to store the training results.
	4. Find 'DrawResults.m' to draw the results graph;
	   two graphs will be presented, figure1 is the error rate vs. lambda, figure2 is the iterationNum vs. lambda.

This Code file contains 8 functions:

(1)costFunction.m 
	The negative log likelihood (NLL) as the cost function for Newton Training
(2)DataProcessing.m
	Three different processing method, use the parameter('bin','log','Znorm') to choose different method
(3)DrawResults.m
	Draw two graphs. The error rate and the iteration number
(4)main.m
	The main function
(5)NewtonTraining.m
	Proceed the Newton Training
(6)sigm.m
	Return the value of sigmoid function
(7)testError.m
	Return the test error
(8)Training.m
	Package Newton Training and do the whole training process

