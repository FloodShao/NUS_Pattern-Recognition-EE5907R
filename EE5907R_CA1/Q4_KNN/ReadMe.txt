Update 10/02/2017

This code file is the source code for Q4_KNN. 
To run the code, please follow the steps:

	1. Include the 'spamData.mat' in the file;
	2. Find 'main.m', and run it. 
	3. And the command window will print the training procedure; after complete all training,
	   the main function will create a 'KNN_Log.mat' to store the training results.
	4. Find 'DrawResults.m' to draw the results graph;

This Code file contains 5 functions:

(1)DataProcessing.m 
	Three different processing method, use the parameter('bin','log','Znorm') to choose different method
(2)distance.m
	calculate the distance of two data point, use the parameter('euclidean', 'hamming') to choose different distance for different data processing
(3)DrawResults.m
	Draw the results graph.
(4)KNN.m
	Proceed the KNN Training
(5)main.m
	The main function

