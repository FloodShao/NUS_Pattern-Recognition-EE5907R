SVM assignment instruction:

This SVM assignment is based on libsvm-3.21, the assignment file is in the file folder'matlab_svm'. There are two files from my contribution:
1) main.m
2) loadData.m

Run the file:
1) Make sure the user's computer has been installed C++ complier. In my case, I uses "Microsoft Windows SDK7.1(c++)".
2) Type 'mex -setup C++' in the command window in matlab, and check if the matlab has found the complier. If so proceed the next step, if not check the complier setup.
3) Direct the path to 'matlab_svm', and type 'make' in the command window, to run the make.m file. (Actually this folder already contains the mex file with file type '.mexw64', you can directly use it in matlab).
4) Run the 'main.m', after completing the training, the results of desired PCA dimension will save in the 'save' variable in the workplace.
5) Change the PCA dimension in the *27 and *28 line in the 'main.m': the original dimension is set as 40, if you want to change the dimension to 80, just replace the 'eigvector(:,1:40)' with 'eigvector(:,1:80)' in both lines.
6) Change the training kernel: the original kernel is set as linear '-t 0' in lines *33 to *36. If you want to change the kernel to radial based kernel, change the parameter in these 4 lines as '-t 2'

 