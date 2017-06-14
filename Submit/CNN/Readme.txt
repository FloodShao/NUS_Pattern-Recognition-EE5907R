CNN Assignment Instruction:

This CNN assignment is based on 'matconvnet-1(1).0-beta24'.
My contribution is in the 'main code' folder.

Run the file:
1) Make sure the user's computer has been installed C++ complier. In my case, I uses "Microsoft Windows SDK7.1(c++)".
2) Type 'mex -setup C++' in the command window in matlab, and check if the matlab has found the complier. If so proceed the next step, if not check the complier setup.
3) Copy the 'CNN' folder to the development toolkit 'matconvnet-1(1).0-beta24' (same level as the 'Makefile' and 'matlab' sub-folder), check whether the 'data' sub-folder contains the 'mnist-baseline-simplenn' sub-folder. If so, rename this folder.
4) Run the 'cnn_mnist.m' or 'cnn_minst_experiments.m' and wait for the training results.
5) Change the configuration of cnn, find the 'cnn_mnist_init.m'.


