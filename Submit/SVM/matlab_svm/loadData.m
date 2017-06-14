%Raw Data convert
function [Training_data, Training_label, N_TrainImages, Testing_data, Testing_label, N_TestImages, Size_image] = loadData()

trans_uint8ToInt = [2^24, 2^16, 2^8, 2^0];

FileName = 'train-images.idx3-ubyte';
f_handle = fopen(FileName, 'r');
[Data, count] = fread(f_handle, inf, 'uint8');
number_images = Data(5:8);
number_rows = Data(9:12);
number_columns = Data(13:16);
Training_data = Data(17:end);
fclose(f_handle);

N_TrainImages = trans_uint8ToInt * number_images;
N_TrainRows = trans_uint8ToInt * number_rows;
N_TrainColumns = trans_uint8ToInt * number_columns;
Size_image = [N_TrainRows, N_TrainColumns];


FileName = 'train-labels.idx1-ubyte';
f_handle = fopen(FileName, 'r');
[Data, count] = fread(f_handle, inf, 'uint8');
Training_label = Data(9:end);
fclose(f_handle);


FileName = 't10k-images.idx3-ubyte';
f_handle = fopen(FileName, 'r');
[Data, count] = fread(f_handle, inf, 'uint8');
number_images = Data(5:8);
Testing_data = Data(17:end);
fclose(f_handle);

N_TestImages = trans_uint8ToInt * number_images;


FileName = 't10k-labels.idx1-ubyte';
f_handle = fopen(FileName, 'r');
[Data, count] = fread(f_handle, inf, 'uint8');
Testing_label = Data(9:end);
fclose(f_handle);

end
