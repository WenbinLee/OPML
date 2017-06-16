%% The results of Euclidean distance
clear  variables; close all; clc;

%% Initialization
benchpath = [pwd '\'];

%% Load data
data_UCI={ 'lsvt'; 'iris'; 'wine';  'glass';  'spect';  'ionosphere';  'balance'; 'breast';  'pima';  'segmentation';  'waveform'; 'optdigits'};
data_path=[benchpath, '\Data Storage\'];
data=dir([data_path, '*.mat']);
Final_Results=zeros(12,2);
Error_Rate_save = cell(12,1);

for jjj=1:12
    data_name=data_UCI{jjj};
    load([data_path, data_name, '.mat']);
    
    S_num=length(random_data);
    Error_rates=zeros(S_num,1);
    for  iii=1:S_num
        train_data=random_data{iii}.train_data;
        train_label=random_data{iii}.train_label;
        test_data=random_data{iii}.test_data;
        test_label=random_data{iii}.test_label;
        
        %% Matlab  function
        knn_mdl = fitcknn(train_data, train_label,'NumNeighbors', 5, 'Standardize', 0);
        pre_y = predict(knn_mdl,test_data);
        
        %% Error Rate
        pre_res = pre_y - test_label;
        error_rate = length(find(pre_res~=0))/length(test_label);
        Error_rates(iii) = error_rate;
        
    end
    
    Error_Rate_save{jjj}=Error_rates;
    Mean_errorrate = mean(Error_rates);
    Sta_dev = std(Error_rates);
    Final_Results(jjj, 1) = Mean_errorrate;
    Final_Results(jjj, 2) = Sta_dev;
    fprintf('Average Error Rate & Standard Deviation---%s:\n %f  +- %f \n', data_name ,[100*Mean_errorrate, 100*Sta_dev]);
    
end












