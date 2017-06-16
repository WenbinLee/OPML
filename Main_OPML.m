%% Title:     OPML: A One-Pass Closed-Form Solution for Online Metric Learning
%     Author:  Wenbin li, Yang Gao, Lei Wang, Luping Zhou, Jing Huo, Yinghuan Shi
%     Journal: Pattern Recognition
%     Year:      2017
%     Version  2.0
%     copyright by Wenbin Li (2017-3-19)
%     Nanjing University in China
%     contact liwenbin.nju@gmail.com


%% Initialization
clear  variables; close all; clc;
benchpath = [pwd '\'];
addpath([benchpath, 'OPML-Metric Learning']);


%% Load data
data_UCI={ 'balance'; 'breast';  'pima'; };
data_path=[benchpath, '\Data Storage\'];

% The  Hyper-parameter of OPML
lambda =  1e-3;

Final_Results=zeros(3,3);
Error_Rate_save = cell(3,1);
Run_Time_save = cell(3,1);
Triplet_Pair_Num = zeros(3,2);
tic;
for jjj=1:3
    data_name=data_UCI{jjj};
    load([data_path, data_name, '.mat']);
    S_num=length(random_data);
    Error_rates=zeros(S_num,1);
    Run_time=zeros(S_num,1);
    
    for  iii=1:S_num
        train_data=random_data{iii}.train_data;
        train_label=random_data{iii}.train_label;
        test_data=random_data{iii}.test_data;
        test_label=random_data{iii}.test_label;
        
        
        %% The OPML Method
        tic
        [L, Triplet_num,aver_time] = OML_OPML(train_data, train_label, lambda);
        t=toc;
        train_data = (L*train_data')';
        test_data  = (L*test_data')';
        
        
        %% Matlab  function
        knn_mdl = fitcknn(train_data, train_label,'NumNeighbors', 5, 'Standardize', 0);
        pre_y = predict(knn_mdl,test_data);
        
        
        %% Error Rate
        pre_res = pre_y - test_label;
        error_rate = length(find(pre_res~=0))/length(test_label);
        Error_rates(iii) = error_rate;
        Run_time(iii) = t;
    end
    
    Error_Rate_save{jjj}=Error_rates;
    Run_Time_save{jjj}=Run_time;
    Mean_errorrate = mean(Error_rates);
    Sta_dev = std(Error_rates);
    Final_Results(jjj, 1) = Mean_errorrate;
    Final_Results(jjj, 2) = Sta_dev;
    Final_Results(jjj, 3) = t;
    fprintf('Average Error Rate & Standard Deviation---%s:\n %f  +- %f \n', data_name ,[100*Mean_errorrate, 100*Sta_dev]);
    fprintf('Running time: %f s \n', t);
    
end
t2=toc;
fprintf('The total running time: %f s \n', t2);









