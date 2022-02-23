%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%
% This is an examplar file on how the GLMAM program could be used.
%
% The experimental datasets are also available at£º
% Yahoo Web Pages(http://www.kecl.ntt.co.jp/as/members/ueda/yahoo.tar)
% Mulan (http://mulan.sourceforge.net/datasets-mlc.html)
% 
% Please feel free to contact me (chengyshaq@163.com), if you have any problem about this programme.
%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
clear;clc;
addpath(genpath('.'));
load('Birds.mat');
%% parameter
param.g = 9;
param.lambda1 = 100;
param.lambda2 = 1;
param.lambda3 = 1;
param.lambda4 = 0.01;
param.lambda5 = 0.01;
param.tooloptions.stopfun = @mystopfun;
%% perpare data
data    = [train_data;test_data];
target  = [train_target,test_target];
[DN,~] = size(data);
[~,TN] = size(target);
%% cross validation
if(DN==TN)
    A = (1:DN)';
    cross_num = 5;
    indices = crossvalind('Kfold',A(1:DN,1),cross_num);
    All_resluts = zeros(5,cross_num);
    
    for k = 1:cross_num
        test = (indices == k);
        test_ID = find(test==1);
        train_ID = find(test==0);
        TE_data = data(test_ID,:);
        TR_data = data(train_ID,:);
        TE_target = target(:,test_ID);
        TR_target = target(:,train_ID);
        
        %get missing labels
        [Ymis, totoalNum, totoaldeleteNum, realpercent]= getIncompleteTarget(TR_target,0.9,0);
        %neighbor number
        Num = 10;
        %encoder for attenton
        [new_Y] = train_encoder( TR_data,Ymis',Num);
        %decoder with trian
        [W,M,Z,obj_old] = train_decoder(TR_data,Ymis,new_Y,param);
        %prediction
        [Output,resluts] = Predict(W,TE_data,TE_target);
        
        All_resluts(1,k) = resluts.AveragePrecision;
        All_resluts(2,k) = resluts.AvgAuc;
        All_resluts(3,k) = resluts.Coverage;
        All_resluts(4,k) = resluts.OneError;
        All_resluts(5,k) = resluts.RankingLoss;
    end
    average_std = [mean(All_resluts,2) std(All_resluts,1,2)];   
else
    error('Dimensional inconsistency');
end