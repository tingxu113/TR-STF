clear all; clc;
close all;
addpath(genpath('code'));
addpath(genpath('corefun'));
%% Randomly Simulated Tensor data
%   d=4; n=40; r=ones(d,1)*5;  
%   node=cell(1,d);
%   for i=1:d-1
%      node{i}=randn(r(i),n,r(i+1));
%   end
%  node{d}=randn(r(d),n,r(1));
%  tr=tensor_ring;
%  tr=cell2core(tr,node);
%  data=full(tr);
%  data = reshape(data,ones(1,d)*n);
%  MaxIter=10*d;
%  tol=1e-4;
%  r=[1;38;40;40];

 
%% Pavia University data set
% S=imread('.\original_rosis.tif');
% data=S(1:64,1:64,11:end);
% data=double(data);
% d=length(data);
% MaxIter=10*d;
% tol=1e-4;
% r=[1;65;93];

%% CBSD68 data set
% data=load('.\CBSD68.mat');
% data=data.im;
% d=length(data);
% MaxIter=10*d;
% tol=1e-5;
%r=[16;16;3300;1100];

%% COIL-100 data set
% data=load('.\coil100.mat');
% data=data.im;
% d=length(data);
% MaxIter=10*d;
% tol=1e-5;
 %r=[5;5;5;5];

%% highly oscillatory functions
d=3; n= 200;
x= linspace(-1,1,n^d)';%f1(x)
fun = @(x) (x+1).*sin(100*(x+1).^2);
data = fun(x);
data = reshape(data,[ones(1,d)*n]);
MaxIter=20*d;
tol=1e-4;
temp=data./std(data(:));

%% THE PROPOSED TR-STF 
fprintf('\tRunning TR-STF\n');
r=[1;4;15];
tic_exp = tic;
[tr_stf,cores_stf] = tensor_ring(temp,'Alg','STF','Rank',r);
time_TR_STF = toc(tic_exp);
%X=coreten2tr(cores);
resTR_STF = norm(temp(:)-full(tr_stf))/norm(temp(:));
% nopTR = sum(tr.n.*tr.r.*[tr.r(2:end);tr.r(1)])

%% TR-SVD
% fprintf('\tRunning TR-SVD\n');
% tic_exp = tic;
% [tr_svd,cores_svd] = tensor_ring(temp,'Tol',tol,'Alg','SVD');
% time_TR_SVD = toc(tic_exp);
% %X=coreten2tr(cores);
% resTR_SVD = norm(temp(:)-full(tr_svd))/norm(temp(:));

%% TR-ALS 
% fprintf('\tRunning TR-ALS\n');
% tic_exp = tic;
% r=[1;2;6];
% [tr_ALS,cores_ALS] = tensor_ring(temp,'Tol',tol,'Alg','ALS','Rank',r,'maxit',MaxIter);
% time_TR_ALS= toc(tic_exp);
% %X=coreten2tr(cores);
% resTR_ALS = norm(temp(:)-full(tr_ALS))/norm(temp(:));

%% TR-BALS
% fprintf('\tRunning TR-BALS\n');
% tic_exp = tic;
% [tr_BALS,cores_BALS] = tensor_ring(temp,'Tol',tol,'Alg','BALS','maxit',MaxIter);
% time_TR_BALS= toc(tic_exp);
% resTR_BALS=norm(temp(:)-full(tr_BALS))/norm(temp(:));

%% TR-ALSAR
% fprintf('\tRunning TR-ALSAR\n');
% tic_exp = tic;
% [tr_ALSAR,cores_ALSAR] = tensor_ring(temp,'Tol',tol,'Alg','ALSAR','maxit',MaxIter);
% time_TR_ALSAR= toc(tic_exp);
% %X=coreten2tr(cores);
% resTR_ALSAR = norm(temp(:)-full(tr_ALSAR))/norm(temp(:));

%% DISP RESULTS
% disp(['time_STF (our): ',num2str(time_TR_STF), '  | ', 'RelCha_STF: ',num2str(resTR_STF)]);
% disp(['time_SVD: ',num2str(time_TR_SVD), ' |  ', 'RelCha_SVD: ',num2str(resTR_SVD)]);
% disp(['time_ALS: ',num2str(time_TR_ALS), '  | ', 'RelCha_ALS: ',num2str(resTR_ALS)]);
% disp(['time_BALS: ',num2str(time_TR_BALS), '  | ', 'RelCha_BALS: ',num2str(resTR_BALS)]);
% disp(['time_ALSAR: ',num2str(time_TR_ALSAR), '  | ', 'RelCha_ALSAR: ',num2str(resTR_ALSAR)]);

%% Feature Extraction for Classification
% feat_mat = reshape(permute(cores{4}, [2 1 3]), 7200, 25);
% class_array=zeros(7200,1);
% for i=1:100
%     k=72;
%     class_array(72*(i-1)+1:72*i,:)=i;
% end
% Mdl = fitcknn(feat_mat, class_array, 'numneighbors', 1);
% cvmodel = crossval(Mdl);
% loss = kfoldLoss(cvmodel, 'lossfun', 'classiferror');
% accuracy = 1 - loss;