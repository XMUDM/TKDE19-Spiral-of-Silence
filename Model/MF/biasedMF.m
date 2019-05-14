function out = biasedMF(trainData,testData,varargin)
%% biasedMF
%   More details:
%       Y. Koren, R. Bell and C. Volinsky. Matrix factorization techniques for recommender systems.
%       Computer, (8):30-37, 2009.
%   Code provided by Dugang Liu (dugang.ldg@gmail.com)
%% 
%   Input:
%       trainData/testData: Train and test dataset
%       varargin:
%           'method': Method type
%           'm': Num of user
%           'n': Num of item
%           'F': Num of feature
%           'lr': Learning rate
%           'regU': Regularization parameter of user
%           'sigV': Regularization parameter of item
%           'sigB': Regularization parameter of bias
%           'momentum': Momentum value of gradient updating
%           'batchNum': Num of batch
%           'topN': Num of recommended list
%           'maxIter': Maximum num of iteration
%   Output:
%       out: Evaluating metrics
%   Usage:
%   biasedMF(train,test)
%   biasedMF(train,test,'F',5);
%% Control random number generation
rng('default');
%% Parse parameters
params = inputParser;
params.addParameter('method','biasedMF',@(x) ischar(x));
params.addParameter('m',5400,@(x) isnumeric(x));
params.addParameter('n',1000,@(x) isnumeric(x));
params.addParameter('F',10,@(x) isnumeric(x));
params.addParameter('lr',50,@(x) isnumeric(x));
params.addParameter('regU',0.01,@(x) isnumeric(x));
params.addParameter('regV',0.01,@(x) isnumeric(x));
params.addParameter('regB',0.01,@(x) isnumeric(x));
params.addParameter('momentum',0.8,@(x) isnumeric(x));
params.addParameter('batchNum',10,@(x) isnumeric(x));
params.addParameter('topN',10,@(x) isnumeric(x));
params.addParameter('maxIter',50,@(x) isnumeric(x));
params.parse(varargin{:});
par = params.Results;
%% Run and evaluate model
methodSolver = str2func([par.method,'_solver']);
[meanRating,U,V,bu,bv] = feval(methodSolver,trainData,testData,par);
out = biasedMF_eval(testData,meanRating,U,V,bu,bv,par.topN);
end

%% biasedMF
function [meanRating,U,V,bu,bv] = biasedMF_solver(trainData,testData,par)
batchIdx = discretize(1:size(trainData,1),par.batchNum);
[~,p] = numunique(batchIdx);
U = normrnd(0,0.1,par.m,par.F);
V = normrnd(0,0.1,par.n,par.F);
bu = normrnd(0,0.1,par.m,1);
bv = normrnd(0,0.1,par.n,1);
incU = zeros(par.m,par.F);
incV = zeros(par.n,par.F);
incbu = zeros(par.m,1);
incbi = zeros(par.n,1);
meanRating = mean(trainData(:,3));
lastLoss = 0;
for i = 1:par.maxIter
    loss = 0;
    for j = 1:par.batchNum
        pred = meanRating+sum(U(trainData(p{j},1),:).*V(trainData(p{j},2),:),2)+bu(trainData(p{j},1))+...
            bv(trainData(p{j},2));
        error = pred-trainData(p{j},3);
        loss = loss+sum(error.^2);
        ixU = error.*V(trainData(p{j},2),:)+par.regU*U(trainData(p{j},1),:);
        ixV = error.*U(trainData(p{j},1),:)+par.regV*V(trainData(p{j},2),:);
        ixbu = error+par.regB*bu(trainData(p{j},1));
        ixbv = error+par.regB*bv(trainData(p{j},2));
        gU = zeros(par.m,par.F);
        gV = zeros(par.n,par.F);
        gbu = zeros(par.m,1);
        gbv = zeros(par.n,1);
        for z = 1:length(p{j})
            gU(trainData(p{j}(z),1),:) = gU(trainData(p{j}(z),1),:)+ixU(z,:);
            gV(trainData(p{j}(z),2),:) = gV(trainData(p{j}(z),2),:)+ixV(z,:);
            gbu(trainData(p{j}(z),1)) = gbu(trainData(p{j}(z),1))+ixbu(z);
            gbv(trainData(p{j}(z),2)) = gbv(trainData(p{j}(z),2))+ixbv(z);
        end
        incU = par.momentum*incU+par.lr*gU/length(p{j});
        incV = par.momentum*incV+par.lr*gV/length(p{j});
        incbu = par.momentum*incbu+par.lr*gbu/length(p{j});
        incbi = par.momentum*incbi+par.lr*gbv/length(p{j});
        U = U - incU;
        V = V - incV;
        bu = bu - incbu;
        bv = bv - incbi;
        loss = loss+par.regU*sum(sum(U(trainData(p{j},1),:).^2))+par.regV*sum(sum(V(trainData(p{j},2),:).^2))+...
            par.regB*sum(bu(trainData(p{j},1)).^2)+par.regB*sum(bv(trainData(p{j},2)).^2);
    end
    deltaLoss = lastLoss-0.5*loss;
    lastLoss = 0.5*loss;
    if abs(deltaLoss)<1e-5
        break;
    end
    fprintf('biasedMF iter [%d/%d] completed, loss = %f, delta_loss: %f, lr: %f\n',i,par.maxIter,0.5*loss,deltaLoss,par.lr);
    out = biasedMF_eval(testData,meanRating,U,V,bu,bv,par.topN);
    fprintf('biasedMF iter [%d/%d], RMSE is %f, MAE is %f\n',i,par.maxIter,out(1:2)); 
    fprintf('biasedMF iter [%d/%d], NDCG is %f/%f/%f/%f/%f/%f/%f/%f/%f/%f\n',i,par.maxIter,out(3:end));      
end
end