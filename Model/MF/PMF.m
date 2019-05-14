function out = PMF(trainData,testData,varargin)
%% biasedMF
%   More details:
%       R. Salakhutdinov and A. Mnih. Probabilistic Matrix Factorization. In NIPS 2007, pages 1257-1264.
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
%           'momentum': Momentum value of gradient updating
%           'batchNum': Num of batch
%           'topN': Num of recommended list
%           'maxIter': Maximum num of iteration
%   Output:
%       out: Evaluating metrics
%   Usage:
%   PMF(train,test)
%   PMF(train,test,'F',5);
%% Control random number generation
rng('default');
%% Parse parameters
params = inputParser;
params.addParameter('method','PMF',@(x) ischar(x));
params.addParameter('m',5400,@(x) isnumeric(x));
params.addParameter('n',1000,@(x) isnumeric(x));
params.addParameter('F',10,@(x) isnumeric(x));
params.addParameter('lr',50,@(x) isnumeric(x));
params.addParameter('regU',0.01,@(x) isnumeric(x));
params.addParameter('regV',0.01,@(x) isnumeric(x));
params.addParameter('momentum',0.8,@(x) isnumeric(x));
params.addParameter('batchNum',10,@(x) isnumeric(x));
params.addParameter('topN',10,@(x) isnumeric(x));
params.addParameter('maxIter',50,@(x) isnumeric(x));
params.parse(varargin{:});
par = params.Results;
%% Run and evaluate model
methodSolver = str2func([par.method,'_solver']);
[meanRating,U,V] = feval(methodSolver,trainData,testData,par);
out = PMF_eval(testData,meanRating,U,V,par.topN);
end

%% PMF
function [meanRating,U,V] = PMF_solver(trainData,testData,par)
batchIdx = discretize(1:size(trainData,1),par.batchNum);
[~,p] = numunique(batchIdx);
U = 0.1*randn(par.m,par.F);
V = 0.1*randn(par.n,par.F);
incU = zeros(par.m,par.F);
incV = zeros(par.n,par.F);
meanRating = mean(trainData(:,3));
trainData(:,3) = trainData(:,3)-meanRating;
for i = 1:par.maxIter
    for j = 1:par.batchNum
        pred = sum(U(trainData(p{j},1),:).*V(trainData(p{j},2),:),2);
        error = pred-trainData(p{j},3);
        ixU = error.*V(trainData(p{j},2),:)+par.regU*U(trainData(p{j},1),:);
        ixV = error.*U(trainData(p{j},1),:)+par.regV*V(trainData(p{j},2),:);
        gU = zeros(par.m,par.F);
        gV = zeros(par.n,par.F);
        for z = 1:length(p{j})
            gU(trainData(p{j}(z),1),:) = gU(trainData(p{j}(z),1),:)+ixU(z,:);
            gV(trainData(p{j}(z),2),:) = gV(trainData(p{j}(z),2),:)+ixV(z,:);
        end
        incU = par.momentum*incU+par.lr*gU/length(p{j});
        incV = par.momentum*incV+par.lr*gV/length(p{j});
        U = U - incU;
        V = V - incV;
    end
    out = PMF_eval(testData,meanRating,U,V,par.topN);
    fprintf('PMF iter [%d/%d], RMSE is %f, MAE is %f\n',i,par.maxIter,out(1:2)); 
    fprintf('PMF iter [%d/%d], NDCG is %f/%f/%f/%f/%f/%f/%f/%f/%f/%f\n',i,par.maxIter,out(3:end));    
end
end
