function out = RAPMF(trainData,testData,varargin)
%% RAPMF
%   More details:
%       H. Yang, G. Ling, Y. Su, MR. Lyu and I. King. Boosting response aware model-based collaborative filtering.
%       IEEE Transactions on Knowledge and Data Engineering, 27(8):2064-2077, 2015.
%   Code provided by Dugang Liu (dugang.ldg@gmail.com)
%% 
%   Input:
%       trainData/testData: Train and test dataset
%       varargin:
%           'method': Method type
%           'm': Num of user
%           'n': Num of item
%           'F': Num of feature
%           'D': Rating range
%           'lr': Learning rate
%           'sigU': Regularization parameter of user
%           'sigV': Regularization parameter of item
%           'sigR': Regularization parameter of rating
%           'topN': Num of recommended list
%           'maxIter': Maximum num of iteration
%   Output:
%       out: Evaluating metrics
%   Usage:
%   RAPMF(train,test)
%   RAPMF(train,test,'F',10);
%% Control random number generation
rng('default');
%% Parse parameters
params = inputParser;
params.addParameter('method','r',@(x) ischar(x));
params.addParameter('m',5400,@(x) isnumeric(x));
params.addParameter('n',1000,@(x) isnumeric(x));
params.addParameter('F',5,@(x) isnumeric(x));
params.addParameter('D',5,@(x) isnumeric(x));
params.addParameter('lr',1,@(x) isnumeric(x));
params.addParameter('sigU',0.2,@(x) isnumeric(x));
params.addParameter('sigV',0.2,@(x) isnumeric(x));
params.addParameter('sigR',0.2,@(x) isnumeric(x));
params.addParameter('topN',10,@(x) isnumeric(x));
params.addParameter('maxIter',250,@(x) isnumeric(x));
params.parse(varargin{:});
par = params.Results;
%% Run and evaluate model
methodSolver = str2func([par.method,'_solver']);
[U,V] = feval(methodSolver,trainData,testData,par);
out = RAPMF_eval(testData,U,V,par.topN);
end

%% RAPMF_r
function [U,V] = r_solver(trainData,testData,par)
fprintf('Adaptive eta: %f\n',normcdf(0,0,1));
% Initialize
U = unifrnd(0,1,par.m,par.F);
V = unifrnd(0,1,par.n,par.F);
par.lr = par.lr/100;
out = zeros(1,12);
countMu = tabulate(trainData(:,3));
countMu = countMu(:,2)';
trainMat = sparse(trainData(:,1),trainData(:,2),trainData(:,3),par.m,par.n);
bufferMat = zeros(par.m,par.n);
mu = [0.073,0.068,0.163,0.308,0.931];
format = repmat('%f/',1,length(mu));
fprintf(['Mu is ',format,'\n'],logsig(mu));
% Iteration
i = 0;
while i < par.maxIter
    tic;
    sigmoidMu = logsig(mu);
    predMat = logsig(U*V');
    missIdx = find(trainMat==0);
    observeIdx = setdiff(1:par.m*par.n,missIdx);
    tmp1 = arrayfun(@(x) (1-sigmoidMu(x))*(normcdf(x/4-predMat(missIdx'),0,par.sigR)-...
        normcdf((x-1)/4-predMat(missIdx'),0,par.sigR)),(1:par.D)','UniformOutput',false);
    tmp2 = arrayfun(@(x) (1-sigmoidMu(x))*(normpdf(predMat(missIdx')-(x-1)/4,0,par.sigR)-...
        normpdf(predMat(missIdx')-x/4,0,par.sigR)),(1:par.D)','UniformOutput',false);
    tmp1 = sum(cell2mat(tmp1));
    tmp2 = sum(cell2mat(tmp2));
    beta = 1/(par.sigR*sqrt(2*pi));
    bufferMat(missIdx) = beta*tmp2./tmp1;
    bufferMat(observeIdx) = ((trainMat(observeIdx)-1)/4-predMat(observeIdx))/par.sigR^2;
    clear predMat tmp1 tmp2
    % Update U
    bufferU = U;
    deltaU = bufferMat*V;
    U = U+par.lr/par.n*(deltaU-U/par.sigU^2);
    clear deltaU
    % Update V
    bufferV = V;
    deltaV = bufferMat'*bufferU;
    V = V+par.lr/par.m*(deltaV-V/par.sigV^2);
    clear deltaV
    % Update Mu
    bufferMu = mu;
    predMat = logsig(U*V');
    tmp1 = arrayfun(@(x) (1-sigmoidMu(x))*(normcdf(x/4-predMat(missIdx'),0,par.sigR)-...
        normcdf((x-1)/4-predMat(missIdx'),0,par.sigR)),(1:par.D)','UniformOutput',false);
    tmp1 = sum(cell2mat(tmp1));
    dsigmoidMu = (logsig(mu).*(1-logsig(mu)));
    deltaMu = countMu./sigmoidMu.*dsigmoidMu;
    tmp = arrayfun(@(x) (normcdf(x/4-predMat(missIdx'),0,par.sigR)-normcdf((x-1)/4-predMat(missIdx'),0,par.sigR))./...
        tmp1,(1:par.D)','UniformOutput',false);
    tmp = sum(cell2mat(tmp),2);
    deltaMu = deltaMu-dsigmoidMu.*tmp';
    mu = mu+par.lr*50/par.m/par.n*deltaMu;
    clear predMat tmp1 tmp deltaMu
    t = toc;     
    % Evaluation
    bufferOut = out;
    out = RAPMF_eval(testData,U,V,par.topN);
    % Adaptive learning rate
    % if i > 0 && out(1) > bufferOut(1)
    if i > 0 && out(1)-bufferOut(1) >= 0.05
        fprintf('RMSE increases from %f to %f.\n',bufferOut(1),out(1));
        answer = input('So does divide lr by 2? y/n: ','s');
        if strcmp(answer,'y')
            par.lr = 0.5*par.lr;
            U = bufferU;
            V = bufferV;
            mu = bufferMu;
            out = bufferOut;
            continue;
        end
    end
    fprintf(['RAPMF iter [%d/%d], time is %f, Mu is ',format,'\n'],i+1,par.maxIter,t,mu);
    fprintf('RAPMF iter [%d/%d], RMSE is %f, MAE is %f\n',i+1,par.maxIter,out(1:2));
    fprintf('RAPMF iter [%d/%d], NDCG is %f/%f/%f/%f/%f/%f/%f/%f/%f/%f\n',i+1,par.maxIter,out(3:end));
    i = i+1;
end
end      