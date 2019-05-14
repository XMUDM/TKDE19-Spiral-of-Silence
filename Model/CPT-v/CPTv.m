function out = CPTv(trainData,testData,muPrior,varargin)
%% CPT-v
%   More details:
%       B. Marlin, R.S. Zemel, S. Roweis and M. Slaney. Collaborative Filtering and the Missing at Random Assumption. In UAI 2007, pages 267-275.
%   Code provided by Dugang Liu (dugang.ldg@gmail.com)
%% 
%   Input:
%       seed: Random number seed to ensure repeatability of experimental results.
%       trainData/testData: Train and test dataset
%       muPrior: informative prior for the \mu parameters
%       varargin:
%           'method': Mehtod type
%           'm': Num of user
%           'n': Num of item
%           'C': Num of mixture components
%           'R': Rating range
%           'S': Prior strength
%           'prior': Prior parameter
%           'topN': Num of recommended list
%           'maxIter': Maximum num of iteration
%   Output:
%       out: Evaluating metrics
%   Usage:
%   CPTv(train,test)
%   CPTv(train,test,'C',20);
%% Control random number generation
rng('default');
%% Parse parameters
params = inputParser;
params.addParameter('method','CPTv',@(x) ischar(x));
params.addParameter('m',5400,@(x) isnumeric(x));
params.addParameter('n',1000,@(x) isnumeric(x));
params.addParameter('C',10,@(x) isnumeric(x));
params.addParameter('R',5,@(x) isnumeric(x));
params.addParameter('S',200,@(x) isnumeric(x));
params.addParameter('prior',2,@(x) isnumeric(x));
params.addParameter('topN',10,@(x) isnumeric(x));
params.addParameter('maxIter',1000,@(x) isnumeric(x));
params.parse(varargin{:});
par = params.Results;
%% Run and evaluate model
methodSolver = str2func([par.method,'_solver']);
[bet,gamm,theta] = feval(methodSolver,trainData,testData,muPrior,par);
out = CPTv_eval(testData,bet,gamm,theta,par);
end

%% CPT-v
function [bet,gamm,theta] = CPTv_solver(trainData,testData,muPrior,par)
% Initialize
temp = gamrnd(par.prior,1,1,par.C);
theta = temp/sum(temp);
temp = gamrnd(par.prior,1,[par.n,par.C,par.R]);
bet = temp./sum(temp,3);
xi0 = par.S*(1-muPrior);
xi1 = par.S*muPrior;
mu = betarnd(xi1,xi0,1,par.R);
format = repmat('%f/',1,length(mu));
trainMat = sparse(trainData(:,1),trainData(:,2),trainData(:,3),par.m,par.n);
obsMat = sparse(trainData(:,1),trainData(:,2),ones(size(trainData,1),1),par.m,par.n);
missMat = 1-obsMat;
% Iteration
i = 0;
lambda = zeros(par.m,par.n,par.C,par.R);
while i < par.maxIter
    tic;
    % E-step
    % update lambda
    for z = 1:par.C
        for v = 1:par.R
            lambda(:,:,z,v) = ((trainMat==v)*mu(v).*repmat(bet(:,z,v),1,par.m)').^obsMat.*((1-mu(v))*repmat(bet(:,z,v),1,par.m)').^missMat;
        end
    end
    % update gamm
    gamm = sum(lambda,4);
    % update group
    group = repmat(theta,par.m,1);
    for m = 1:par.n
        group = group.*squeeze(gamm(:,m,:));
        group = group./sum(group,2);
    end
    % M-step
    % update theta
    theta = par.prior-1+sum(group);
    theta = theta./sum(theta);
    % update bet
    for m = 1:par.n
        for z = 1:par.C
            for v = 1:par.R
                bet(m,z,v) = par.prior-1+sum(group(:,z).*lambda(:,m,z,v)./gamm(:,m,z));
            end
        end
    end
    bet = bet./sum(bet,3);
    % update mu   
    tempGroup = zeros(par.m,par.n,par.C);
    for m = 1:par.n
        tempGroup(:,m,:) = group;
    end
    tempObs = zeros(par.m,par.n,par.C);
    for z = 1:par.C
        tempObs(:,:,z) = obsMat;
    end
    for v = 1:par.R
        res1 = sum(sum(sum(tempObs.*lambda(:,:,:,v)./gamm.*tempGroup)));
        res2 = sum(sum(sum(lambda(:,:,:,v)./gamm.*tempGroup)));
        mu(v) = (xi1(v)-1+res1)/(xi0(v)+xi1(v)-2+res2);
    end
    t = toc;     
    % Evaluation
    out = CPTv_eval(testData,bet,gamm,theta,par);
    fprintf(['CPT-v iter [%d/%d], time is %f, Mu is ',format,'\n'],i+1,par.maxIter,t,mu);
    fprintf('CPT-v iter [%d/%d], RMSE is %f, MAE is %f\n',i+1,par.maxIter,out(1:2));
    fprintf('CPT-v iter [%d/%d], NDCG is %f/%f/%f/%f/%f/%f/%f/%f/%f/%f\n',i+1,par.maxIter,out(3:end));
    i = i+1;
end
end      

