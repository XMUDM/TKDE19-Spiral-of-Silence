function out = Logitvd(trainData,testData,varargin)
%% Logit-vd
%   More details:
%       B. Marlin and R.S. Zemel. Collaborative prediction and ranking with non-random missing data. In RecSys 2009, pages 5-12.
%   Code provided by Dugang Liu (dugang.ldg@gmail.com)
%% 
%   Input:
%       seed: Random number seed to ensure repeatability of experimental results.
%       trainData/testData: Train and test dataset
%       varargin:
%           'method': Method type
%           'm': Num of user
%           'n': Num of item
%           'C': Num of mixture components
%           'R': Rating range
%           'prior': Prior parameter
%           'topN': Num of recommended list
%           'maxIter': Maximum num of iteration
%           'adaptive': Adopt adaptive learning rate or not
%   Output:
%       out: Evaluating metrics
%   Usage:
%   Logitvd(1,train,test)
%   Logitvd(1,train,test,'C',20);
%% Control random number generation
rng('default');
%% Parse parameters
params = inputParser;
params.addParameter('method','Logitvd',@(x) ischar(x));
params.addParameter('m',5400,@(x) isnumeric(x));
params.addParameter('n',1000,@(x) isnumeric(x));
params.addParameter('C',10,@(x) isnumeric(x));
params.addParameter('R',5,@(x) isnumeric(x));
params.addParameter('prior',2,@(x) isnumeric(x));
params.addParameter('topN',10,@(x) isnumeric(x));
params.addParameter('maxIter',1000,@(x) isnumeric(x));
params.addParameter('adaptive','y',@(x) ischar(x));
params.parse(varargin{:});
par = params.Results;
%% Run and evaluate model
methodSolver = str2func([par.method,'_solver']);
[bet,gamm,theta,mu] = feval(methodSolver,trainData,testData,par);
out = Logitvd_eval(testData,bet,gamm,theta,mu,par);
end

%% Logit-vd
function [bet,gamm,theta,mu] = Logitvd_solver(trainData,testData,par)
% Initialize
temp = gamrnd(par.prior,1,1,par.C);
theta = temp/sum(temp);
temp = gamrnd(par.prior,1,[par.n,par.C,par.R]);
bet = temp./sum(temp,3);
sig = repmat(0.1*normrnd(0,sqrt(10),1,1),1,par.R);
ome = 0.1*normrnd(0,sqrt(10),par.n,1);
mu = 1./(1+exp(-(ome+repmat(sig,par.n,1))));
trainMat = sparse(trainData(:,1),trainData(:,2),trainData(:,3),par.m,par.n);
obsMat = sparse(trainData(:,1),trainData(:,2),ones(size(trainData,1),1),par.m,par.n);
missMat = 1-obsMat;
% Iteration
i = 0;
while i < par.maxIter
    tic;
    % E-step
    % update q_nk
    temp = zeros(par.m,par.n,par.C,par.R);
    for k = 1:par.C
        for v = 1:par.R
            temp(:,:,k,v) = ((trainMat==v).*mu(:,v)'.*repmat(bet(:,k,v),1,par.m)').^obsMat.*((1-mu(:,v)').*repmat(bet(:,k,v),1,par.m)').^missMat;
        end
    end
    gamm = sum(temp,4);
    q_nk = repmat(theta,par.m,1);
    for d = 1:par.n
        q_nk = q_nk.*squeeze(gamm(:,d,:));
        q_nk = q_nk./sum(q_nk,2);
    end
    % update q_nkvd
    temp = zeros(par.n,par.C,par.R);
    for k = 1:par.C
        temp(:,k,:) = (1-mu).*squeeze(bet(:,k,:));
    end
    temp = temp./sum(temp,3);
    q_nkvd = zeros(par.m,par.n,par.C,par.R);
    for k = 1:par.C
        for v = 1:par.R
            q_nkvd(:,:,k,v) = repmat(q_nk(:,k),1,par.n).*repmat(temp(:,k,v)',par.m,1).^missMat;
        end
    end
    % update q_nvd
    q_nvd = squeeze(sum(q_nkvd,3)); 
    % M-step
    % update theta
    theta = par.prior-1+sum(q_nk);
    theta = theta./sum(theta);
    % update bet
    temp = zeros(par.n,par.C,par.R);
    for k = 1:par.C
        for v = 1:par.R
            temp(:,k,v) = sum(q_nk(:,k).*obsMat.*(trainMat==v)+q_nkvd(:,:,k,v).*missMat)';
        end
    end
    bet = par.prior-1+temp;
    bet = bet./sum(bet,3);
    % update sig/ome/mu
    temp = zeros(par.n,par.R);
    for v = 1:par.R
        temp(:,v) = sum(obsMat.*(trainMat==v))./mu(:,v)'+sum(q_nvd(:,:,v).*missMat)./(1-mu(:,v))';
    end
    sigD = sum(temp.*mu.*(1-mu)-1/10*sig);
    omeD = sum(temp.*mu.*(1-mu)-1/10*ome,2);
    if strcmp(par.adaptive,'y')
        lr = armijo(1,[sig,ome'],[-sigD,-omeD'],trainMat,obsMat,missMat,q_nvd,par);
    else
        lr = 1e-6;
    end
    sig = sig-lr*sigD;
    ome = ome-lr*omeD;
    mu = 1./(1+exp(-(ome+repmat(sig,par.n,1)))); 
    t = toc;     
    % Evaluation
    out = Logitvd_eval(testData,bet,gamm,theta,mu,par);
    fprintf('Logit-vd iter [%d/%d], time is %f\n',i+1,par.maxIter,t);
    fprintf('Logit-vd iter [%d/%d], mu is %f,%f,%f,%f,%f\n',i+1,par.maxIter,mu(1,:));
    fprintf('Logit-vd iter [%d/%d], RMSE is %f, MAE is %f\n',i+1,par.maxIter,out(1:2));
    fprintf('Logit-vd iter [%d/%d], NDCG is %f/%f/%f/%f/%f/%f/%f/%f/%f/%f\n',i+1,par.maxIter,out(3:end));
    i = i+1;
end
end      


%% Backtracking-Armijo Line Search for \lr
function f = fun(x,trainMat,obsMat,missMat,q_nvd,par)
sig = x(1:par.R);
ome = x(par.R+1:end)';
mu = 1./(1+exp(-(ome+repmat(sig,par.n,1)))); 
temp = zeros(par.n,par.R);
for v = 1:par.R
    temp(:,v) = sum(obsMat.*(trainMat==v)).*log(mu(:,v))'+sum(q_nvd(:,:,v).*missMat).*log(1-mu(:,v))';
end 
f =  -sum(-1/20*sig.^2)-sum(-1/20*ome.^2)+sum(sum(temp));
f = -f;
end

function lr = armijo(alphaInit,xk,dk,trainMat,obsMat,missMat,q_nvd,par)
gamma = 1e-4;
delta = 0.5;
rhok = eps;
temp = fun(xk,trainMat,obsMat,missMat,q_nvd,par);

alpha = alphaInit;
temp1 = fun(xk+alpha*dk,trainMat,obsMat,missMat,q_nvd,par);
temp2 = temp-gamma*alpha^2*(dk*dk');
while isnan(temp1) || temp1 > temp2
    if (alpha*norm(dk) < rhok)
        alpha = 0;
    else
        alpha = alpha*delta;
    end
    temp1 = fun(xk+alpha*dk,trainMat,obsMat,missMat,q_nvd,par);
    temp2 = temp-gamma*alpha^2*(dk*dk');
end
lr1 = alpha;
F1 = temp1-temp2;

alpha = alphaInit;
temp1 = fun(xk-alpha*dk,trainMat,obsMat,missMat,q_nvd,par);
temp2 = temp-gamma*alpha^2*(dk*dk');
while isnan(temp1) || temp1 > temp2
    if (alpha*norm(dk) < rhok)
        alpha = 0;
    else
        alpha = alpha*delta;
    end
    temp1 = fun(xk-alpha*dk,trainMat,obsMat,missMat,q_nvd,par);
    temp2 = temp-gamma*alpha^2*(dk*dk');
end
lr2 = -alpha;
F2 = temp1-temp2;
if F1 < F2
    lr = lr1;
else
    lr = lr2;
end
fprintf('Line search completed, lr is %f\n',lr);
end