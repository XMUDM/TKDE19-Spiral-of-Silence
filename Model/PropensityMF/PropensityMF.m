function [out,meanRating,U,V,bu,bv] = PropensityMF(trainData,testData,varargin)
%% PropensityMF
%   More details:
%       T. Schnabel, A. Swaminathan, A. Singh, N. Chandak and T. Joachims. Recommendations as Treatments: Debiasing Learning and Evaluation.
%       In ICML 2016, pages 1670-1679.
%   Code provided by Dugang Liu (dugang.ldg@gmail.com)
%% 
%   Input:
%       trainData/testData: Train and test dataset
%       varargin:
%           'method': Method type
%           'm': Num of user
%           'n': Num of item
%           'F': Num of feature
%           'R': Rating range
%           'reg': Regularization parameter
%           'estimation': Propensity estimation type
%           'p': Path of propensity data
%           'topN': Num of recommended list
%           'maxIter': Maximum num of iteration
%   Output:
%       out: Evaluating metrics
%   Usage:
%   PropensityMF(train,test)
%   PropensityMF(train,test,'estimation','LR','p',path);
%% Control random number generation
rng('default');
%% Parse parameters
params = inputParser;
params.addParameter('method','PropensityMF',@(x) ischar(x));
params.addParameter('m',5400,@(x) isnumeric(x));
params.addParameter('n',1000,@(x) isnumeric(x));
params.addParameter('F',40,@(x) isnumeric(x));
params.addParameter('R',5,@(x) isnumeric(x));
params.addParameter('reg',0.008,@(x) isnumeric(x));
params.addParameter('estimation','NB',@(x) ischar(x));
params.addParameter('p','',@(x) ischar(x));
params.addParameter('topN',10,@(x) isnumeric(x));
params.addParameter('maxIter',50,@(x) isnumeric(x));
params.parse(varargin{:});
par = params.Results;
%% Run and evaluate model
methodSolver = str2func([par.method,'_solver']);
[meanRating,U,V,bu,bv] = feval(methodSolver,trainData,testData,par);
out = PropensityMF_eval(testData,meanRating,U,V,bu,bv,par.topN);
end

%% PropensityMF
function [meanRating,U,V,bu,bv] = PropensityMF_solver(trainData,testData,par)
trainMat = sparse(trainData(:,1),trainData(:,2),trainData(:,3),par.m,par.n);
if strcmp(par.estimation,'NB')
    % Propensity Estimation via Naive Bayes
    temp = tabulate(trainData(:,3));
    temp = temp(:,3)'/100;
    temp1 = temp*size(trainData,1)/(par.m*par.n);
    sample = datasample(testData,round(0.05*size(testData,1)),'Replace',false);
    temp = tabulate(sample(:,3));
    propensity = temp1./(temp(:,3)'/100);
    invP = zeros(par.m,par.n);
    for v = 1:par.R
        invP(trainMat==v) = 1/propensity(v);
    end
    invP = sparse(invP);
else
    % Propensity Estimation via Logistic Regression (Note: in the current implementation, we use the propensity provided by the author directly)
    propensityMat = load(par.p);
    invP = 1./propensityMat;
    invP(trainMat==0) = 0;
end
fprintf('PropensityMF, Propensity Estimation completed\n'); 
% Get starting params by SVD
completeMat = full(trainMat);
completeMat(completeMat==0) = mean(trainData(:,3));
[U,s,vt] = svds(completeMat,par.F,'largest','Tolerance',1e-7,'MaxIterations',2000);
V = (s*vt')';
bu = zeros(par.m,1);
bv = zeros(par.n,1);
meanRating = mean(trainData(:,3));
fprintf('PropensityMF, Get starting params completed\n'); 
% Solve params via Limited-memory BFGS 
userParams = [U,bu];
itemParams = [V,bv];
allParams = [userParams;itemParams];
paramVector = reshape(allParams,1,(par.m+par.n)*(par.F+1));
paramVector = [paramVector,meanRating];
options = struct('GoalsExactAchieve',0,'HessUpdate','lbfgs','GradObj','on','Display','iter','MaxIter',2000);
f = @(paramVector) fun(paramVector,trainMat,invP,par);
paramVector = fminlbfgs(f,paramVector,options);
meanRating = paramVector(end);
allParams = reshape(paramVector(1:end-1),par.m+par.n,par.F+1);
userParams = allParams(1:par.m,:);
itemParams = allParams(par.m+1:end,:);
U = userParams(:,1:end-1);
bu = userParams(:,end);
V = itemParams(:,1:end-1);
bv = itemParams(:,end);
% Evaluation
out = PropensityMF_eval(testData,meanRating,U,V,bu,bv,par.topN);
fprintf('PropensityMF, RMSE is %f, MAE is %f\n',out(1:2)); 
fprintf('PropensityMF, NDCG is %f/%f/%f/%f/%f/%f/%f/%f/%f/%f\n',out(3:end));
end

%% Objective/Gradient
function [objective,grad] = fun(paramVector,trainMat,invP,par)
meanRating = paramVector(end);
allParams = reshape(paramVector(1:end-1),par.m+par.n,par.F+1);
userParams = allParams(1:par.m,:);
itemParams = allParams(par.m+1:end,:);
U = userParams(:,1:end-1);
bu = userParams(:,end);
V = itemParams(:,1:end-1);
bv = itemParams(:,end);
pred = U*V'+bu+bv'+meanRating;
objective = sum(sum((pred-trainMat).^2.*invP));
scaledPenalty = par.reg*par.m*par.n/(par.m+par.n)/(par.F+1);
gradientMultiplier = invP.*2.*(pred-trainMat);
userVGradient = gradientMultiplier*V+2*scaledPenalty*U;
itemVGradient = gradientMultiplier'*U+2*scaledPenalty*V;
userBGradient = sum(gradientMultiplier,2)+2*scaledPenalty*bu;
itemBGradient = sum(gradientMultiplier)'+2*scaledPenalty*bv;
globalBGradient = 2*scaledPenalty*meanRating;
objective = objective+scaledPenalty*(sum(sum(U.^2))+sum(sum(V.^2))+sum(bu.^2)+sum(bv.^2)+meanRating^2);
userGrads = [userVGradient,userBGradient];
itemGrads = [itemVGradient,itemBGradient];
allGrads = [userGrads;itemGrads];
grad = reshape(allGrads,1,(par.m+par.n)*(par.F+1));
grad = [grad,globalBGradient];
end
