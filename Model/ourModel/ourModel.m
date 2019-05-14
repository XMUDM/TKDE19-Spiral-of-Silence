function [out,bu,bv,U,V] = ourModel(trainData,testData,varargin)
rng('default')
%% Parse parameters
params = inputParser;
params.addParameter('method','MCO',@(x) ischar(x));
params.addParameter('m',5400,@(x) isnumeric(x));
params.addParameter('n',1000,@(x) isnumeric(x));
params.addParameter('F',5,@(x) isnumeric(x));
params.addParameter('C',2,@(x) isnumeric(x));
params.addParameter('lrT',1e-8,@(x) isnumeric(x));
params.addParameter('lr',5*1e-8,@(x) isnumeric(x));
params.addParameter('sigU',0.5,@(x) isnumeric(x));
params.addParameter('sigV',0.5,@(x) isnumeric(x));
params.addParameter('sigB',0.5,@(x) isnumeric(x));
params.addParameter('sigR',1,@(x) isnumeric(x));
params.addParameter('prior',2,@(x) isnumeric(x));
params.addParameter('maxIter',1500,@(x) isnumeric(x));
params.addParameter('topN',10,@(x) isnumeric(x));
params.parse(varargin{:});
par = params.Results;
%% Run and evaluate model
methodSolver = str2func([par.method,'_solver']);
[bu,bv,U,V] = feval(methodSolver,trainData,testData,par);
out = modelEval(testData,bu,bv,U,V,par.topN);
end

%% MCO model
function [bu,bv,U,V,Tau] = MCO_solver(trainData,testData,par)
% initial
U = 0.1*rand(par.m,par.F);
V = 0.1*rand(par.n,par.F);
bu = 0.1*rand(par.m,1);
bv = 0.1*rand(par.n,1);
Tau = rand;
dataMat = sparse(trainData(:,1),trainData(:,2),trainData(:,3),par.m,par.n);
observeIdx = find(dataMat~=0);
missIdx = find(dataMat==0);
[~,observeCol] = ind2sub([par.m,par.n],observeIdx);
[~,missCol] = ind2sub([par.m,par.n],missIdx);
temp = [trainData;testData];
[itemSet,p] = numunique(temp(:,2));
itemMean = arrayfun(@(x) mean(temp(p{x},3)),(1:length(itemSet))');
% iteration
oldLikelihood = -1e10;
for i = 1:par.maxIter
    tic;
    pred = U*V'+repmat(bu,1,par.n)+repmat(bv',par.m,1);
    % update
    temp = zeros(par.m,par.n);
    temp(observeIdx) = -abs(dataMat(observeIdx)-itemMean(observeCol));
    temp(missIdx) = abs(pred(missIdx)-itemMean(missCol))./(exp(Tau*abs(pred(missIdx)-itemMean(missCol)))-1);
    Tau = Tau+par.lrT*sum(sum(temp));
    posIdx = missIdx((pred(missIdx)-itemMean(missCol))>=0);
    negIdx = missIdx((pred(missIdx)-itemMean(missCol))<0);
    posCol = missCol((pred(missIdx)-itemMean(missCol))>=0);
    negCol = missCol((pred(missIdx)-itemMean(missCol))<0);
    s1 = zeros(par.m,par.F);
    s2 = zeros(par.m,1);
    s3 = zeros(par.n,par.F);
    s4 = zeros(par.n,1);
    temp = zeros(par.m,par.n);
    temp(observeIdx) = (dataMat(observeIdx)-pred(observeIdx))/par.sigR^2;
    temp(posIdx) = Tau./((exp(Tau*(pred(posIdx)-itemMean(posCol))))-1);
    temp(negIdx) = -Tau./(exp(Tau*(itemMean(negCol)-pred(negIdx)))-1);
    s2 = s2+sum(temp,2);
    temp1 = V';
    temp1(:,:,ones(1,par.m));
    temp1 = permute(temp1,[3 2 1]);
    temp1 = temp.*temp1;
    s1 = s1+reshape(sum(temp1,2),size(temp1,1),size(temp1,3));
    s4 = s4+sum(temp)';
    temp1 = U';
    temp1(:,:,ones(1,par.n));
    temp1 = permute(temp1,[3 2 1]);
    temp1 = temp'.*temp1;
    s3 = s3+reshape(sum(temp1,2),size(temp1,1),size(temp1,3));
    U = U+par.lr*(s1-U/par.sigU^2);
    bu = bu+par.lr*(s2-bu/par.sigB^2);
    V = V+par.lr*(s3-V/par.sigV^2);
    bv = bv+par.lr*(s4-bv/par.sigB^2);
    % log-likelihood
    likelihood = 0;
    likelihood = likelihood-sum(sum(U.^2))/(2*par.sigU^2);
    likelihood = likelihood-sum(sum(V.^2))/(2*par.sigV^2);
    likelihood = likelihood-sum(bu.^2)/(2*par.sigB^2);
    likelihood = likelihood-sum(bv.^2)/(2*par.sigB^2);
    pred = U*V'+repmat(bu,1,par.n)+repmat(bv',par.m,1);
    temp = zeros(par.m,par.n);
    temp(observeIdx) = -(dataMat(observeIdx)-pred(observeIdx)).^2/(2*par.sigR^2)-Tau*abs(dataMat(observeIdx)-itemMean(observeCol));
    temp(missIdx) = log(exp(Tau*abs(pred(missIdx)-itemMean(missCol)))-1)-Tau*abs(pred(missIdx)-itemMean(missCol));
    likelihood = likelihood+sum(sum(temp));
    t = toc;
    fprintf('MCO iter [%d/%d], likelihood is %f, time is %f\n',i,par.maxIter,likelihood,t); 
    fprintf('MCO iter [%d/%d], Tau is %f\n',i,par.maxIter,Tau); 
    % eval
    out = modelEval(testData,bu,bv,U,V,par.topN);
    fprintf('MCO iter [%d/%d], RMSE is %f, MAE is %f\n',i,par.maxIter,out(1:2)); 
    fprintf('MCO iter [%d/%d], NDCG is %f/%f/%f/%f/%f/%f/%f/%f/%f/%f\n',i,par.maxIter,out(3:end));
    if likelihood >= oldLikelihood && i > 2
        par.lrT = par.lrT*1.05;
        par.lr = par.lr*1.05;
    else
        par.lrT = par.lrT*0.5;
        par.lr = par.lr*0.5;
    end
    oldLikelihood = likelihood;
end
end

%% MCP model
function [bu,bv,U,V,Tau] = MCP_solver(trainData,testData,par)
% initial
U = 0.1*rand(par.m,par.F);
V = 0.1*rand(par.n,par.F);
bu = 0.1*rand(par.m,1);
bv = 0.1*rand(par.n,1);
Tau = rand(par.C,1);
Beta = 0.5*ones(1,par.C);
dataMat = sparse(trainData(:,1),trainData(:,2),trainData(:,3),par.m,par.n);
observeIdx = find(dataMat~=0);
missIdx = find(dataMat==0);
[~,observeCol] = ind2sub([par.m,par.n],observeIdx);
[~,missCol] = ind2sub([par.m,par.n],missIdx);
temp = [trainData;testData];
[itemSet,p] = numunique(temp(:,2));
itemMean = arrayfun(@(x) mean(temp(p{x},3)),(1:length(itemSet))');
% iteration
oldLikelihood = -1e10;
for i = 1:par.maxIter
    tic;
    pred = U*V'+repmat(bu,1,par.n)+repmat(bv',par.m,1);
    % E-step 
    Omega = zeros(par.C,par.m,par.n);
    for j = 1:par.C
        temp = zeros(par.m,par.n);
        temp(observeIdx) = normpdf(dataMat(observeIdx)-pred(observeIdx),0,par.sigR).*(1./exp(Tau(j)*abs(dataMat(observeIdx)-itemMean(observeCol))));
        temp(missIdx) = 1-(1./exp(Tau(j)*abs(pred(missIdx)-itemMean(missCol))));
        Omega(j,:,:) = temp;
    end
    Pi = repmat(Beta,par.m,1);
    for j = 1:par.n
        Pi = Pi.*Omega(:,:,j)';
        Pi = Pi./sum(Pi,2);
    end
    % M-step
    Beta = (sum(Pi)+par.prior-1)/sum((sum(Pi)+par.prior-1)); 
    for j = 1:par.C 
        temp = zeros(par.m,par.n);
        temp(observeIdx) = -abs(dataMat(observeIdx)-itemMean(observeCol));
        temp(missIdx) = abs(pred(missIdx)-itemMean(missCol))./(exp(Tau(j)*abs(pred(missIdx)-itemMean(missCol)))-1);
        Tau(j) = Tau(j)+par.lrT*sum(Pi(:,j).*sum(temp,2));
    end
    posIdx = missIdx((pred(missIdx)-itemMean(missCol))>=0);
    negIdx = missIdx((pred(missIdx)-itemMean(missCol))<0);
    posCol = missCol((pred(missIdx)-itemMean(missCol))>=0);
    negCol = missCol((pred(missIdx)-itemMean(missCol))<0);
    s1 = zeros(par.m,par.F);
    s2 = zeros(par.m,1);
    s3 = zeros(par.n,par.F);
    s4 = zeros(par.n,1);
    for j = 1:par.C
        temp = zeros(par.m,par.n);
        temp(observeIdx) = (dataMat(observeIdx)-pred(observeIdx))/par.sigR^2;
        temp(posIdx) = Tau(j)./((exp(Tau(j)*(pred(posIdx)-itemMean(posCol))))-1);
        temp(negIdx) = -Tau(j)./(exp(Tau(j)*(itemMean(negCol)-pred(negIdx)))-1);
        s2 = s2+Pi(:,j).*sum(temp,2);
        temp1 = V';
        temp1(:,:,ones(1,par.m));
        temp1 = permute(temp1,[3 2 1]);
        temp1 = (Pi(:,j).*temp).*temp1;
        s1 = s1+reshape(sum(temp1,2),size(temp1,1),size(temp1,3));
        s4 = s4+sum(Pi(:,j).*temp)';
        temp1 = U';
        temp1(:,:,ones(1,par.n));
        temp1 = permute(temp1,[3 2 1]);
        temp1 = (Pi(:,j).*temp)'.*temp1;
        s3 = s3+reshape(sum(temp1,2),size(temp1,1),size(temp1,3));
    end
    U = U+par.lr*(s1-U/par.sigU^2);
    bu = bu+par.lr*(s2-bu/par.sigB^2);
    V = V+par.lr*(s3-V/par.sigV^2);
    bv = bv+par.lr*(s4-bv/par.sigB^2); 
    % log-likelihood
    likelihood = 0;
    likelihood = likelihood-sum(sum(U.^2))/(2*par.sigU^2);
    likelihood = likelihood-sum(sum(V.^2))/(2*par.sigV^2);
    likelihood = likelihood-sum(bu.^2)/(2*par.sigB^2);
    likelihood = likelihood-sum(bv.^2)/(2*par.sigB^2);
    likelihood = likelihood+sum(Pi)*log(Beta)';
    pred = U*V'+repmat(bu,1,par.n)+repmat(bv',par.m,1);
    for j = 1:par.C
        temp = zeros(par.m,par.n);
        temp(observeIdx) = -(dataMat(observeIdx)-pred(observeIdx)).^2/(2*par.sigR^2)-Tau(j)*abs(dataMat(observeIdx)-itemMean(observeCol));
        temp(missIdx) = log(exp(Tau(j)*abs(pred(missIdx)-itemMean(missCol)))-1)-Tau(j)*abs(pred(missIdx)-itemMean(missCol));
        likelihood = likelihood+sum(sum(Pi(:,j).*temp));
    end
    t = toc;
    fprintf('MCP iter [%d/%d], likelihood is %f, time is %f\n',i,par.maxIter,likelihood,t); 
    fprintf('MCP iter [%d/%d], Beta is %f/%f, Tau is %f/%f\n',i,par.maxIter,Beta,Tau); 
    % eval
    out = modelEval(testData,bu,bv,U,V,par.topN);
    fprintf('MCP iter [%d/%d], RMSE is %f, MAE is %f\n',i,par.maxIter,out(1:2)); 
    fprintf('MCP iter [%d/%d], NDCG is %f/%f/%f/%f/%f/%f/%f/%f/%f/%f\n',i,par.maxIter,out(3:end));
    if likelihood >= oldLikelihood && i > 2
        par.lrT = par.lrT*1.05;
        par.lr = par.lr*1.05;
    else
        par.lrT = par.lrT*0.5;
        par.lr = par.lr*0.5;
    end
    oldLikelihood = likelihood;
end
end

%% MCM model
function [bu,bv,U,V,Tau] = MCM_solver(trainData,testData,par)
% initial
U = 0.1*rand(par.m,par.F);
V = 0.1*rand(par.n,par.F);
bu = 0.1*rand(par.m,1);
bv = 0.1*rand(par.n,1);
Tau = rand(par.C,1);
dataMat = sparse(trainData(:,1),trainData(:,2),trainData(:,3),par.m,par.n);
observeIdx = find(dataMat~=0);
missIdx = find(dataMat==0);
[~,observeCol] = ind2sub([par.m,par.n],observeIdx);
[~,missCol] = ind2sub([par.m,par.n],missIdx);
temp = [trainData;testData];
[itemSet,p] = numunique(temp(:,2));
itemMean = arrayfun(@(x) mean(temp(p{x},3)),(1:length(itemSet))');
itemGroup = ones(length(itemMean),1);
itemGroup(itemMean<=3) = 2;
% iteration
oldLikelihood = -1e10;
for i = 1:par.maxIter
    tic;
    pred = U*V'+repmat(bu,1,par.n)+repmat(bv',par.m,1);
    % update
    temp = zeros(par.m,par.n);
    temp(observeIdx) = -abs(dataMat(observeIdx)-itemMean(observeCol));
    temp(missIdx) = abs(pred(missIdx)-itemMean(missCol))./(exp(Tau(itemGroup(missCol)).*abs(pred(missIdx)-itemMean(missCol)))-1);
    for j = 1:par.C
        Tau(j) = Tau(j)+par.lrT*sum(sum(temp(:,itemGroup==j)));
    end
    posIdx = missIdx((pred(missIdx)-itemMean(missCol))>=0);
    negIdx = missIdx((pred(missIdx)-itemMean(missCol))<0);
    posCol = missCol((pred(missIdx)-itemMean(missCol))>=0);
    negCol = missCol((pred(missIdx)-itemMean(missCol))<0);
    s1 = zeros(par.m,par.F);
    s2 = zeros(par.m,1);
    s3 = zeros(par.n,par.F);
    s4 = zeros(par.n,1);
    temp = zeros(par.m,par.n);
    temp(observeIdx) = (dataMat(observeIdx)-pred(observeIdx))/par.sigR^2;
    temp(posIdx) = Tau(itemGroup(posCol))./((exp(Tau(itemGroup(posCol)).*(pred(posIdx)-itemMean(posCol))))-1);
    temp(negIdx) = -Tau(itemGroup(negCol))./(exp(Tau(itemGroup(negCol)).*(itemMean(negCol)-pred(negIdx)))-1);
    s2 = s2+sum(temp,2);
    temp1 = V';
    temp1(:,:,ones(1,par.m));
    temp1 = permute(temp1,[3 2 1]);
    temp1 = temp.*temp1;
    s1 = s1+reshape(sum(temp1,2),size(temp1,1),size(temp1,3));
    s4 = s4+sum(temp)';
    temp1 = U';
    temp1(:,:,ones(1,par.n));
    temp1 = permute(temp1,[3 2 1]);
    temp1 = temp'.*temp1;
    s3 = s3+reshape(sum(temp1,2),size(temp1,1),size(temp1,3));
    U = U+par.lr*(s1-U/par.sigU^2);
    bu = bu+par.lr*(s2-bu/par.sigB^2);
    V = V+par.lr*(s3-V/par.sigV^2);
    bv = bv+par.lr*(s4-bv/par.sigB^2);
    % log-likelihood
    likelihood = 0;
    likelihood = likelihood-sum(sum(U.^2))/(2*par.sigU^2);
    likelihood = likelihood-sum(sum(V.^2))/(2*par.sigV^2);
    likelihood = likelihood-sum(bu.^2)/(2*par.sigB^2);
    likelihood = likelihood-sum(bv.^2)/(2*par.sigB^2);
    pred = U*V'+repmat(bu,1,par.n)+repmat(bv',par.m,1);
    temp = zeros(par.m,par.n);
    temp(observeIdx) = -(dataMat(observeIdx)-pred(observeIdx)).^2/(2*par.sigR^2)-Tau(itemGroup(observeCol)).*abs(dataMat(observeIdx)-itemMean(observeCol));
    temp(missIdx) = log(exp(Tau(itemGroup(missCol)).*abs(pred(missIdx)-itemMean(missCol)))-1)-Tau(itemGroup(missCol)).*abs(pred(missIdx)-itemMean(missCol));
    likelihood = likelihood+sum(sum(temp));
    t = toc;
    fprintf('MCM iter [%d/%d], likelihood is %f, time is %f\n',i,par.maxIter,likelihood,t); 
    fprintf('MCM iter [%d/%d], Tau is %f/%f\n',i,par.maxIter,Tau); 
    % eval
    out = modelEval(testData,bu,bv,U,V,par.topN);
    fprintf('MCM iter [%d/%d], RMSE is %f, MAE is %f\n',i,par.maxIter,out(1:2)); 
    fprintf('MCM iter [%d/%d], NDCG is %f/%f/%f/%f/%f/%f/%f/%f/%f/%f\n',i,par.maxIter,out(3:end));
    if likelihood >= oldLikelihood && i > 2
        par.lrT = par.lrT*1.05;
        par.lr = par.lr*1.05;
    else
        par.lrT = par.lrT*0.5;
        par.lr = par.lr*0.5;
    end
    oldLikelihood = likelihood;
end
end

%% MCC model
function [bu,bv,U,V,Tau] = MCC_solver(trainData,testData,par)
% cluster users
[userSet,userP] = numunique(trainData(:,1));
userRatingNum = arrayfun(@(x) length(userP{x}),(1:length(userSet))');
[itemSet,itemP] = numunique(trainData(:,2));
itemPop = arrayfun(@(x) length(itemP{x}),1:length(itemSet));
itemMean = arrayfun(@(x) mean(trainData(itemP{x},3)),1:length(itemSet));
userPop = zeros(length(userSet),1);
userAbn = zeros(length(userSet),1);
for i = 1:length(userSet)
    temp = ismember(itemSet,trainData(userP{i},2));
    userPop(i) = sum(itemPop(temp))/sum(temp);
    idx = arrayfun(@(x) find(trainData(userP{i}(x),2)==itemSet),1:length(userP{i}));
    userAbn(i) = sum(abs(trainData(userP{i},3)-itemMean(idx)'))/sum(temp);
end
userFeature = [userRatingNum,userPop,userAbn];
options = statset('UseParallel',1);
myfunc = @(X,K)(kmeans(X,K,'Options',options,'Display','final'));
eva = evalclusters(userFeature,myfunc,'DaviesBouldin','klist',2:10);
userGroup = kmeans(userFeature,eva.OptimalK);
par.C = eva.OptimalK;
% initial
U = 0.1*rand(par.m,par.F);
V = 0.1*rand(par.n,par.F);
bu = 0.1*rand(par.m,1);
bv = 0.1*rand(par.n,1);
Tau = rand(par.C,1);
dataMat = sparse(trainData(:,1),trainData(:,2),trainData(:,3),par.m,par.n);
observeIdx = find(dataMat~=0);
missIdx = find(dataMat==0);
[observeRow,observeCol] = ind2sub([par.m,par.n],observeIdx);
[missRow,missCol] = ind2sub([par.m,par.n],missIdx);
temp = [trainData;testData];
[itemSet,p] = numunique(temp(:,2));
itemMean = arrayfun(@(x) mean(temp(p{x},3)),(1:length(itemSet))');
% iteration
oldLikelihood = -1e10;
for i = 1:par.maxIter
    tic;
    pred = U*V'+repmat(bu,1,par.n)+repmat(bv',par.m,1);
    % update
    temp = zeros(par.m,par.n);
    temp(observeIdx) = -abs(dataMat(observeIdx)-itemMean(observeCol));
    temp(missIdx) = abs(pred(missIdx)-itemMean(missCol))./(exp(Tau(userGroup(missRow)).*abs(pred(missIdx)-itemMean(missCol)))-1);
    for j = 1:par.C
        Tau(j) = Tau(j)+par.lrT*sum(sum(temp(userGroup==j,:)));
    end
    posIdx = missIdx((pred(missIdx)-itemMean(missCol))>=0);
    negIdx = missIdx((pred(missIdx)-itemMean(missCol))<0);
    posCol = missCol((pred(missIdx)-itemMean(missCol))>=0);
    posRow = missRow((pred(missIdx)-itemMean(missCol))>=0);
    negCol = missCol((pred(missIdx)-itemMean(missCol))<0);
    negRow = missRow((pred(missIdx)-itemMean(missCol))<0);
    s1 = zeros(par.m,par.F);
    s2 = zeros(par.m,1);
    s3 = zeros(par.n,par.F);
    s4 = zeros(par.n,1);
    temp = zeros(par.m,par.n);
    temp(observeIdx) = (dataMat(observeIdx)-pred(observeIdx))/par.sigR^2;
    temp(posIdx) = Tau(userGroup(posRow))./((exp(Tau(userGroup(posRow)).*(pred(posIdx)-itemMean(posCol))))-1);
    temp(negIdx) = -Tau(userGroup(negRow))./(exp(Tau(userGroup(negRow)).*(itemMean(negCol)-pred(negIdx)))-1);
    s2 = s2+sum(temp,2);
    temp1 = V';
    temp1(:,:,ones(1,par.m));
    temp1 = permute(temp1,[3 2 1]);
    temp1 = temp.*temp1;
    s1 = s1+reshape(sum(temp1,2),size(temp1,1),size(temp1,3));
    s4 = s4+sum(temp)';
    temp1 = U';
    temp1(:,:,ones(1,par.n));
    temp1 = permute(temp1,[3 2 1]);
    temp1 = temp'.*temp1;
    s3 = s3+reshape(sum(temp1,2),size(temp1,1),size(temp1,3));
    U = U+par.lr*(s1-U/par.sigU^2);
    bu = bu+par.lr*(s2-bu/par.sigB^2);
    V = V+par.lr*(s3-V/par.sigV^2);
    bv = bv+par.lr*(s4-bv/par.sigB^2);
    % log-likelihood
    likelihood = 0;
    likelihood = likelihood-sum(sum(U.^2))/(2*par.sigU^2);
    likelihood = likelihood-sum(sum(V.^2))/(2*par.sigV^2);
    likelihood = likelihood-sum(bu.^2)/(2*par.sigB^2);
    likelihood = likelihood-sum(bv.^2)/(2*par.sigB^2);
    pred = U*V'+repmat(bu,1,par.n)+repmat(bv',par.m,1);
    temp = zeros(par.m,par.n);
    temp(observeIdx) = -(dataMat(observeIdx)-pred(observeIdx)).^2/(2*par.sigR^2)-Tau(userGroup(observeRow)).*abs(dataMat(observeIdx)-itemMean(observeCol));
    temp(missIdx) = log(exp(Tau(userGroup(missRow)).*abs(pred(missIdx)-itemMean(missCol)))-1)-Tau(userGroup(missRow)).*abs(pred(missIdx)-itemMean(missCol));
    likelihood = likelihood+sum(sum(temp));
    t = toc;
    fprintf('MCC iter [%d/%d], likelihood is %f, time is %f\n',i,par.maxIter,likelihood,t); 
    format = repmat('%f/',1,par.C);
    fprintf(['MCC iter [%d/%d], Tau is ',format,'\n'],i,par.maxIter,Tau); 
    % eval
    out = modelEval(testData,bu,bv,U,V,par.topN);
    fprintf('MCC iter [%d/%d], RMSE is %f, MAE is %f\n',i,par.maxIter,out(1:2)); 
    fprintf('MCC iter [%d/%d], NDCG is %f/%f/%f/%f/%f/%f/%f/%f/%f/%f\n',i,par.maxIter,out(3:end));
    if likelihood >= oldLikelihood && i > 2
        par.lrT = par.lrT*1.05;
        par.lr = par.lr*1.05;
    else
        par.lrT = par.lrT*0.5;
        par.lr = par.lr*0.5;
    end
    oldLikelihood = likelihood;
end
end