function out = KNNRec(trainData,testData,varargin)
%% Memory-based CF
%   More details:
%       http://csse.szu.edu.cn/staff/panwk/recommendation/CF/MemoryBasedCF.pdf
%   Code provided by Dugang Liu (dugang.ldg@gmail.com)
%%
%   Input:
%       trainData/testData: Train and test dataset
%       varargin:
%           'method': Method type
%           'similarity': Similarity measurement
%           'm': Num of user
%           'n': Num of item
%           'k': Num of nearest neighbor
%           'topN': Num of recommended list
%   Output:
%       out: Evaluating metrics
%   Usage:
%   KNNRec(train,test)
%   KNNRec(train,test,'method','item','similarity','cosine');
%% Parse parameters
params = inputParser;
params.addParameter('method','user',@(x) ischar(x));
params.addParameter('similarity','pearson',@(x) ischar(x));
params.addParameter('m',5400,@(x) isnumeric(x));
params.addParameter('n',1000,@(x) isnumeric(x));
params.addParameter('k',10,@(x) isnumeric(x));
params.addParameter('topN',10,@(x) isnumeric(x));
params.parse(varargin{:});
par = params.Results;
%% Run and evaluate model
methodSolver = str2func([par.method,'_solver']);
out = feval(methodSolver,trainData,testData,par);
end

%% UserKNN
function out = user_solver(trainData,testData,par)
% Precomputation
dataMat = sparse(trainData(:,1),trainData(:,2),trainData(:,3),par.m,par.n);
dataMat = full(dataMat);
userMean = arrayfun(@(x) mean(dataMat(x,dataMat(x,:)~=0)),(1:size(dataMat,1))');
simSolver = str2func([par.similarity,'_sim']);
simValue = arrayfun(@(x) feval(simSolver,x,dataMat(x,:),dataMat,userMean(x),userMean),1:size(dataMat,1),'UniformOutput',false);
temp = cell2mat(simValue);
simValue = temp+temp';
userNeighbors = cell(1,size(simValue,1));
userWeights = cell(1,size(simValue,1));
for i = 1:size(simValue,1)
    idx = find(simValue(i,:)~=0);
    [B,I] = sort(simValue(i,idx),'descend');
    userWeights{i} = B';
    userNeighbors{i} = idx(I)';
end
fprintf('UserKNN, Precomputation completed\n');
% Prediction
pred = zeros(size(testData,1),1);
[itemSet,p] = numunique(trainData(:,2));
for i = 1:size(testData,1)
    temp = ismember(userNeighbors{testData(i,1)},trainData(p{itemSet==testData(i,2)},1));
    topNeighbors = find(temp==1,par.k);
    if ~isempty(topNeighbors)
        pred(i) = userMean(testData(i,1))+sum(userWeights{testData(i,1)}(topNeighbors).*(dataMat(userNeighbors{testData(i,1)}(topNeighbors),testData(i,2))-...
            userMean(userNeighbors{testData(i,1)}(topNeighbors))))/sum(abs(userWeights{testData(i,1)}(topNeighbors)));
    else
        pred(i) = userMean(testData(i,1));
    end
end
pred(pred>5) = 5;
pred(pred<1) = 1;
fprintf('UserKNN, Prediction completed\n');
% Evaluation
[userSet,p] = numunique(testData(:,1));
out = zeros(1,12);
out(1) = sqrt(sum((testData(:,3)-pred).^2)/size(testData,1));
out(2) = sum(abs(testData(:,3)-pred))/size(testData,1);
temp = zeros(length(userSet),10);
for i = 1:length(userSet)
    for j = 1:par.topN
        r = (1:j)';
        if length(p{i})>r(end)
            rel = testData(p{i},3);
            [~,I] = sort(pred(p{i}),'descend');
            dcg = sum((2.^rel(I(r))-1)./log2(r+1));
            [~,I] = sort(rel,'descend');
            idcg = sum((2.^rel(I(r))-1)./log2(r+1));
            temp(i,j) = sum(dcg/idcg);
        else
            tr = (1:length(p{i}))';
            rel = testData(p{i},3);
            [~,I] = sort(pred(p{i}),'descend');
            dcg = sum((2.^rel(I(tr))-1)./log2(tr+1));
            [~,I] = sort(rel,'descend');
            idcg = sum((2.^rel(I(tr))-1)./log2(tr+1));
            temp(i,j) = sum(dcg/idcg);
        end
    end
end
out(3:end) = mean(temp);
fprintf('UserKNN, RMSE is %f, MAE is %f\n',out(1:2));
fprintf('UserKNN, NDCG is %f/%f/%f/%f/%f/%f/%f/%f/%f/%f\n',out(3:end));
end

%% ItemKNN
function out = item_solver(trainData,testData,par)
% Precomputation
dataMat = sparse(trainData(:,1),trainData(:,2),trainData(:,3),par.m,par.n);
dataMat = full(dataMat);
userMean = arrayfun(@(x) mean(dataMat(x,dataMat(x,:)~=0)),(1:size(dataMat,1))');
simSolver = str2func([par.similarity,'_sim']);
simValue = arrayfun(@(x) feval(simSolver,x,dataMat(:,x),dataMat,userMean),1:size(dataMat,2),'UniformOutput',false);
temp = cell2mat(simValue);
simValue = temp+temp';
itemNeighbors = cell(1,size(simValue,1));
itemWeights = cell(1,size(simValue,1));
for i = 1:size(simValue,1)
    idx = find(simValue(i,:)~=0);
    [B,I] = sort(simValue(i,idx),'descend');
    itemWeights{i} = B;
    itemNeighbors{i} = idx(I);
end
fprintf('ItemKNN, Precomputation completed\n');
% Prediction
pred = zeros(size(testData,1),1);
[userSet,p] = numunique(trainData(:,1));
for i = 1:size(testData,1)
    temp = ismember(itemNeighbors{testData(i,2)},trainData(p{userSet==testData(i,1)},2));
    topNeighbors = find(temp==1,par.k);
    if ~isempty(topNeighbors)
        pred(i) = sum(itemWeights{testData(i,2)}(topNeighbors).*dataMat(testData(i,1),itemNeighbors{testData(i,2)}(topNeighbors)))/...
            sum(abs(itemWeights{testData(i,2)}(topNeighbors)));
    else
        pred(i) = userMean(testData(i,1));
    end
end
pred(pred>5) = 5;
pred(pred<1) = 1;
fprintf('ItemKNN, Prediction completed\n');
% Evaluation
[userSet,p] = numunique(testData(:,1));
out = zeros(1,12);
out(1) = sqrt(sum((testData(:,3)-pred).^2)/size(testData,1));
out(2) = sum(abs(testData(:,3)-pred))/size(testData,1);
temp = zeros(length(userSet),10);
for i = 1:length(userSet)
    for j = 1:par.topN
        r = (1:j)';
        if length(p{i})>r(end)
            rel = testData(p{i},3);
            [~,I] = sort(pred(p{i}),'descend');
            dcg = sum((2.^rel(I(r))-1)./log2(r+1));
            [~,I] = sort(rel,'descend');
            idcg = sum((2.^rel(I(r))-1)./log2(r+1));
            temp(i,j) = sum(dcg/idcg);
        else
            tr = (1:length(p{i}))';
            rel = testData(p{i},3);
            [~,I] = sort(pred(p{i}),'descend');
            dcg = sum((2.^rel(I(tr))-1)./log2(tr+1));
            [~,I] = sort(rel,'descend');
            idcg = sum((2.^rel(I(tr))-1)./log2(tr+1));
            temp(i,j) = sum(dcg/idcg);
        end
    end
end
out(3:end) = mean(temp);
fprintf('ItemKNN, RMSE is %f, MAE is %f\n',out(1:2));
fprintf('ItemKNN, NDCG is %f/%f/%f/%f/%f/%f/%f/%f/%f/%f\n',out(3:end));
end

%% Pearson correlation coefficient
function dist = pearson_sim(userID,oneUserVec,userMat,oneUserMean,userMean)
dist = zeros(size(userMat,1),1);
for i = userID+1:length(dist)
    interItem = find((oneUserVec~=0)&(userMat(i,:)~=0));
    if ~isempty(interItem)
        dist(i) = sum((oneUserVec(interItem)-oneUserMean).*(userMat(i,interItem)-userMean(i)))./...
            sqrt(sum((oneUserVec(interItem)-oneUserMean).^2))./sqrt(sum((userMat(i,interItem)-userMean(i)).^2));
    else
        dist(i) = 0;
    end
end
dist(isnan(dist)) = 0;
end

%% Adjusted Cosine similarity
function dist = cosine_sim(itemID,oneItemVec,itemMat,userMean)
dist = zeros(size(itemMat,2),1);
for i = itemID+1:length(dist)
    interUser = find((oneItemVec~=0)&(itemMat(:,i)~=0));
    if ~isempty(interUser)
        dist(i) = sum((oneItemVec(interUser)-userMean(interUser)).*(itemMat(interUser,i)-userMean(interUser)))./...
            sqrt(sum((oneItemVec(interUser)-userMean(interUser)).^2))./sqrt(sum((itemMat(interUser,i)-userMean(interUser)).^2));
    else
        dist(i) = 0;
    end
end
dist(isnan(dist)) = 0;
end