function out = Logitvd_eval(testData,bet,gamm,theta,mu,par)
%%
%   Input:
%       testData: Test dataset
%       bet: Probability that the component z gives the rating v to item m
%       gamma, theta: Parameters for computing the posterior distribution
%       mu: probability of observing a rating given that its value is v
%       par: Default parameters
%   Output:
%       out: Evaluating metrics
%%
q_nk = repmat(theta,par.m,1);
for d = 1:par.n
    q_nk = q_nk.*squeeze(gamm(:,d,:));
    q_nk = q_nk./sum(q_nk,2);
end
for k = 1:par.C
    bet(:,k,:) = squeeze(bet(:,k,:)).*(1-mu);
end
bet = bet./sum(bet,3);
predDist = zeros(size(testData,1),par.R);
for v = 1:par.R
    predDist(:,v) = sum(bet(testData(:,2),:,v).*q_nk(testData(:,1),:),2);
end
temp = cumsum(predDist,2);
medianPred = arrayfun(@(x) find(temp(x,:)>=0.5,1),(1:size(temp,1))');
meanPred = sum(predDist.*(1:par.R),2);
out = zeros(1,12);
out(1) = sqrt(sum((testData(:,3)-medianPred).^2)/size(testData,1));
out(2) = sum(abs(testData(:,3)-medianPred))/size(testData,1);
[userSet,p] = numunique(testData(:,1));
temp = zeros(length(userSet),10);
for i = 1:length(userSet)
    for j = 1:par.topN
        r = (1:j)';
        if length(p{i})>r(end)
            rel = testData(p{i},3);
            [~,I] = sort(meanPred(p{i}),'descend');
            dcg = sum((2.^rel(I(r))-1)./log2(r+1));
            [~,I] = sort(rel,'descend');
            idcg = sum((2.^rel(I(r))-1)./log2(r+1));
            temp(i,j) = sum(dcg/idcg);
        else
            tr = (1:length(p{i}))';
            rel = testData(p{i},3);
            [~,I] = sort(meanPred(p{i}),'descend');
            dcg = sum((2.^rel(I(tr))-1)./log2(tr+1));
            [~,I] = sort(rel,'descend');
            idcg = sum((2.^rel(I(tr))-1)./log2(tr+1));
            temp(i,j) = sum(dcg/idcg);
        end
    end
end
out(3:end) = mean(temp);