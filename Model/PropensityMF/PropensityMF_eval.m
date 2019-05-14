function out = PropensityMF_eval(testData,meanRating,U,V,bu,bv,topN)
%%
%   Input:
%       testData: Test dataset
%       meanRating: Average rating
%       U: User feature
%       V: Item feature
%       bu: User bias
%       bv: Item bias
%       topN: Num of recommended list
%   Output:
%       out: Evaluating metrics
%%
out = zeros(1,12);
pred = meanRating+sum(U(testData(:,1),:).*V(testData(:,2),:),2)+bu(testData(:,1))+bv(testData(:,2));
pred(pred>5) = 5;
pred(pred<1) = 1;
out(1) = sqrt(sum((testData(:,3)-pred).^2)/size(testData,1));
out(2) = sum(abs(testData(:,3)-pred))/size(testData,1);
[userSet,p] = numunique(testData(:,1));
temp = zeros(length(userSet),topN);
for i = 1:length(userSet)
    for j = 1:topN
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
end