function out = modelEval(testData,bu,bv,U,V,topN)
out = zeros(1,12);
pred = sum(U(testData(:,1),:).*V(testData(:,2),:),2)+bu(testData(:,1))+bv(testData(:,2));
accPred = pred;
accPred(accPred>5) = 5;
accPred(accPred<1) = 1;
out(1) = sqrt(sum((testData(:,3)-accPred).^2)/size(testData,1));
out(2) = sum(abs(testData(:,3)-accPred))/size(testData,1);
[userSet,p] = numunique(testData(:,1));
temp = zeros(length(userSet),10);
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


