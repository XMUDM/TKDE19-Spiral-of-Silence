%% Experiments codes for the section 3
fileAddress = 'F:\Matlab_workspace\TKDE19_SpiralSilence\Empirical study\';
mkdir([fileAddress,'The results of section 3'])
%% Table 3
% Threshold-based
fileAddress = 'F:\Matlab_workspace\TKDE19_SpiralSilence\Empirical study\';
load([fileAddress,'dataset\EEpinions.mat']);
[x,p]=numunique(dataMat(:,1));
missIdx=find(dataMat(:,4)==1);
obsIdx=find(dataMat(:,4)==0);
divergence=zeros(size(dataMat,1),1);
for i=1:length(x)
    divergence(p{i})=dataMat(p{i},3)-dataMat(p{i},6);
end
pResult=zeros(11,100);
qResult=zeros(11,100);
for i=1:100
    tmp=randsample(1:length(obsIdx),length(missIdx));
    missDivergence=divergence(missIdx);
    obsDivergence=divergence(obsIdx(tmp));
    majorityRange=prctile(obsDivergence,[15:25,85:-1:75]);
    majorityRange=reshape(majorityRange,[11,2]);
    for j=1:size(majorityRange,1)
        majorityTotal=length(find(obsDivergence>=majorityRange(j,1)&obsDivergence<=majorityRange(j,2)))+...
            length(find(missDivergence>=majorityRange(j,1)&missDivergence<=majorityRange(j,2)));
        p=length(find(obsDivergence>=majorityRange(j,1)&obsDivergence<=majorityRange(j,2)))/majorityTotal;
        minorityTotal=length(find(obsDivergence<majorityRange(j,1)|obsDivergence>majorityRange(j,2)))+...
            length(find(missDivergence<majorityRange(j,1)|missDivergence>majorityRange(j,2)));
        q=length(find(obsDivergence<majorityRange(j,1)|obsDivergence>majorityRange(j,2)))/minorityTotal;
        pResult(j,i)=p;
        qResult(j,i)=q;
    end
    fprintf('Threshold-based: Boost %d completed\n',i);
end
thresholdOut=zeros(11,2);
thresholdOut(:,1)=mean(pResult,2);
thresholdOut(:,2)=mean(qResult,2);
save([fileAddress,'The results of section 3\Tab3_Threshold.mat'],'thresholdOut');
%--------------------------------------------------------------------------
% Model-based
fileAddress = 'F:\Matlab_workspace\TKDE19_SpiralSilence\Empirical study\';
load([fileAddress,'dataset\EEpinions.mat']);
[x,p]=numunique(dataMat(:,1));
idx=ones(size(dataMat,1),1);
idx(dataMat(:,4)==1)=0;
divergence=zeros(size(dataMat,1),1);
for i=1:length(x)
    divergence(p{i})=dataMat(p{i},3)-dataMat(p{i},6);
end
dataMat=[divergence,double(idx)];
result=zeros(100,7);
for i=1:100
    obsIdx=find(dataMat(:,2)==1);
    tmp=randsample(1:length(obsIdx),length(find(dataMat(:,2)==0)));
    train=[dataMat(obsIdx(tmp),:);dataMat(dataMat(:,2)==0,:)];
    train=train(randperm(size(train,1)),:);
    maxIter=100;
    majoritySigma=1;
    minoritySigma=1;
    majorityMu=0;
    minorityMu=1;
    phi=0.9;
    p=0.5;
    q=0.5;
    oldOut=0;
    for j=1:maxIter
        % E-step
        tmp=phi*normpdf(train(:,1),majorityMu,majoritySigma).*p.^train(:,2).*(1-p).^(1-train(:,2))+...
            (1-phi)*normpdf(abs(train(:,1)),minorityMu,minoritySigma).*q.^train(:,2).*(1-q).^(1-train(:,2));
        majorityProb=(phi*normpdf(train(:,1),majorityMu,majoritySigma).*p.^train(:,2).*(1-p).^(1-train(:,2)))./tmp;
        %minorityProb=((1-phi)*normpdf(abs(train(:,1)),minorityMu,minoritySigma).*q.^train(:,2).*(1-q).^(1-train(:,2)))./tmp;
        minorityProb=1-majorityProb;
        % M-step
        phi=sum(majorityProb)/size(train,1);
        p=sum(majorityProb.*train(:,2))/sum(majorityProb);
        q=sum(minorityProb.*train(:,2))/sum(minorityProb);
        majoritySigma=sqrt(sum(majorityProb.*train(:,1).^2)/sum(majorityProb));
        minoritySigma=sqrt(sum(minorityProb.*(abs(train(:,1))-minorityMu).^2)/sum(minorityProb));
        minorityMu=sum(minorityProb.*abs(train(:,1)))/sum(minorityProb);
        % Convergence condition
        tmp1=sum(normpdf(train(train(:,2)==1,1),majorityMu,majoritySigma)*p*phi+...
            normpdf(abs(train(train(:,2)==1,1)),minorityMu,minoritySigma)*q*(1-phi));
        tmp2=sum(normpdf(train(train(:,2)==0,1),majorityMu,majoritySigma)*(1-p)*phi+...
            normpdf(abs(train(train(:,2)==0,1)),minorityMu,minoritySigma)*(1-q)*(1-phi));
        out=(tmp1+tmp2)/size(train,1);
        if out<=1&&out-oldOut>=10e-4
            result(i,:)=[phi,p,q,majorityMu,majoritySigma,minorityMu,minoritySigma];
            oldOut=out;
        else
            break;
        end
    end
    result(i,:)=[phi,p,q,majorityMu,majoritySigma,minorityMu,minoritySigma];
    fprintf('Model-based: Boost %d completed\n',i);
end
modelOut=mean(result);
save([fileAddress,'The results of section 3\Tab3_Modelbased.mat'],'modelOut');
%% Table 4
% Threshold-based
fileAddress = 'F:\Matlab_workspace\TKDE19_SpiralSilence\Empirical study\';
load([fileAddress,'dataset\EEpinions.mat']);
[x,P]=numunique(dataMat(:,1));
divergence=zeros(size(dataMat,1),1);
for i=1:length(x)
    divergence(P{i})=dataMat(P{i},3)-dataMat(P{i},6);
end
thresholdOut=zeros(7,2);
thresholdParaRec=cell(7,1);
for h=4:10
    label=zeros(size(dataMat,1),1);
    for i=1:length(x)
        if length(P{i})>=50
            tmp=discretize(1:length(P{i}),h);
            label(P{i})=tmp;
        end
    end
    partResult=zeros(h,2);
    paraResult=zeros(h,2);
    for k=1:h
        trainData=dataMat(label==k,:);
        trainDivergence=divergence(label==k);
        boostResult=zeros(100,2);
        boostParaResult=zeros(100,2);
        missIdx=find(trainData(:,4)==1);
        obsIdx=find(trainData(:,4)==0);
        for i=1:100
            tmp=randsample(1:length(obsIdx),length(missIdx));
            missDivergence=trainDivergence(missIdx);
            obsDivergence=trainDivergence(obsIdx(tmp));
            majorityRange=prctile(obsDivergence,[25,75]);
            majorityTotal=length(find(obsDivergence>=majorityRange(1)&obsDivergence<=majorityRange(2)))+...
                length(find(missDivergence>=majorityRange(1)&missDivergence<=majorityRange(2)));
            p=length(find(obsDivergence>=majorityRange(1)&obsDivergence<=majorityRange(2)))/majorityTotal;
            minorityTotal=length(find(obsDivergence<majorityRange(1)|obsDivergence>majorityRange(2)))+...
                length(find(missDivergence<majorityRange(1)|missDivergence>majorityRange(2)));
            q=length(find(obsDivergence<majorityRange(1)|obsDivergence>majorityRange(2)))/minorityTotal;
            boostResult(i,:)=[p,q];
            boostParaResult(i,:)=majorityRange;
        end
        partResult(k,:)=mean(boostResult);
        paraResult(k,:)=mean(boostParaResult);
    end
    thresholdOut(h-3,:)=partResult(end,:)-partResult(1,:);
    thresholdParaRec{h-3}=paraResult;
    fprintf('Threshold-based: Snapshots %d completed\n',h);
end
save([fileAddress,'The results of section 3\Tab4_Threshold.mat'],'thresholdOut');
%--------------------------------------------------------------------------
% Model-based
fileAddress = 'F:\Matlab_workspace\TKDE19_SpiralSilence\Empirical study\';
load([fileAddress,'dataset\EEpinions.mat']);
[x,P]=numunique(dataMat(:,1));
idx=ones(size(dataMat,1),1);
idx(dataMat(:,4)==1)=0;
divergence=zeros(size(dataMat,1),1);
for i=1:length(x)
    divergence(P{i})=dataMat(P{i},3)-dataMat(P{i},6);
end
dataMat=[divergence,double(idx)];
modelOut=zeros(7,2);
for h=4:10
    label=zeros(size(dataMat,1),1);
    for i=1:length(x)
        if length(P{i})>=50
            tmp=discretize(1:length(P{i}),h);
            label(P{i})=tmp;
        end
    end
    partResult=zeros(h,7);
    for k=1:h
        trainData=dataMat(label==k,:);
        boostResult=zeros(100,7);
        for i=1:100
            obsIdx=find(trainData(:,2)==1);
            tmp=randsample(1:length(obsIdx),length(find(trainData(:,2)==0)));
            train=[trainData(obsIdx(tmp),:);trainData(trainData(:,2)==0,:)];
            train=train(randperm(size(train,1)),:);
            train=train(randperm(size(train,1)),:);
            maxIter=200;
            majoritySigma=1;
            minoritySigma=1;
            majorityMu=0;
            minorityMu=1;
            phi=0.9;
            p=0.5;
            q=0.5;
            oldOut=0;
            for j=1:maxIter
                % E-step
                tmp=phi*normpdf(train(:,1),majorityMu,majoritySigma).*p.^train(:,2).*(1-p).^(1-train(:,2))+...
                    (1-phi)*normpdf(abs(train(:,1)),minorityMu,minoritySigma).*q.^train(:,2).*(1-q).^(1-train(:,2));
                majorityProb=(phi*normpdf(train(:,1),majorityMu,majoritySigma).*p.^train(:,2).*(1-p).^(1-train(:,2)))./tmp;
                %minorityProb=((1-phi)*normpdf(abs(train(:,1)),minorityMu,minoritySigma).*q.^train(:,2).*(1-q).^(1-train(:,2)))./tmp;
                minorityProb=1-majorityProb;
                % M-step
                phi=sum(majorityProb)/size(train,1);
                p=sum(majorityProb.*train(:,2))/sum(majorityProb);
                q=sum(minorityProb.*train(:,2))/sum(minorityProb);
                majoritySigma=sqrt(sum(majorityProb.*train(:,1).^2)/sum(majorityProb));
                minoritySigma=sqrt(sum(minorityProb.*(abs(train(:,1))-minorityMu).^2)/sum(minorityProb));
                minorityMu=sum(minorityProb.*abs(train(:,1)))/sum(minorityProb);
                % Convergence condition
                tmp1=sum(normpdf(train(train(:,2)==1,1),majorityMu,majoritySigma)*p*phi+...
                    normpdf(abs(train(train(:,2)==1,1)),minorityMu,minoritySigma)*q*(1-phi));
                tmp2=sum(normpdf(train(train(:,2)==0,1),majorityMu,majoritySigma)*(1-p)*phi+...
                    normpdf(abs(train(train(:,2)==0,1)),minorityMu,minoritySigma)*(1-q)*(1-phi));
                out=(tmp1+tmp2)/size(train,1);
                if out<=1&&out-oldOut>=10e-4
                    boostResult(i,:)=[phi,p,q,majorityMu,majoritySigma,minorityMu,minoritySigma];
                    oldOut=out;
                else
                    break;
                end
            end
        end
        partResult(k,:)=mean(boostResult);
    end
    modelOut(h-3,:)=partResult(end,2:3)-partResult(1,2:3);
    fprintf('Model-based: Snapshots %d completed\n',h);
end
save([fileAddress,'The results of section 3\Tab4_Modelbased.mat'],'modelOut');