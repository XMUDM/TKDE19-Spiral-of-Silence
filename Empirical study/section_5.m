%% Experiments codes for the section 5
fileAddress = 'F:\Matlab_workspace\TKDE19_SpiralSilence\Empirical study\';
mkdir([fileAddress,'The results of section 5'])
%% Table 9
fileAddress = 'F:\Matlab_workspace\TKDE19_SpiralSilence\Empirical study\';
load([fileAddress,'dataset\Yahoo_user.mat']);
load([fileAddress,'dataset\Yahoo_random.mat']);
[itemSet,P]=numunique(train(:,2));
itemMean=arrayfun(@(x) mean(train(P{x},3)),1:length(itemSet));
[trainUserSet,trainP]=numunique(train(:,1));
[testUserSet,testP]=numunique(test(:,1));
tab=zeros(3,5);
for h=0.4:0.1:0.8
    trainScore=zeros(length(trainUserSet),1);
    testScore=zeros(length(testUserSet),1);
    for i=1:length(trainUserSet)
        tmp=train(trainP{i},3)'-itemMean(train(trainP{i},2));
        trainScore(i)=length(find(abs(tmp)>1.7))/length(trainP{i});
    end
    for i=1:length(testUserSet)
        tmp=test(testP{i},3)'-itemMean(test(testP{i},2));
        testScore(i)=length(find(abs(tmp)>1.7))/length(testP{i});
    end
    if h==0.4
        trainHardcore=find(trainScore<0.5);
        trainHardcore(trainHardcore>5400)=[];
        testHardcore=find(testScore>=0.5);
    else
        trainHardcore=find(trainScore>=h);
        trainHardcore(trainHardcore>5400)=[];
        testHardcore=find(testScore>=h);
    end
    tab(1,int8((h-0.3)*10))=length(intersect(trainHardcore,testHardcore))/...
        length(union(trainHardcore,testHardcore));
    hardcoreProb=length(trainHardcore)/5400;
    sampleResult=zeros(1000,1);
    for i=1:1000
        sampleHardcore=find(rand(5400,1)<=hardcoreProb);
        sampleResult(i)=length(intersect(sampleHardcore,testHardcore))/...
            length(union(sampleHardcore,testHardcore));
    end
    tab(2,int8((h-0.3)*10))=mean(sampleResult);
    tab(3,int8((h-0.3)*10))=ranksum(tab(1,int8((h-0.3)*10)),sampleResult,'tail','right');
end
save([fileAddress,'The results of section 5\Tab9.mat'],'tab');
%% Figure 2
% 8 real dataset
dataName={'Ciao','Epinion','Eachmovie','Movielens20m'};
fileAddress = 'F:\Matlab_workspace\TKDE19_SpiralSilence\Empirical study\';
for i=1:length(dataName)
    load([fileAddress,'dataset\',dataName{i},'.mat']);
    tmp=dataMat(:,3)-dataMat(:,5);
    [userSet,P]=numunique(dataMat(:,1));
    score=zeros(length(userSet),1);
    extremeRate=zeros(length(userSet),1);
    for j=1:length(userSet)
        score(j)=length(find(abs(tmp(P{j}))>1.7))/length(P{j});
        extremeRate(j)=length(find(dataMat(P{j},3)==1|dataMat(P{j},3)==5))/length(P{j});
    end
    groupIdx=zeros(length(userSet),1);
    groupIdx(score>=0.5)=1;
    bh=boxplot(extremeRate,groupIdx,'symbol','');
    set(bh,'linewidth',3);
    ylabel('p_u(extreme)');
    set(gca,'XTickLabel',{'non-hardcore','hardcore'});
    set(gca,'box','off');
    set(gca,'FontName','Arial Rounded MT Bold','FontSize',20,'linewidth',3);
    axis tight
    saveas(gcf,[fileAddress,'The results of section 5\',dataName{i},'_extreme'],'epsc');
    clf;
    fprintf([dataName{i},' process completed\n']);
end
%--------------------------------------------------------------------------
% Yahoo_user
load([fileAddress,'dataset\Yahoo_user.mat']);
[itemSet,P]=numunique(train(:,2));
itemMean=arrayfun(@(x) mean(train(P{x},3)),1:length(itemSet));
tmp=train(:,3)'-itemMean(train(:,2));
[userSet,P]=numunique(train(:,1));
score=zeros(length(userSet),1);
extremeRate=zeros(length(userSet),1);
for i=1:length(userSet)
    score(i)=length(find(abs(tmp(P{i}))>1.7))/length(P{i});
    extremeRate(i)=length(find(train(P{i},3)==1|train(P{i},3)==5))/length(P{i});
end
groupIdx=zeros(length(userSet),1);
groupIdx(score>=0.5)=1;
bh=boxplot(extremeRate,groupIdx,'symbol','');
set(bh,'linewidth',3);
ylabel('p_u(extreme)');
set(gca,'XTickLabel',{'non-hardcore','hardcore'});
set(gca,'box','off');
set(gca,'FontName','Arial Rounded MT Bold','FontSize',20,'linewidth',3);
axis tight
saveas(gcf,[fileAddress,'The results of section 5\Yahoo_user_extreme'],'epsc');
clf;
fprintf('Yahoo_user process completed\n');
%--------------------------------------------------------------------------
% Yahoo_random
load([fileAddress,'dataset\Yahoo_random.mat']);
tmp=test(:,3)'-itemMean(test(:,2));
[userSet,P]=numunique(test(:,1));
score=zeros(length(userSet),1);
extremeRate=zeros(length(userSet),1);
for i=1:length(userSet)
    score(i)=length(find(abs(tmp(P{i}))>1.7))/length(P{i});
    extremeRate(i)=length(find(test(P{i},3)==1|test(P{i},3)==5))/length(P{i});
end
groupIdx=zeros(length(userSet),1);
groupIdx(score>=0.5)=1;
bh=boxplot(extremeRate,groupIdx,'symbol','');
set(bh,'linewidth',3);
ylabel('p_u(extreme)');
set(gca,'XTickLabel',{'non-hardcore','hardcore'});
set(gca,'box','off');
set(gca,'FontName','Arial Rounded MT Bold','FontSize',20,'linewidth',3);
axis tight
saveas(gcf,[fileAddress,'The results of section 5\Yahoo_random_extreme'],'epsc');
clf;
fprintf('Yahoo_random process completed\n');
%% Table 10
fileAddress = 'F:\Matlab_workspace\TKDE19_SpiralSilence\Empirical study\';
tab=zeros(4,3);
%Ciao and Epinion
dataName={'Ciao_tag','Epinion_tag'};
for i=1:length(dataName)
    load([fileAddress,'dataset\',dataName{i},'.mat']);
    [userSet,P]=numunique(dataMat(:,1));
    userRate=nan*ones(length(userSet),2);
    for j=1:length(userSet)
        tmp=unique(dataMat(P{j},3));
        num=arrayfun(@(x)sum(dataMat(P{j},3)==x),tmp);
        mostTag=tmp(num==max(num));
        leastTag=tmp(num==min(num));
        if ~isequal(unique(mostTag),unique(leastTag))
            tmp=dataMat(P{j},4)-dataMat(P{j},6);
            userRate(j,1)=mean(arrayfun(@(x) length(find(abs(tmp(dataMat(P{j},3)==mostTag(x)))>1.7))/max(num),1:length(mostTag)));
            userRate(j,2)=mean(arrayfun(@(x) length(find(abs(tmp(dataMat(P{j},3)==leastTag(x)))>1.7))/min(num),1:length(leastTag)));
        end
    end
    tab(i,1)=quantile(userRate(:,1),0.6);
    tab(i,2)=quantile(userRate(:,2),0.6);
    tab(i,3)=ranksum(userRate(:,1),userRate(:,2),'tail','right');
end
%--------------------------------------------------------------------------
%Movielens 20M and Eachmovie
dataName={'Movielens20m','Eachmovie'};
for i=1:length(dataName)
    load([fileAddress,'dataset\',dataName{i},'.mat']);
    load([fileAddress,'dataset\',dataName{i},'_tag.mat']);
    itemID=tag(:,1);
    itemTag=tag(:,2:end);
    [userSet,P]=numunique(dataMat(:,1));
    userRate=nan*ones(length(userSet),2);
    for j=1:length(userSet)
        tmp=abs(dataMat(P{j},3)-dataMat(P{j},5));
        userTag=zeros(1,max(max(itemTag)));
        tagSet=cell(1,max(max(itemTag)));
        for k=1:length(P{j})
            itemIdx=find(itemID==dataMat(P{j}(k),2));
            userTag(itemTag(itemIdx,itemTag(itemIdx,:)>0))=userTag(itemTag(itemIdx,itemTag(itemIdx,:)>0))+1;
            tagIdx=itemTag(itemIdx,itemTag(itemIdx,:)>0);
            for h=1:length(tagIdx)
                tagSet{tagIdx(h)}=[tagSet{tagIdx(h)} tmp(k)]; 
            end
        end
        if isempty(tagIdx)
            continue;
        end
        mostTag=find(userTag==max(userTag));
        leastTag=find(userTag==min(userTag(userTag~=0)));
        if ~isequal(unique(mostTag),unique(leastTag))
            userRate(j,1)=mean(arrayfun(@(x) length(find(tagSet{x}>1.7))/max(userTag),mostTag));
            userRate(j,2)=mean(arrayfun(@(x) length(find(tagSet{x}>1.7))/min(userTag(userTag~=0)),leastTag));
        end
    end
    tab(i+2,1)=quantile(userRate(:,1),0.6);
    tab(i+2,2)=quantile(userRate(:,2),0.6);
    tab(i+2,3)=ranksum(userRate(:,1),userRate(:,2),'tail','right');
end
save([fileAddress,'The results of section 5\Tab10.mat'],'tab');
%% Figure 3
dataName={'Ciao','Epinion','Eachmovie','Movielens20m'};
fileAddress = 'F:\Matlab_workspace\TKDE19_SpiralSilence\Empirical study\';
for i=1:length(dataName)
    load([fileAddress,'dataset\',dataName{i},'.mat']);
    tmp=dataMat(:,3)-dataMat(:,5);
    [itemSet,itemP]=numunique(dataMat(:,2));
    itemMean=zeros(1,max(dataMat(:,2)));
    for j=1:length(itemSet)
        itemMean(itemSet(j))=mean(dataMat(itemP{j},3));
    end
    [userSet,P]=numunique(dataMat(:,1));
    userRate=nan*ones(length(userSet),2);
    for j=1:length(userSet)
        badIdx=find(itemMean(dataMat(P{j},2))<3);
        bestIdx=find(itemMean(dataMat(P{j},2))>=3);
        if ~isempty(badIdx)&&~isempty(bestIdx)
            userRate(j,1)=length(find(tmp(P{j}(badIdx))>1.7))/length(badIdx);
            userRate(j,2)=length(find(tmp(P{j}(bestIdx))<-1.7))/length(bestIdx);
        end
    end
    bh=boxplot([userRate(:,2),userRate(:,1)],[zeros(1,size(userRate,1)),ones(1,size(userRate,1))],'symbol','');
    set(bh,'linewidth',3);
    ylabel('h');
    set(gca,'XTickLabel',{'CP','PN'});
    set(gca,'box','off');
    set(gca,'FontName','Arial Rounded MT Bold','FontSize',20,'linewidth',3);
    axis tight
    ylim([0 0.5])
    saveas(gcf,[fileAddress,'The results of section 5\',dataName{i},'_moral'],'epsc');
    clf;
    fprintf([dataName{i},' process completed\n']);
end
%--------------------------------------------------------------------------
% Yahoo_user
load([fileAddress,'dataset\Yahoo_user.mat']);
[itemSet,P]=numunique(train(:,2));
itemMean=arrayfun(@(x) mean(train(P{x},3)),1:length(itemSet));
tmp=train(:,3)'-itemMean(train(:,2));
[userSet,P]=numunique(train(:,1));
userRate=nan*ones(length(userSet),2);
for i=1:length(userSet)
    badIdx=find(itemMean(train(P{i},2))<3);
    bestIdx=find(itemMean(train(P{i},2))>=3);
    if ~isempty(badIdx)&&~isempty(bestIdx)
        userRate(i,1)=length(find(tmp(P{i}(badIdx))>1.7))/length(badIdx);
        userRate(i,2)=length(find(tmp(P{i}(bestIdx))<-1.7))/length(bestIdx);
    end
end
bh=boxplot([userRate(:,2),userRate(:,1)],[zeros(1,size(userRate,1)),ones(1,size(userRate,1))],'symbol','');
set(bh,'linewidth',3);
ylabel('h');
set(gca,'XTickLabel',{'CP','PN'});
set(gca,'box','off');
set(gca,'FontName','Arial Rounded MT Bold','FontSize',20,'linewidth',3);
axis tight
saveas(gcf,[fileAddress,'The results of section 5\Yahoo_user_moral'],'epsc');
clf;
fprintf('Yahoo_user process completed\n');
%--------------------------------------------------------------------------
% Yahoo_random
load([fileAddress,'dataset\Yahoo_random.mat']);
tmp=test(:,3)'-itemMean(test(:,2));
[userSet,P]=numunique(test(:,1));
userRate=nan*ones(length(userSet),2);
for i=1:length(userSet)
    badIdx=find(itemMean(test(P{i},2))<3);
    bestIdx=find(itemMean(test(P{i},2))>=3);
    if ~isempty(badIdx)&&~isempty(bestIdx)
        userRate(i,1)=length(find(tmp(P{i}(badIdx))>1.7))/length(badIdx);
        userRate(i,2)=length(find(tmp(P{i}(bestIdx))<-1.7))/length(bestIdx);
    end
end
bh=boxplot([userRate(:,2),userRate(:,1)],[zeros(1,size(userRate,1)),ones(1,size(userRate,1))],'symbol','');
set(bh,'linewidth',3);
ylabel('h');
set(gca,'XTickLabel',{'CP','PN'});
set(gca,'box','off');
set(gca,'FontName','Arial Rounded MT Bold','FontSize',20,'linewidth',3);
axis tight
saveas(gcf,[fileAddress,'The results of section 5\Yahoo_random_moral'],'epsc');
clf;
fprintf('Yahoo_random process completed\n');