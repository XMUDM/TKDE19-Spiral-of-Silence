function [majorityOriRate,pluralityOriRate,majorityNewRate,pluralityNewRate,majorityRiseIdx,itemTrendHis]=spiralDiscreteProcess(dataMat,idx,pLevel)
%% Discrete
% Consider separately, because the Movielens20m's rating is 0.5:0.5:5 and the other is 1:1:5
if idx<8
    disLength=5;
    disInterval=6:10;
else
    disLength=10;
    disInterval=6:15;
end
[~,P]=numunique(dataMat(:,2));
itemRatingDis=zeros(length(P),disLength); 
itemTrendHis=cell(length(P),1); 
itemMajorityHis=cell(length(P),1);
itemKurtosis=nan*ones(length(P),1); 
pValue=nan*ones(length(P),1); 
for i=1:length(P)
    if length(P{i})>=50 % Remove items taht less than 50 rating
        itemRatingDis(i,:)=dataMat(P{i}(end),disInterval);
        % 0 in the distribution will lead to the inaccuracy of the
        % kurtosis, fix it with 1
        correctItemRatingDis=itemRatingDis(i,:);
        correctItemRatingDis(correctItemRatingDis==0)=1;
        % Consider separately,as mentioned above
        if idx<8
            tmp=[ones(1,correctItemRatingDis(1)),2*ones(1,correctItemRatingDis(2)),3*ones(1,correctItemRatingDis(3)),4*ones(1,correctItemRatingDis(4)),5*ones(1,correctItemRatingDis(5))];
        else
            tmp=[0.5*ones(1,correctItemRatingDis(1)),1*ones(1,correctItemRatingDis(2)),1.5*ones(1,correctItemRatingDis(3)),2*ones(1,correctItemRatingDis(4)),2.5*ones(1,correctItemRatingDis(5)),...
                3*ones(1,correctItemRatingDis(6)),3.5*ones(1,correctItemRatingDis(7)),4*ones(1,correctItemRatingDis(8)),4.5*ones(1,correctItemRatingDis(9)),5*ones(1,correctItemRatingDis(10))]; 
        end
        itemKurtosis(i)=kurtosis(tmp); % Calculated kurtosis
    end
end
for i=1:length(P) 
    if length(P{i})>=50
        if idx<8
            % The changing sequence of majority ratio over time
            itemRatingDis(i,:)=dataMat(P{i}(end),6:10)/sum(dataMat(P{i}(end),6:10));
            tmp=[sum(itemRatingDis(i,1:3)),sum(itemRatingDis(i,2:4)),sum(itemRatingDis(i,3:5))]; 
            [~,majorityIdx]=max(tmp);
            tmp=sum(dataMat(P{i},6:10),2);
            data=dataMat(P{i},6:10)./tmp;
            tmp=arrayfun(@(x) sum(data(x,majorityIdx:majorityIdx+2)),11:10:length(P{i})); 
            pValue(i)=cumKendallTest(tmp',(1:length(tmp))','gt'); % Mann-Kendall trend test
            itemTrendHis{i}=tmp;
            % The changing sequence of the majority over time
            itemRatingDis1=dataMat(P{i},6:10)./sum(dataMat(P{i},6:10),2);
            tmp=arrayfun(@(x) [sum(itemRatingDis1(x,1:3)),sum(itemRatingDis1(x,2:4)),sum(itemRatingDis1(x,3:5))],...
                (11:10:length(P{i}))','UniformOutput',false); 
            tmp=cell2mat(tmp);
            [~,majorityIdx]=max(tmp,[],2);
            itemMajorityHis{i}=majorityIdx;
        else
            % The changing sequence of majority ratio over time
            itemRatingDis(i,:)=dataMat(P{i}(end),6:15)/sum(dataMat(P{i}(end),6:15));
            tmp=[sum(itemRatingDis(i,1:6)),sum(itemRatingDis(4,8)),sum(itemRatingDis(i,6:10))]; 
            [~,majorityIdx]=max(tmp);
            tmp=sum(dataMat(P{i},6:15),2);
            data=dataMat(P{i},6:15)./tmp;
            if majorityIdx==1
                tmp=arrayfun(@(x) sum(data(x,majorityIdx:majorityIdx+5)),11:10:length(P{i}));
            else
                tmp=arrayfun(@(x) sum(data(x,2*majorityIdx:2*majorityIdx+4)),11:10:length(P{i})); 
            end
            pValue(i)=cumKendallTest(tmp',(1:length(tmp))','gt'); % Mann-Kendall trend test
            itemTrendHis{i}=tmp;
            % The changing sequence of the majority over time
            itemRatingDis1=dataMat(P{i},6:15)./sum(dataMat(P{i},6:15),2);
            tmp=arrayfun(@(x) [sum(itemRatingDis1(x,1:6)),sum(itemRatingDis1(x,4:8)),sum(itemRatingDis1(x,6:10))],...
                (11:10:length(P{i}))','UniformOutput',false); 
            tmp=cell2mat(tmp);
            [~,majorityIdx]=max(tmp,[],2);
            itemMajorityHis{i}=majorityIdx;
        end
    end
end
% If item's kurtosis less than 3, we think the item has not dominant majority
% opinion, and remove the idx that Mann-Kendall trend test can not be
% calculated.
pluralityIdx=setdiff(find(itemKurtosis<3),find(isnan(pValue)==1)); 
% If item's kurtosis more than 3, we think the item has dominant majority
% opinion, and remove the idx that Mann-Kendall trend test can not be
% calculated.
majorityIdx=setdiff(find(itemKurtosis>=3),find(isnan(pValue)==1)); 
%% Original
majorityOriRate=length(find((pValue(majorityIdx)<=pLevel&pValue(majorityIdx)>=0)==1))/...
    length(majorityIdx);
majorityRiseIdx=majorityIdx((pValue(majorityIdx)<=pLevel&pValue(majorityIdx)>=0)==1);
pluralityOriRate=length(find((pValue(pluralityIdx)<=pLevel&pValue(pluralityIdx)>=0)==1))/...
    length(pluralityIdx);
%% Correction
tmp=arrayfun(@(x) length(unique(itemMajorityHis{x})),majorityIdx);
% Add items that it reaches a dead end
majorityNewRate=length(union(find(tmp==1),find((pValue(majorityIdx)<=pLevel&pValue(majorityIdx)>=0)==1)))/...
    length(majorityIdx);
tmp=arrayfun(@(x) length(unique(itemMajorityHis{x})),pluralityIdx);
% Remove items that majority opinion is fluctuating
pluralityNewRate=length(setdiff(find((pValue(pluralityIdx)<=pLevel&pValue(pluralityIdx)>=0)==1),find(tmp~=1)))/...
    length(pluralityIdx);
end