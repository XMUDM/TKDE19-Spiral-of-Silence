%% Fig2
dataName = {'Books','Clothes','Electronics','Movies','Epinion','Ciao','Eachmovie','Movielens20m'};
fileAddress = 'F:\Matlab_workspace\TKDE19_SpiralSilence\Empirical study\';
for i = 1:length(dataName)
    load([fileAddress,'dataset\',dataName{i},'.mat']);
    [userSet,userP] = numunique(dataMat(:,1));
    userRatingNum = arrayfun(@(x) length(userP{x}),1:length(userSet));
    activeTime = arrayfun(@(x) max(dataMat(userP{x},4))-min(dataMat(userP{x},4)),1:length(userSet));
    [itemSet,itemP] = numunique(dataMat(:,2));
    itemMean = arrayfun(@(x) mean(dataMat(itemP{x},3)),1:length(itemSet));
    % Current
    currentMeanDivergence = dataMat(:,3)-dataMat(:,5);
    fprintf('dataset/Method: %s/Current completed\n',dataName{i});
    % Final
    finalRatingDivergence = cell(length(itemP),1);
    for j = 1:length(itemP)
        finalRatingDivergence{j} = dataMat(itemP{j},3)-itemMean(j);
    end
    finalRatingDivergence = cell2mat(finalRatingDivergence);
    fprintf('dataset/Method: %s/Final completed\n',dataName{i});
    % Timely
    mostTimelyDivergence = cell(length(userP),1);
    for j = 1:length(itemP)
        temp = round(0.05*length(itemP{j}));
        mostTimelyDivergence{j} = dataMat(itemP{j}(temp+1:end),3)-mean(dataMat(itemP{j}(1:temp),3));
    end
    mostTimelyDivergence = cell2mat(mostTimelyDivergence);
    fprintf('dataset/Method: %s/Timely completed\n',dataName{i});
    % Active
    highestNumDivergence = cell(length(itemP),1);
    [~,idx] = sort(userRatingNum,'descend');
    highestNumUser = idx(1:round(0.05*length(idx)));
    for j = 1:length(itemP)
        temp = [];
        idx = ismember(dataMat(itemP{j},1),highestNumUser);
        idx = find(idx==1);
        if ~isempty(idx)
            for k = 1:length(idx)-1
                temp = [temp;dataMat(itemP{j}(idx(k)+1:idx(k+1)-1),3)-dataMat(itemP{j}(idx(k)),3)];
            end
            temp = [temp;dataMat(itemP{j}(idx(k+1)+1:end),3)-dataMat(itemP{j}(idx(k+1)),3)];
            highestNumDivergence{j} = temp;
        end
    end
    highestNumDivergence = cell2mat(highestNumDivergence);
    fprintf('dataset/Method: %s/Active completed\n',dataName{i});
    % Regular
    mostActiveDivergence = cell(length(itemP),1);
    [~,idx] = sort(activeTime,'descend');
    mostActiveUser = userSet(idx(1:round(0.05*length(idx))));
    for j = 1:length(itemP)
        temp = [];
        idx = ismember(dataMat(itemP{j},1),mostActiveUser);
        idx = find(idx==1);
        if ~isempty(idx)
            for k = 1:length(idx)-1
                temp = [temp;dataMat(itemP{j}(idx(k)+1:idx(k+1)-1),3)-dataMat(itemP{j}(idx(k)),3)];
            end
            temp = [temp;dataMat(itemP{j}(idx(k+1)+1:end),3)-dataMat(itemP{j}(idx(k+1)),3)];
            mostActiveDivergence{j} = temp;
        end
    end
    mostActiveDivergence = cell2mat(mostActiveDivergence);
    fprintf('dataset/Method: %s/Regular completed\n',dataName{i});
    % Figure
    divergence = [currentMeanDivergence;finalRatingDivergence;highestNumDivergence;mostActiveDivergence;mostTimelyDivergence];
    groupIdx = [zeros(length(currentMeanDivergence),1);ones(length(finalRatingDivergence),1);2*ones(length(highestNumDivergence),1);3*ones(length(mostActiveDivergence),1);4*ones(length(mostTimelyDivergence),1)];
    bh = boxplot(divergence,groupIdx,'symbol','');
    set(bh,'linewidth',3);
    set(gca,'XTickLabel',{'current','final','active','regular','timely'});
    ylabel('rating divergence');
    set(gca,'box','off');
    set(gca,'FontName','Arial Rounded MT Bold','FontSize',20,'linewidth',3);
    axis tight 
    saveas(gcf,[fileAddress,dataName{i},'_opinion'],'epsc');
    clf;
end