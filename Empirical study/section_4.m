%% Experiments codes for the section 4
fileAddress = 'F:\Matlab_workspace\TKDE19_SpiralSilence\Empirical study\';
mkdir([fileAddress,'The results of section 4'])
%% Table 6 and Table 7
dataName={'Books','Clothes','Electronics','Movies','Epinion','Ciao','Eachmovie','Movielens20m'};
fileAddress = 'F:\Matlab_workspace\TKDE19_SpiralSilence\Empirical study\';
pLevel=[0.01,0.05,0.1];
numericalOut=zeros(8,6);
discreteOut=zeros(8,6);
numericalIdx=cell(8,3);
discreteIdx=cell(8,3);
numericalHis=cell(8,1);
discreteHis=cell(8,1);
for i=1:length(dataName)
    for j=1:length(pLevel)
        load([fileAddress,'dataset\',dataName{i},'.mat']);
        [~,~,majorityRate,pluralityRate,majorityRiseIdx,itemTrendHis]=spiralContinuousProcess(dataMat,i,pLevel(j));
        numericalIdx{i,j}=majorityRiseIdx;
        numericalHis{i}=itemTrendHis;
        numericalOut(i,2*j-1:2*j)=[majorityRate,pluralityRate];
        [~,~,majorityRate,pluralityRate,majorityRiseIdx,itemTrendHis]=spiralDiscreteProcess(dataMat,i,pLevel(j));
        discreteIdx{i,j}=majorityRiseIdx;
        discreteHis{i}=itemTrendHis;
        discreteOut(i,2*j-1:2*j)=[majorityRate,pluralityRate];
        fprintf([dataName{i},' ',num2str(pLevel(j)),' process completed\n']);
    end
end
save([fileAddress,'The results of section 4\Tab5.mat'],'numericalOut');
save([fileAddress,'The results of section 4\Tab6.mat'],'discreteOut');
save([fileAddress,'The results of section 4\Output.mat'],'numericalIdx','discreteIdx','numericalHis','discreteHis');