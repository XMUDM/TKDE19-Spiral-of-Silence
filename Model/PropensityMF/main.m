%% Yahoo! R3 Dataset - PropensityMF
load('yahoo_user.mat')
load('yahoo_random.mat');
out1 = PropensityMF(train,test,'reg',25);
%% Coat Shopping Dataset - PropensityMF
load('coat_user.mat')
load('coat_random.mat');
out2 = PropensityMF(train,test,'m',290,'n',300,'reg',25,'estimation','LR','p','C:\Users\uesr\Desktop\dgliu\Model\PropensityMF\coat_propensities.ascii');