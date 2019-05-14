%% Yahoo! R3 Dataset - biasedMF
load('yahoo_user.mat')
load('yahoo_random.mat');
out1 = biasedMF(train,test,'lr',5);
%% Coat Shopping Dataset - biasedMF
load('coat_user.mat')
load('coat_random.mat');
out2 = biasedMF(train,test,'m',290,'n',300,'lr',1);
%% Yahoo! R3 Dataset - PMF
load('yahoo_user.mat')
load('yahoo_random.mat');
out3 = PMF(train,test,'lr',10);
%% Coat Shopping Dataset - PMF
load('coat_user.mat')
load('coat_random.mat');
out4 = PMF(train,test,'m',290,'n',300,'lr',1);