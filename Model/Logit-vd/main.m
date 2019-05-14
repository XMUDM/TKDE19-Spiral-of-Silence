%% Yahoo! R3 Dataset - Logit-vd
load('yahoo_user.mat')
load('yahoo_random.mat');
out1 = Logitvd(train,test);
%% Coat Shopping Dataset - Logit-vd
load('coat_user.mat')
load('coat_random.mat');
out2 = Logitvd(train,test,'m',290,'n',300,'C',5,'adaptive','n','maxIter',150);