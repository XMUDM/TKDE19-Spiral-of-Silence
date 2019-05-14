%% Yahoo! R3 Dataset - RAPMF
load('yahoo_user.mat')
load('yahoo_random.mat');
out1 = RAPMF(train,test,'sigU',0.04,'sigV',0.04,'sigR',0.008);
%% Coat Shopping Dataset - RAPMF
load('coat_user.mat')
load('coat_random.mat');
out2 = RAPMF(train,test,'m',290,'n',300,'sigU',0.04,'sigV',0.04,'sigR',0.02,'maxIter',3000);