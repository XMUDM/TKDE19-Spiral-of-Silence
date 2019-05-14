%% Yahoo! R3 Dataset - UserKNN
load('yahoo_user.mat')
load('yahoo_random.mat');
out1 = KNNRec(train,test);
%% Coat Shopping Dataset - UserKNN
load('coat_user.mat')
load('coat_random.mat');
out2 = KNNRec(train,test,'m',290,'n',300);
%% Yahoo! R3 Dataset - ItemKNN
load('yahoo_user.mat')
load('yahoo_random.mat');
out3 = KNNRec(train,test,'method','item','similarity','cosine');
%% Coat Shopping Dataset - ItemKNN
load('coat_user.mat')
load('coat_random.mat');
out4 = KNNRec(train,test,'m',290,'n',300,'method','item','similarity','cosine');