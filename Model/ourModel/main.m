%% Yahoo! R3 Dataset - MCO
load('yahoo_user.mat')
load('yahoo_random.mat');
out1 = ourModel(train,test,'maxIter',3000);
%% Coat Shopping Dataset - MCO
load('coat_user.mat')
load('coat_random.mat');
out2 = ourModel(train,test,'m',290,'n',300,'maxIter',400);
%% Yahoo! R3 Dataset - MCP
load('yahoo_user.mat')
load('yahoo_random.mat');
out3 = ourModel(train,test,'Method','MCP','lr',1e-8,'lrT',2*1e-9);
%% Coat Shopping Dataset - MCP
load('coat_user.mat')
load('coat_random.mat');
out4 = ourModel(train,test,'m',290,'n',300,'Method','MCP','lr',1e-8,'lrT',2*1e-9,'maxIter',420);
%% Yahoo! R3 Dataset - MCM
load('yahoo_user.mat')
load('yahoo_random.mat');
out5 = ourModel(train,test,'Method','MCM');
%% Coat Shopping Dataset - MCM
load('coat_user.mat')
load('coat_random.mat');
out6 = ourModel(train,test,'m',290,'n',300,'Method','MCM','maxIter',370);
%% Yahoo! R3 Dataset - MCC
load('yahoo_user.mat')
load('yahoo_random.mat');
out7 = ourModel(train,test,'Method','MCC','lr',5*1e-9,'lrT',1e-9);
%% Coat Shopping Dataset - MCC
load('coat_user.mat')
load('coat_random.mat');
out8 = ourModel(train,test,'m',290,'n',300,'Method','MCC','lr',5*1e-9,'lrT',1e-9,'maxIter',360);