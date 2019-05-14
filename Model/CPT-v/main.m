%% Yahoo! R3 Dataset - CPT-v
load('yahoo_user.mat')
load('yahoo_random.mat');
out1 = CPTv(train,test,[0.014,0.011,0.027,0.063,0.225]);
%% Coat Shopping Dataset - CPT-v
load('coat_user.mat')
load('coat_random.mat');
out2 = CPTv(train,test,[0.0545 0.0866 0.0890 0.1033 0.2330],'m',290,'n',300,'C',2,'S',40000);