% Shallow Neural Network
% returns training accuracy, training loss, trainPerfo (loss on training
% data each epoch), valPerf(loss on val data each epoch)

function [net, tr_accuracy, tr_loss, trainPerf, valPerf] = MLP(units,trainData,train_t)
% % %%%%%
% % addpath("data/new04Jl/")
% % addpath("testolin/")
% % load 50_50_TrainTestData.mat trainData train_t
% % load t_model DN
% % vishid_1 = DN.L{1,1}.vishid;
% % hidbiases_1 =DN.L{1,1}.hidbiases;
% % clear DN
% % load g_rbm_2.mat vishid_2 hidbiases_2
% % 
% % g_pass = 1./(1 + exp(-trainData*vishid_1 - repmat(hidbiases_1,size(trainData,1),1)));
% % hid_out_2 = 1./(1 + exp(-g_pass*vishid_2 - repmat(hidbiases_2,size(trainData,1),1)));
% % units = 256;
% % trainData = hid_out_2;
% % %%%%%

net = patternnet(units,'trainscg','crossentropy'); % GradDesc: 'traingd'
net.divideFcn = 'dividerand'; % partition indices randomly
net.divideParam.trainRatio = 0.9;
net.divideParam.valRatio = 0.1;
net.divideParam.testRatio = 0.0;
net.layers{1}.transferFcn = 'poslin'; %% 'help nntransfer' -- to list alternatives
net.trainParam.epoch = 1000;

% default values of stopping criteria:
% net.trainParam.min_grad = 1e-5 // Minimum Gradient Magnitude        
% net.trainParam.max_fail = 6 // patience

net.trainParam.showCommandLine =true;
net.trainParam.showWindow = 0;
% train the net
[net,t] = train(net,trainData',train_t','useParallel','yes','showResources','yes');

% get performance
trainPerf = t.perf; % perf over epochs
tr_loss = trainPerf(size(trainPerf,2));
valPerf = t.vperf;
y = net(trainData(t.trainInd,:)'); % Prediction of training data
tind = vec2ind(train_t(t.trainInd,:)');
yind = vec2ind(y);
tr_accuracy = 1 - sum(tind ~= yind)/numel(tind);
end
