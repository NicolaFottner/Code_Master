
%%%% CAUTION: this is not the perceptron but ..
% a one hidden layer MLP /classical NN
% with 256 hidden units

% perceptron is a "binary classifier"

% cross entropy == D_KL minimization
% I need softmax for valid distribution? as activation function
% "the code runs"

% keep in mind the train function -
% my problem: single label and mutiple classes
% softmax + crossentropy
% tensorflow uses adam as a algorithm instead of 'scg'


function [net, tr_accuracy, te_accuracy,tr_loss,te_loss] = MLP(data,p,targets)

% index = floor(size(targets,1)*p); % p = percentage of test data
% tr_patterns = data(index+1 : size(data,1) , :);
te_patterns = data(1:index, :);
tr_labels = targets(index+1 : size(data,1) , :);
te_labels =  targets(1:index,:);

addpath("data/new04Jl/")
load 50_50_TrainTestData.mat
g_pass = 1./(1 + exp(-data*vishid_1 - repmat(hidbiases_1,size(data,1),1)));
hid_out_2 = 1./(1 + exp(-g_pass*vishid_2 - repmat(hidbiases_2,size(data,1),1)));

p = 0.2;
index = floor(size(targets,1)*p); % p = percentage of test data
tr_patterns = data(index+1 : size(data,1) , :);


units = 256;
net = patternnet(units,'trainscg','crossentropy'); % GradDesc: 'traingd'
net.divideFcn = 'dividerand'; % partition indices randomly
net.divideParam.trainRatio = 0.7;
net.divideParam.valRatio = 0.15;
net.divideParam.testRatio = 0.15;
net.layers{1}.transferFcn = 'poslin'; %% 'help nntransfer' -- to list alternatives
net.trainParam.epoch = 50;

% default values of stopping criteria:
% net.trainParam.min_grad = 1e-5 // Minimum Gradient Magnitude        
% net.trainParam.max_fail = 6 // patience

view(net)
net.trainParam.showCommandLine =true;

% train the net
net = train(net,hid_out_2',targets');


% get training error

% get training loss 
net.performFcn = 'crossentropy';
y = net(hid_out_2');
perf = crossentropy(net,hid_out_2',targets',{1}); %,'regularization',0.1)





%% predict on test set

[Y,scores] = predict(net,hid_out_2);





%net = train(net,cl_data',cl_target');
% eval net --- train and test data the same @todo: check if ok


%%% the following I meant to do:
% check how to access performance data ..
% .. from the performance object? __ "validation_loss"



%perf_values_b(b) = perform(net,train_d_2,test_d_2);

%end
