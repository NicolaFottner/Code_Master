
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


%function [weights, tr_accuracy, te_accuracy,tr_loss,te_loss] = mlp(pattners,targets)

% NO DROPOUT for now

geo_shape_class = 6;
g_batchsize = 12;
import_shapes;
g_batchdata = reshape(permute(g_batchdata,[1,3,2]),[size(g_batchdata,1)*size(g_batchdata,3),size(g_batchdata,2)]);
g_batchtargets = reshape(permute(g_batchtargets,[1,3,2]),[size(g_batchtargets,1)*size(g_batchtargets,3),size(g_batchtargets,2)]);
load t_model DN
vishid_1 = DN.L{1,1}.vishid;
hidbiases_1 = DN.L{1,1}.hidbiases;
clear DN
load rbm2_16J11h39.mat hidbiases_2 vishid_2
g_pass = 1./(1 + exp(-g_batchdata*vishid_1 - repmat(hidbiases_1,size(g_batchtargets,1),1)));
hid_out_2 = 1./(1 + exp(-g_pass*vishid_2 - repmat(hidbiases_2,size(g_batchtargets,1),1)));

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
net = train(net,hid_out_2',g_batchtargets');



%net = train(net,cl_data',cl_target');
% eval net --- train and test data the same @todo: check if ok


%%% the following I meant to do:
% check how to access performance data ..
% .. from the performance object? __ "validation_loss"



%perf_values_b(b) = perform(net,train_d_2,test_d_2);

%end
