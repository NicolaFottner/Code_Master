
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


clear net;

% diverand function might be of interest

g_pass_class =g_pass;
cl_data = 1./(1 + exp(-g_pass_class*vishid_2 - repmat(hidbiases_2,size(g_test_data,1),1)));
cl_target = g_test_target;
net = patternnet(256,'trainscg','crossentropy');
net.layers{1}.transferFcn = 'softmax';
%net.trainParam.showWindow = false;
net.trainParam.showCommandLine =true;

net.divideFcn = 'dividerand'; % @todo : remember: this divides data randomly
net.divideParam.trainRatio = 0.7;
net.divideParam.valRatio = 0.15;
net.divideParam.testRatio = 0.15;


% train net
net = train(net,cl_data',cl_target');
% eval net --- train and test data the same @todo: check if ok


%%% the following I meant to do:
% check how to access performance data ..
% .. from the performance object? __ "validation_loss"



%perf_values_b(b) = perform(net,train_d_2,test_d_2);


