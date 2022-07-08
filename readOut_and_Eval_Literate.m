% Code does the following
% 1. Compute supervised read-out with perceptron
%       a. Prepare geo-shape Data for perceptron
%       b. Traing and evaluation perceptron
% 2. Compute performance measures 
%       a. all the data from training and testing
%       b. 
%       b. Plotting of Receptive fields of rbm 1 and rbm2
%       b. call methods to make more measures:
%          overfitting, reco-error, histograms 

dd = strsplit(date,'-'); clean_date = strcat(dd(1),dd(2));c=clock; %store date without "-YYYY"

%% Create train and test set for perceptron:
addpath("data/new04Jl/")
load 50_50_trainData.mat
g_pass = 1./(1 + exp(-data*vishid_1 - repmat(hidbiases_1,size(data,1),1)));
hid_out_2 = 1./(1 + exp(-g_pass*vishid_2 - repmat(hidbiases_2,size(data,1),1)));
%%%
least_square = true;
%%%
p = 0.2;
if least_square == true
    %% if classifier = multivariate least square regression
    [W1, tr_acc1, te_acc1,tr_loss1,te_loss1] = perceptron(a1,p,g_pass,targets);
    [W2, tr_acc2, te_acc2,tr_loss2,te_loss2] = perceptron(a2,p,hid_out_2,target);
else
    %% íf MLP
    [W1, tr_acc1, te_acc1,tr_loss1,te_loss1] = mlp(g_pass,p,targets);
    [W2, tr_acc2, te_acc2,tr_loss2,te_loss2] = mlp(hid_out_2,p,targets);

end

%%  Plot and Save performance measurements:
fprintf(1,'\n Linear Classifier of "rbm1 output" = \n');
fprintf(1,'\n Train accuracy =  %d\n',tr_acc1);
fprintf(1,'\n Test accuracy =  %d\n',te_acc1);
fprintf(1,'\n Train Loss =  %d\n',tr_loss1);
fprintf(1,'\n Test Loss =  %d\n\n',te_loss1);
fprintf(1,'\n Linear Classifier of "rbm2 output" =\n');
fprintf(1,'\n Train accuracy =  %d\n',tr_acc2);
fprintf(1,'\n Test accuracy =  %d\n',te_acc2);
fprintf(1,'\n Train Loss =  %d\n',tr_loss2);
fprintf(1,'\n Test Loss =  %d\n',te_loss2);

if numhid3 == 0
    X = ["Final_layer";"From_RBM1";"Epochs"];
    tr_acc = [tr_acc2;tr_acc1;NaN];
    te_acc = [te_acc2;te_acc1;NaN];
    tr_loss=[tr_loss2;tr_loss1;NaN];
    te_loss = [te_loss2;te_loss1;NaN];
    Epoch = [NaN;NaN;final_epoch];
    Classifier = table(X,tr_acc,te_acc,tr_loss,te_loss,Epoch);
else   
    fprintf(1,'\n Linear Classifier of "rbm3 output" =\n');
    fprintf(1,'\n Train accuracy =  %d\n',tr_acc3);
    fprintf(1,'\n Test accuracy =  %d\n',te_acc3);
    fprintf(1,'\n Train Loss =  %d\n',tr_loss3);
    fprintf(1,'\n Test Loss =  %d\n',te_loss3);
    X = ["Final_layer";"From_RBM2";"From_RBM1";"Epochs"];
    tr_acc = [tr_acc3;tr_acc2;tr_acc1;NaN];
    te_acc = [te_acc3;te_acc2;te_acc1;NaN];
    tr_loss=[tr_loss3;tr_loss2;tr_loss1;NaN];
    te_loss = [te_loss3;te_loss2;te_loss1;NaN];
    Epoch = [NaN;NaN;NaN;final_epoch];          
    Classifier = table(X,tr_acc,te_acc,tr_loss,te_loss,Epoch);
end
class_specific_output; %compute details of the output and saves them

%% Perform Assesment: Classifaction as Shape Id.

Letter_Assesment;
% to get all the matrix data for the stat analysis
pred_ce_effect;
pred_ce_effect_ALL;
properties.dropout = dropout;
properties.dropout_p1 = p_layer1;
properties.dropout_cl = a1;
properties.minibatchsize = g_batchsize;
properties.epoch2 = final_epoch;

properties.numhid2 = numhid2;
properties.numhid3 = numhid3;

histo;
if numhid3 == 0
    reco_error = full_rec_err_g;
    Overfitting = overfitting_g_2;
else
    properties.epoch3 = final_epoch_3;
    properties.dropout_p2 = p_layer2;
    Overfitting.layer2 = overfitting_g_2;
    Overfitting.layer3 = overfitting_g_3; 
    reco_error.layer2 = full_rec_err_g;
    reco_error.layer3 = full_rec_err_3;
end
% save data into single file:
% /clock string to better scan the eval files
hour_str = int2str(c(4));
min_str = int2str(c(5));
if length(hour_str) == 1
    hour_str = ['0' hour_str(1)];
end
if length(min_str) == 1
    min_str = ['0' min_str(1)];
end
filename = "Evals/" + clean_date + "_" + hour_str + "h" + min_str+"m_" + "H2"+ int2str(numhid2)+ "_H3"+ int2str(numhid3);
save(filename,'properties','Classifier','Classifier_Details','Id_BasedOnGeoS','histograms', ...
    'CE_eval','Overfitting','reco_error');


%% Plot receptive fields
% create DN struct for facilitating later "plotting the receptive fields"
% for most fields, any number will do
if numhid3 == 0
    DN.layersize   = [1000 numhid2];           % network architecture
    DN.nlayers     = length(DN.layersize);
    DN.maxepochs   = 60;                    % unsupervised learning epochs
    DN.batchsize   = 160;                   % mini-batch size
    %set parameters of rbm 1 layer:
    DN.L{1}.hidbiases  = hidbiases_1;
    DN.L{1}.vishid     = vishid_1;
    DN.L{1}.visbiases  = visbiases_1;
    %set parameters of rbm 2 layer:
    DN.L{2}.hidbiases  = hidbiases_2;
    DN.L{2}.vishid     = vishid_2;
    DN.L{2}.visbiases  = visbiases_2;
else
    DN.layersize   = [1000 numhid2 numhid3];           % network architecture
    DN.nlayers     = length(DN.layersize);
    DN.maxepochs   = 60;                    % unsupervised learning epochs
    DN.batchsize   = 160;                   % mini-batch size
    %set parameters of rbm 1 layer:
    DN.L{1}.hidbiases  = hidbiases_1;
    DN.L{1}.vishid     = vishid_1;
    DN.L{1}.visbiases  = visbiases_1;
    %set parameters of rbm 2 layer:
    DN.L{2}.hidbiases  = hidbiases_2;
    DN.L{2}.vishid     = vishid_2;
    DN.L{2}.visbiases  = visbiases_2;
    %set parameters of rbm 3 layer:
    DN.L{3}.hidbiases  = hidbiases_3;
    DN.L{3}.vishid     = vishid_3;
    DN.L{3}.visbiases  = visbiases_3;
end

%plot_L1(DN,1000);

 
% if ii == 1 && numhid3 == 0
%     plot_L2(DN,numhid2,final_epoch);
% elseif ii == 1 && numhid3 ~= 0
%     plot_L2(DN,numhid2,final_epoch);
%     plot_L3(DN,numhid3,final_epoch_3);
% end
