% Version 1.000 
%
% Code provided by Geoff Hinton and Ruslan Salakhutdinov 
%
% Permission is granted for anyone to copy, use, modify, or distribute this
% program and accompanying programs and documents for any purpose, provided
% this copyright notice is retained and prominently displayed, along with
% a note saying that the original programs are available from our
% web page.
% The programs and documents are distributed without any warranty, express or
% implied.  As the programs were written for research purposes only, they have
% not been tested to the degree that would be advisable in any important
% application.  All use of these programs is entirely at the user's own risk.

% This program trains Restricted Boltzmann Machines in which
% visible, binary, stochastic pixels are connected to
% hidden, binary, stochastic feature detectors using symmetrically
% weighted connections. Learning is done with 1-step Contrastive Divergence.   
% The program assumes that the following variables are set externally:

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% maxepoch  -- maximum number of epochs
% numhid    -- number of hidden units of rbm1
% numhid3    -- number of hidden units of rbm2
% batchdata -- the data that is divided into batches (numcases numdims numbatches)
% restart   -- set to 1 if learning starts from beginning 
clear deltas;
epsilonw      = rbm2.epsilonw;   % Learning rate for weights 
epsilonvb     = rbm2.epsilonvb;   % Learning rate for biases of visible units 
epsilonhb     = rbm2.epsilonhb;   % Learning rate for biases of hidden units 
weightcost  = rbm2.weightcost;   
initialmomentum  = rbm2.initialmomentum;
finalmomentum    = rbm2.finalmomentum;
earlystopping = rbm2.earlyStopping;
patience =  rbm2.patience;
maxepoch = rbm2.maxepoch;
[g_numcases g_numdims g_numbatches]=size(g_batchdata);

if restart ==1
  restart=0;
  epoch=1;

  final_epoch_3 = 1;
%% For overfitting computation
% For computing overfitting / used in early stopping
% "randomize", the validation set used in the computation 
g_val_data_1 = g_val_data(1: size(g_val_data,1)/3,:);
g_val_data_2 = g_val_data(1 + size(g_val_data,1)/3 :size(g_val_data,1)/3 * 2 ,:);
g_val_data_3 = g_val_data(1 + size(g_val_data,1)/3 * 2: size(g_val_data,1),:);
% create g_pass for subset of train data:
r = randperm(g_numbatches,floor(size(g_val_data_1,1)/g_numcases)); %with random interval, size(subtrain)=size(val)
% upper line: /g_numcases because we pick batches here of size g_numcases
g_train = g_batchdata(:,:,r);
g_train_flat = [];
for i=1:size(r,2) % I prefer this reshape to ensure to keep the dimensionality right (for now)
    g_train_flat = [g_train_flat;g_train(:,:,i)];
end
% Compute Pass-through
pass_subtrain = 1./(1 + exp(-g_train_flat*vishid_1 - repmat(hidbiases_1,size(g_val_data_1,1),1)));  
g_pass_subtrain = 1./(1 + exp(-pass_subtrain*vishid_2 - repmat(hidbiases_2,size(g_val_data_1,1),1)));  
%g_pass_val is changed in each "overfitting-computation" // "randomized"
v_perm = randi(3);
if v_perm==1
    pass_val = 1./(1 + exp(-g_val_data_1*vishid_1 - repmat(hidbiases_1,size(g_val_data_1,1),1)));
    g_pass_val = 1./(1 + exp(-pass_val*vishid_2 - repmat(hidbiases_2,size(pass_val,1),1)));
elseif v_perm == 2
        pass_val = 1./(1 + exp(-g_val_data_2*vishid_1 - repmat(hidbiases_1,size(g_val_data_2,1),1)));  
        g_pass_val = 1./(1 + exp(-pass_val*vishid_2 - repmat(hidbiases_2,size(pass_val,1),1)));
elseif v_perm == 3
        pass_val = 1./(1 + exp(-g_val_data_3*vishid_1 - repmat(hidbiases_1,size(g_val_data_3,1),1)));  
        g_pass_val = 1./(1 + exp(-pass_val*vishid_2 - repmat(hidbiases_2,size(pass_val,1),1)));
end


%%%%%%%%%%%%%%%% TRAIN RBM3 %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Initializing parameters: 
  g_numdims = numhid2;
  vishid     = 0.01*randn(numhid2, numhid3);
  hidbiases  = zeros(1,numhid3);
  visbiases  = zeros(1,numhid2);
  poshidprobs = zeros(g_numcases,numhid3); 
  posprods    = zeros(numhid2,numhid3);
  neghidprobs = zeros(g_numcases,numhid3); 
  negprods    = zeros(numhid2,numhid3);
  vishidinc  = zeros(numhid2,numhid3);
  hidbiasinc = zeros(1,numhid3);
  visbiasinc = zeros(1,numhid2);
  batchposhidprobs_2=zeros(g_numcases,numhid3,g_numbatches);
end
% to save/assess reconstruction error
full_rec_err_3=zeros(maxepoch, g_numbatches);
% this is to analyse overfitting
overfitting_g_3 = zeros(maxepoch,2);
% for monitoring purpose
deltas = zeros(maxepoch,1);
wait = 0;

%% Start Training
for epoch = epoch:maxepoch
    fprintf(1,'epoch %d\r',epoch); 
    errsum2=0;
    errvec2=[];
    
    for batch = 1:g_numbatches
        fprintf(1,'epoch %d batch %d\r',epoch,batch); 
        
%%%%%%%%% START POSITIVE PHASE %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        data = hid_out_2(:,:,batch);
        % apply dropout
        r1 = (rand(g_numcases,numhid2)<=p_layer2);
        data = data .* r1; 
        poshidprobs = 1./(1 + exp(-data*vishid - repmat(hidbiases,g_numcases,1)));   
        batchposhidprobs_2(:,:,batch)=poshidprobs;
        posprods    = data' * poshidprobs; 
        poshidact   = sum(poshidprobs);
        posvisact = sum(data);
        
%%%%%%%%% END OF POSITIVE PHASE  %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        poshidstates = poshidprobs > rand(g_numcases,numhid3);
        
%%%%%%%%% START NEGATIVE PHASE  %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        negdata = 1./(1 + exp(-poshidstates*vishid' - repmat(visbiases,g_numcases,1)));
        negdata = negdata .* r1; % dropout
        neghidprobs = 1./(1 + exp(-negdata*vishid - repmat(hidbiases,g_numcases,1)));  
        negprods  = negdata'*neghidprobs;
        neghidact = sum(neghidprobs);
        negvisact = sum(negdata); 
        
%%%%%%%%% END OF NEGATIVE PHASE %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        err= sum(sum( (data-negdata).^2 ));
        errsum2 = err + errsum2;
        errvec2 =[errvec2;err];
        
        if epoch>5,
         momentum=finalmomentum;
        else
         momentum=initialmomentum;
        end;
        
%%%%%%%%% UPDATE WEIGHTS AND BIASES %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% 
        vishidinc = momentum*vishidinc + ...
                    epsilonw*( (posprods-negprods)/g_numcases - weightcost*vishid);
        visbiasinc = momentum*visbiasinc + (epsilonvb/g_numcases)*(posvisact-negvisact);
        hidbiasinc = momentum*hidbiasinc + (epsilonhb/g_numcases)*(poshidact-neghidact);
        
        vishid = vishid + vishidinc;
        visbiases = visbiases + visbiasinc;
        hidbiases = hidbiases + hidbiasinc;
%%%%%%%%%%%%%%%% END OF UPDATES %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% 
    end 

fprintf(1, 'epoch %4i error %6.1f  \n', epoch, errsum2); 
full_rec_err_3(epoch,:) = errvec2;

%% Early Stopping computation
% compute overfitting measures:
overfitting_g_3(epoch,1) = mean(-(g_pass_subtrain*visbiases' + sum(log( 1+exp(g_pass_subtrain*vishid + repmat(hidbiases,size(g_pass_subtrain,1),1))),2)));
overfitting_g_3(epoch,2) = mean(-(g_pass_val*visbiases' + sum(log( 1+exp(g_pass_val*vishid + repmat(hidbiases,size(g_pass_val,1),1))),2)));

% early stopping condition
if earlystopping == true
    if epoch > 1
        ov_diff = abs(overfitting_g_3(epoch,1) - overfitting_g_3(epoch,2));
        if ov_diff > deltas(epoch-1)
            if wait == 0
                vishid_cpS = vishid_cp;hidbiases_cpS = hidbiases_cp;visbiases_cpS = visbiases_cp;batchposhidprobs_2_cpS = batchposhidprobs_2_cp;
            end
            wait = wait +1;
            if wait == patience
                % break out the epoch loop:
                vishid=vishid_cpS;hidbiases=hidbiases_cpS ;visbiases=visbiases_cpS ;batchposhidprobs_2=batchposhidprobs_2_cpS;
                overfitting_g_3 = overfitting_g_3(1:epoch-(patience),:);
                full_rec_err_3 = full_rec_err_3(1:epoch-(patience),:);
                final_epoch_3 = epoch-(patience);
                break;
            end
        else
            wait =0;
        end
    end
else
    final_epoch_3 = maxepoch;
end
deltas(epoch,:)=abs(overfitting_g_3(epoch,1) - overfitting_g_3(epoch,2));
vishid_cp = vishid;hidbiases_cp = hidbiases;visbiases_cp = visbiases;batchposhidprobs_2_cp = batchposhidprobs_2;
end
fprintf(1,'number of runned epoch = %d \r',final_epoch_3); 

% applying Dropout "weight scaling inference rule":
vishid = vishid * p_layer2;
save err_rbm_3 full_rec_err_3 overfitting_g_3 deltas final_epoch_3;

