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

% This program trains Restricted Boltzmann Machine in which
% visible, binary, stochastic pixels are connected to
% hidden, binary, stochastic feature detectors using symmetrically
% weighted connections. Learning is done with 1-step Contrastive Divergence.   
% The program assumes that the following variables are set externally:

% maxepoch  -- maximum number of epochs
% numhid    -- number of hidden units 
% n_batchdata -- the data that is divided into batches (numcases numdims numbatches)
% restart   -- set to 1 if learning starts from beginning 

% parameters as set by testolin 2017
epsilonw      = 0.04;   % Learning rate for weights 
epsilonvb     = 0.04;   % Learning rate for biases of visible units 
epsilonhb     = 0.04;   % Learning rate for biases of hidden units 
weightcost  = 0.0001;   
initialmomentum  = 0.5;
finalmomentum    = 0.8;
earlystopping = rbm1.earlyStopping;
patience =  rbm1.patience;

% default
% epsilonw      = 0.1;   % Learning rate for weights 
% epsilonvb     = 0.1;   % Learning rate for biases of visible units 
% epsilonhb     = 0.1;   % Learning rate for biases of hidden units 
% weightcost  = 0.0002;   
% initialmomentum  = 0.5;
% finalmomentum    = 0.9;

[numcases numdims numbatches]=size(n_batchdata);
% numcases: size of current mini-batch / number of training samples
% numdims: size of input
% numbatches: number of batches
if restart ==1,
  restart=0;
  epoch=1;
% Initializing symmetric weights and biases. 
  vishid     = 0.01*randn(numdims, numhid); %weights
  hidbiases  = zeros(1,numhid);
  visbiases  = zeros(1,numdims);
% variables used in positive phase
  poshidprobs = zeros(numcases,numhid); 
  posprods    = zeros(numdims,numhid);
% neg for variables in negative phase
  neghidprobs = zeros(numcases,numhid); 
  negprods    = zeros(numdims,numhid);
% variables used for updating weights in order to facilitate "Momentum"
  vishidinc  = zeros(numdims,numhid);
  hidbiasinc = zeros(1,numhid);
  visbiasinc = zeros(1,numdims);
 
  batchposhidprobs=zeros(numcases,numhid,numbatches);
end
% rec_err = "full" reconstruction error over epochs and numbatches
% CD-n reconstruction starts at 0 at each mini-batch
full_rec_err=zeros(maxepoch, numbatches);
% overfitting: avgEnergy(training) && avgEnergy(validation) -- REMEMBER: !diff
overfitting = zeros(maxepoch,2);
% for monitoring purpose
deltas = zeros(maxepoch,1);
wait = 0;

for epoch = epoch:maxepoch %maxepoch
    fprintf(1,'epoch %d\r',epoch); 
    errsum=0;
    errvec=[];

    for batch = 1:numbatches
        fprintf(1,'epoch %d batch %d\r',epoch,batch); 
%%%%%%% START POSITIVE PHASE %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        data = n_batchdata(:,:,batch);
        % prob that hidden units fire under given sample.
        % poshidprobs of dim: numcases x numhid
        poshidprobs = 1./(1 + exp(-data*vishid - repmat(hidbiases,numcases,1)));   
        batchposhidprobs(:,:,batch)=poshidprobs;
        % just need for calculating "deltas", see slide/larochelle
        posprods    = data' * poshidprobs; 
        % - sum becasue weight update averaged over each mini-batch
        poshidact   = sum(poshidprobs); %dim 1xnumhid -- sum of each collumn of poshidprobs
        % for update visible layer bias
        posvisact = sum(data);
        
%%%%%%% END OF POSITIVE PHASE  %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        % see paper: final hidden layer unit should be binary
        poshidstates = poshidprobs > rand(numcases,numhid);
        
%%%%%%% START NEGATIVE PHASE  %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        % "gibbs sampling", here CD-1
        negdata = 1./(1 + exp(-poshidstates*vishid' - repmat(visbiases,numcases,1)));
        neghidprobs = 1./(1 + exp(-negdata*vishid - repmat(hidbiases,numcases,1)));  
        negprods  = negdata'*neghidprobs;
        neghidact = sum(neghidprobs);
        negvisact = sum(negdata); 
        
%%%%%%% END OF NEGATIVE PHASE %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        err= sum(sum( (data-negdata).^2 )); %square error between data and reconstruction
        %first sum over column/numdim, second over row/numcase
        errsum = err + errsum;
        errvec =[errvec;err];
        
        if epoch>5,
         momentum=finalmomentum;
        else
         momentum=initialmomentum;
        end;
        
%%%%%%% UPDATE WEIGHTS AND BIASES %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% 
        vishidinc = momentum*vishidinc + ...
                    epsilonw*( (posprods-negprods)/numcases - weightcost*vishid);
        visbiasinc = momentum*visbiasinc + (epsilonvb/numcases)*(posvisact-negvisact);
        hidbiasinc = momentum*hidbiasinc + (epsilonhb/numcases)*(poshidact-neghidact);
        
        vishid = vishid + vishidinc;
        visbiases = visbiases + visbiasinc;
        hidbiases = hidbiases + hidbiasinc;
        
%%% END OF UPDATES %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% 
    end
    fprintf(1, 'epoch %4i error %6.1f  \n', epoch, errsum); 
    full_rec_err(epoch,:) = errvec;

    % randomize validation set:
    rand_values = randi(10000,[2500 1]); % values here are n_dataset specific
    n_val = double(d_val(rand_values,:));
    % compute overfitting measures:
    overfitting(epoch,1) = mean(-(n_train*visbiases' + sum(log( 1+exp(n_train*vishid + repmat(hidbiases,size(n_train,1),1))),2)));
    overfitting(epoch,2) = mean(-(n_val*visbiases' + sum(log( 1+exp(n_val*vishid + repmat(hidbiases,size(n_val,1),1))),2)));
    
    % early stopping condition
    if earlystopping == true
        if epoch > 1
            ov_diff = abs(overfitting(epoch,1) - overfitting(epoch,2));
            if ov_diff > deltas(epoch-1)
                if wait == 0
                    vishid_cpS = vishid_cp;hidbiases_cpS = hidbiases_cp;visbiases_cpS = visbiases_cp;batchposhidprobs_2_cpS = batchposhidprobs_2_cp;
                end
                wait = wait +1;
                if wait == patience+1
                    % break out the epoch loop:
                    vishid=vishid_cpS;hidbiases=hidbiases_cpS ;visbiases=visbiases_cpS ;batchposhidprobs=batchposhidprobs_2_cpS;
                    overfitting = overfitting(1:epoch-(patience+1),:);
                    full_rec_err = full_rec_err(1:epoch-(patience+1),:);
                    final_epoch_1 = epoch-(patience);
                    break;
                end
            else
                wait =0;
            end
        end
    else
        final_epoch_1 = maxepoch;
    end
    deltas(epoch,:)=abs(overfitting(epoch,1) - overfitting(epoch,2));
    vishid_cp = vishid;hidbiases_cp = hidbiases;visbiases_cp = visbiases;batchposhidprobs_2_cp = batchposhidprobs_2;
end
clear data;
%fplot(size(errvec),errvec);
save err_rbm_1 full_rec_err overfitting deltas final_epoch_1;


