% Version 1.00
% Code provided by Ruslan Salakhutdinov and Geoff Hinton  
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

% This program creates the so-called "Illiterate Network", 
% with 2 stacked-RBMS and an linear classifier in the top/output layer

%%%% INFO outputed files: // for each single "model simulation/run"
% ./g_rbm_2.mat & ./g_rbm_3.mat
% ./err_rbm_2.mat & ./err_rbm_3.mat
% ./data_plotting/histograms.pdf:         histogram graphicfile
% ./data_plotting/overfitting.pdf:        overfitting graphicfile
% ./data_plotting/reco_plotting.pdf:      recognition-error graphicfile
% /plots_results/fig/..:                  "graphicfile of receptive fields"
% /plots_results/classf_perf/..Train:           "data and performance on 3Lay training"
% /plots_results/classf_perf/..detail_distr:    "graphicfile Pr.Distr. of each geo_s"
% /plots_results/classf_perf/.. detail_acc:     "detailed data and performance of 3L"
% /plots_results/Id_basedOnGeoS(6)/..L_PrD:     "graphicfile Pr.Distr"
% /plots_results/Id_basedOnGeoS(6)/..pL_PrD:    "graphicfile Pr.Distr"
% /plots_results/Id_basedOnGeoS(6)/..Eval_L&pL: "Letter classif./assesment"
% ... for the simulation with a third rbm layer, see folder "/three

configurations;

configurations_list = [];
elem.layer2=300;
elem.layer3= 0;
elem.dropout= 0;
elem.minibatchsize= 12;
configurations_list = [configurations_list;elem];

elem.layer2=350;
elem.layer3= 0;
elem.dropout= 0;
elem.minibatchsize= 24;
configurations_list = [configurations_list;elem];

% and do all again with dropout 0.4

% from 12J: missing z=34 onwards

for z=1:size(configurations_list,1)
    for ii=1:5    %% --- 10
        % initialize hyperparameters
        maxepoch=500; % 
        geo_shape_class = 6; %problem of 6 class/shapes
        numhid2 = configurations_list(z).layer2;
        numhid3 = configurations_list(z).layer3;
        dropout = configurations_list(z).dropout;
        g_batchsize = configurations_list(z).minibatchsize;
        numhid = 1000;
        if dropout
            p_layer1 = 0.5;
            a1=1;a2=a1; %see fast dropout 2013 paper: a = (1-p) / p
            p_layer2 = NaN;
            if numhid3 ~= 0
                p_layer2 = 0.5;a3=a1;
            end 
        else
            p_layer1 = NaN;
            p_layer2 = NaN;
            a1 = NaN;
        end

        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

        fprintf(1,'\n\nXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX\n\n');
        fprintf(1,'Number of Iterations: %d\n',ii);
        fprintf(1,'\n\nXXXXXXXX\n\n');
        addpath("testolin/")
        addpath("data_plotting/")
        addpath("data")

        %% Import Data from testolin
        fprintf(1,'Importing first layer from Testolin:\n');
        load t_model DN
        numdims = 1600;
        numbatches = 80000/160;
        numcases = DN.batchsize;
              
        fprintf(1,'Importing data: Geometric Shapes \n');
        import_shapes;
        fprintf(1,'Start Training. \n'); 
        fprintf(1,'Number Epochs: %3i \n', maxepoch);
        
        % data for training rbm2:
        [g_numcases g_numdims g_numbatches]=size(g_batchdata);
        % create file to access info from outside
        save info maxepoch numdims numhid numhid2 numbatches numcases g_numbatches g_numcases
        %%% @todo use the above line in the "plotting methods" for modularity

        %% RBM 1 / fetching it from Testolin
        %%%% first rbm layer
        vishid_1 = double(DN.L{1,1}.vishid);
        hidbiases_1 = double(DN.L{1,1}.hidbiases);
        visbiases_1 = double(DN.L{1,1}.visbiases);
        clear DN
        
        %% RBM 2
        %%%%% second rbm layer
        fprintf(1,'\nTraining Layer 2 with RBM: %d-%d \n',numhid,numhid2);
        rbm2.maxepoch = maxepoch;
        rbm2.epsilonw      = 0.01;   % Learning rate for weights 
        rbm2.epsilonvb     = 0.01;   % Learning rate for biases of visible units 
        rbm2.epsilonhb     = 0.01;   % Learning rate for biases of hidden units 
        rbm2.weightcost  = 0.000004;   
        rbm2.initialmomentum  = 0.5;
        rbm2.patience = 3;
        rbm2.finalmomentum    = 0.9;
        rbm2.earlyStopping = true;
        restart=1;
        if dropout
            % @todo: DO I NEED THIS? can I just use p_layer2 = 1?
            rbm_2DropO;
        else
            rbm_2;
        end
        vishid_2=vishid; hidbiases_2=hidbiases; visbiases_2=visbiases; hid_out_2 = batchposhidprobs_2;
        save g_rbm_2 vishid_2 hidbiases_2 visbiases_2; % hid_out_2;
        %load g_rbm_2 vishid_2 hidbiases_2 visbiases_2 hid_out_2 g_pass;

        %% RBM 3 
        if numhid3 ~= 0
            fprintf(1,'\nPretraining Layer 3 with RBM: %d-%d-%d \n',numhid,numhid2,numhid3);                
            rbm3.maxepoch = maxepoch;
            rbm3.epsilonw      = 0.01;   % Learning rate for weights 
            rbm3.epsilonvb     = 0.01;   % Learning rate for biases of visible units 
            rbm3.epsilonhb     = 0.01;   % Learning rate for biases of hidden units 
            rbm3.weightcost  = 0.000004;   
            rbm3.initialmomentum  = 0.5;
            rbm3.patience = 3;
            rbm3.finalmomentum    = 0.9;
            rbm3.earlyStopping = true;
            restart=1;
            if dropout
                rbm_3DropO;
            else
                rbm_3;
            end
            vishid_3=vishid; hidbiases_3=hidbiases; visbiases_3=visbiases; hid_out_3 = batchposhidprobs_2;
            save g_rbm_3 vishid_3 hidbiases_3 visbiases_3; % hid_out_3;
            %load g_rbm_3 vishid_3 hidbiases_3 visbiases_3 hid_out_3;
        end    

        %% Classifer and performance measurements
        readOut_and_Eval;

        %% restart the run:
        clearvars -except z configurations_list; 
        close all;
    end    
end