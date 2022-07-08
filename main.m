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
% Evals/ --- file with everything - from overfitting measure to CE assesm.

configurations;

% RUN WITH:
% SHAPE_POS3
% LETTER_POS1
% newCE_data_(pos1)


configurations_list = [];
elem.layer2=350;
elem.layer3= 0;
elem.dropout= 0;
elem.minibatchsize= 24;
configurations_list = [configurations_list;elem];

for z=1:size(configurations_list,1)
    for ii=1:5    %% --- 10
        % initialize hyperparameters
        maxepoch=500; % 500
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
            p_layer1 = 1;
            p_layer2 = 1;
            a1 = 0;a2=0;a3=0;
        end
        no_N_img = false;
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

        fprintf(1,'\n\nXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX\n\n');
        fprintf(1,'Number of Iterations: %d\n',ii);
        fprintf(1,'\n\nXXXXXXXX\n\n');
        addpath("testolin/")
        addpath("data_plotting/")
        addpath("data")
        addpath("data/new04Jl/")

        %% Import Data from testolin
        fprintf(1,'Importing first layer from Testolin:\n');
        load t_model DN
        numdims = 1600;
        numbatches = 80000/160;
        numcases = DN.batchsize;
        
        fprintf(1,'Importing data: Geometric Shapes \n');
        
        if g_batchsize == 6
            load batchdata_m6.mat
        elseif g_batchsize == 12
            load batchdata_m12.mat
        elseif g_batchsize == 24
            load batchdata_m24.mat
        end
        fprintf(1,'Start Training. \n'); 
        fprintf(1,'Number Epochs: %3i \n', maxepoch);
        % data for training rbm2:
        [g_numcases g_numdims g_numbatches]=size(g_batchdata);
        % create file to access info from outside
        save info maxepoch numdims numhid numhid2 numbatches numcases g_numbatches g_numcases
        %%% @todo use the above line in the "plotting methods" for modularity

        %% RBM 1 / fetching it from Testolin
        %%%% first rbm layer
        vishid_1 = DN.L{1,1}.vishid;
        hidbiases_1 = DN.L{1,1}.hidbiases;
        visbiases_1 = DN.L{1,1}.visbiases;
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
        rbm_2DropO;
        vishid_2=vishid; hidbiases_2=hidbiases; visbiases_2=visbiases; hid_out_2 = batchposhidprobs_2;
        save g_rbm_2 vishid_2 hidbiases_2 visbiases_2; % hid_out_2;

%         load rbm2_16J11h39.mat vishid_2 hidbiases_2 visbiases_2;
%         load rbm2_16J11h39_err.mat overfitting_g_2 full_rec_err_g
%         addpath("Evals/");
%         load 16Jun_11h39m_H2350_H30.mat properties;
%         final_epoch = properties.epoch2;

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
            rbm_3DropO;
            vishid_3=vishid; hidbiases_3=hidbiases; visbiases_3=visbiases; hid_out_3 = batchposhidprobs_2;
            save g_rbm_3 vishid_3 hidbiases_3 visbiases_3; % hid_out_3;
            %load g_rbm_3 vishid_3 hidbiases_3 visbiases_3 hid_out_3;
        end    

        %% Classifer and performance measurements
        if ~no_N_img % run like usual
            readOut_and_Eval;   
        else         % no first/natural layer
            readOut_and_Eval_noN_img;
        end
        
    
        %%% save network for the training phase
        dd = strsplit(date,'-'); clean_date = strcat(dd(1),dd(2));c=clock; %store date without "-YYYY"
        hour_str = int2str(c(4));
        min_str = int2str(c(5));
        if length(hour_str) == 1
            hour_str = ['0' hour_str(1)];
        end
        if length(min_str) == 1
            min_str = ['0' min_str(1)];
        end
            
        model_name = "Evals/" + clean_date + "_" + hour_str + "h" + min_str + "m_" + "illit_n" + int2str(ii);
        save(model_name,'vishid_2','hidbiases_2', 'visbiases_2', 'W2');

        %% restart the run:
        clearvars -except z ii configurations_list matrix_1 matrix_1_pd matrix_2 matrix_3_outer matrix_3_inner; 
        close all;
    end
dd = strsplit(date,'-'); clean_date = strcat(dd(1),dd(2));c=clock; %store date without "-YYYY"
hour_str = int2str(c(4));
min_str = int2str(c(5));
if length(hour_str) == 1
    hour_str = ['0' hour_str(1)];
end
if length(min_str) == 1
    min_str = ['0' min_str(1)];
end
filename = "Evals/" + clean_date + "_" + hour_str + "h" + min_str+"m_" + "TF_matrix";
save(filename,'matrix_1','matrix_1_pd','matrix_2','matrix_3_outer','matrix_3_inner');
excel_TF_export;
end