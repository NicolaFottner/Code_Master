clearvars;
dd = strsplit(date,'-');clean_date = strcat(dd(1),dd(2)); %without "-YY
% Computing average of CE-assesments stores in file /CE_22_April

geo_shape_class = 6;
addpath("plots_results/classf_perf6/");
addpath("Evals/");
numhid3 = true;

%% 2 Layer Case
if numhid3 == false
    %% For 6 class case
    if geo_shape_class == 6 
        addpath("plots_results/Id_basedOnGeoS6/1300n/")
        sourceDir = 'plots_results/Id_basedOnGeoS6/1300n/'; % already in 64x64 format
        fprintf(1,'Importing Data from runned simulations \n');
        loadData = dir([sourceDir '*.mat']);
        acc_shape =[];
        acc_letter = [];
        acc_pletter = [];
        mode_l=zeros(size(loadData,1),6);
        mode_pl=zeros(size(loadData,1),6);
        epoch = [];
        acc1 =zeros(size(loadData,1),6);acc2 =zeros(size(loadData,1),6);
        l_pdf_a = [];l_pdf_h = [];l_pdf_m = [];l_pdf_u = [];l_pdf_t = [];l_pdf_x = [];     
        pl_pdf_a = [];pl_pdf_h = [];pl_pdf_m = [];pl_pdf_u = [];pl_pdf_t = [];pl_pdf_x = [];     
        for i=1:length(loadData)
            load([sourceDir loadData(i).name],"accuracy_l","letter_pdr","pletter_pdr","accuracy_s","accuracy_pl","final_epoch","table_letter","table_pletter");
            acc_letter = [acc_letter;accuracy_l];
            acc_pletter = [acc_pletter;accuracy_pl];
            acc_shape = [acc_shape;accuracy_s];
            epoch = [epoch;final_epoch];
            mode_l(i,:) = table_letter.Mode;
            mode_pl(i,:)=table_pletter.pMode;
            acc1(i,:) = table_letter.Acc;
            acc2(i,:) = table_pletter.Acc;
            l_pdf_a = [l_pdf_a;letter_pdr(1,:)];
            l_pdf_h = [l_pdf_h;letter_pdr(2,:)];
            l_pdf_m = [l_pdf_m;letter_pdr(3,:)];
            l_pdf_u = [l_pdf_u;letter_pdr(4,:)];
            l_pdf_t = [l_pdf_t;letter_pdr(5,:)];
            l_pdf_x = [l_pdf_x;letter_pdr(6,:)];
            pl_pdf_a = [pl_pdf_a;pletter_pdr(1,:)];
            pl_pdf_h = [pl_pdf_h;pletter_pdr(2,:)];
            pl_pdf_m = [pl_pdf_m;pletter_pdr(3,:)];
            pl_pdf_u = [pl_pdf_u;pletter_pdr(4,:)];
            pl_pdf_t = [pl_pdf_t;pletter_pdr(5,:)];
            pl_pdf_x = [pl_pdf_x;pletter_pdr(6,:)];
        end
        mean_acc_s  = mean(acc_shape);
        mean_acc_l  = mean(acc_letter);
        mean_acc_pl  = mean(acc_pletter);
        mean_epoch = mean(epoch);
        modes_of_mode_L = mode(mode_l);
        modes_of_mode_pL = mode(mode_pl);
        letter_Acc = mean(acc1,1)';
        pletter_Acc = mean(acc2,1)';
    
        mean_l_pdf_a = mean(l_pdf_a,1);
        mean_l_pdf_h = mean(l_pdf_h,1);
        mean_l_pdf_m = mean(l_pdf_m,1);
        mean_l_pdf_u = mean(l_pdf_u,1);
        mean_l_pdf_t = mean(l_pdf_t,1);
        mean_l_pdf_x = mean(l_pdf_x,1);
        mean_pl_pdf_a = mean(pl_pdf_a,1);
        mean_pl_pdf_h = mean(pl_pdf_h,1);
        mean_pl_pdf_m = mean(pl_pdf_m,1);
        mean_pl_pdf_u = mean(pl_pdf_u,1);
        mean_pl_pdf_t = mean(pl_pdf_t,1);
        mean_pl_pdf_x = mean(pl_pdf_x,1);
        
        Targets = ["GeoShape";"Letter";"PLetter"];
        Accuracy = [mean_acc_s;mean_acc_l;mean_acc_pl];
        Modes = [[1,2,3,4,5,6];modes_of_mode_L;modes_of_mode_pL];
        Id_BasedonS = table(Targets,Accuracy,Modes);
        lTargets = ["(p/)A" ; "(p/)H"; "(p/)M"; "(p/)U"; "(p/)T"; "(p/)X"];
        Modes_L=modes_of_mode_L';
        Modes_pL=modes_of_mode_pL';
        Id_BasedonS_details = table(lTargets,letter_Acc,pletter_Acc,Modes_L,Modes_pL);
        pdf_Targets = ["A" ; "H"; "M"; "U"; "T"; "X";"pA" ; "pH"; "pM"; "pU"; "pT"; "pX"];
        Models_Output_PDF = [mean_l_pdf_a;mean_l_pdf_h;mean_l_pdf_m;mean_l_pdf_u;mean_l_pdf_t;mean_l_pdf_x; ...
            mean_pl_pdf_a;mean_pl_pdf_h;mean_pl_pdf_m;mean_pl_pdf_u;mean_pl_pdf_t;mean_pl_pdf_x];
        Id_BasedonS_PDFs = table(pdf_Targets,Models_Output_PDF);    
        
% 12 ex.:
%                 Targets = ["GeoShape";"Letter";"PLetter"];
%         Accuracy = [d_mean_acc_s;d_mean_acc_l;d_mean_acc_pl];
%         Modes = [[1,2,3,4,5,6];Modes_L;Modes_pL];
%         Id_BasedonS = table(Targets,Accuracy,Modes);
%         lTargets = ["(p/)A" ; "(p/)H"; "(p/)M"; "(p/)U"; "(p/)T"; "(p/)X"];
%         Modes_L=Modes_L';
%         Modes_pL=Modes_pL';
%         Id_BasedonS_details = table(lTargets,letter_Acc,pletter_Acc,Modes_L,Modes_pL);
%         


        % Compute Train classifier performance
        addpath("plots_results/classf_perf6/train/1300n/");
        sourceDir = 'plots_results/classf_perf6/train/1300n/'; % already in 64x64 format
        fprintf(1,'Importing Data from runned simulations \n');
        loadData = dir([sourceDir '*.mat']);
        xte_acc_1=[];xte_acc_2=[];xte_loss_1=[];xte_loss_2=[];xtr_acc_1=[];xtr_acc_2=[];xtr_loss_1=[];xtr_loss_2=[];m_ep = [];
        for i=1:length(loadData)
            load([sourceDir loadData(i).name],"te_acc_1","te_acc_2","te_loss_1","te_loss_2","tr_acc_1","tr_acc_2","tr_loss_1","tr_loss_2","final_epoch");
            xte_acc_1 = [xte_acc_1;te_acc_1];
            xte_acc_2 = [xte_acc_2;te_acc_2];
            xte_loss_1 = [xte_loss_1;te_loss_1];
            xte_loss_2 = [xte_loss_2;te_loss_2];
            xtr_acc_1 = [xtr_acc_1;tr_acc_1];
            xtr_acc_2 = [xtr_acc_2;tr_acc_2];
            xtr_loss_1 = [xtr_loss_1;tr_loss_1];
            xtr_loss_2 = [xtr_loss_2;tr_loss_2];
            m_ep=[m_ep;final_epoch];
        end
        te_acc1  = mean(xte_acc_1);
        te_acc2  = mean(xte_acc_2);
        te_loss1  = mean(xte_loss_1);
        te_loss2  = mean(xte_loss_2);
        tr_acc1  = mean(xtr_acc_1);
        tr_acc2  = mean(xtr_acc_2);
        tr_loss1  = mean(xtr_loss_1);
        tr_loss2  = mean(xtr_loss_2);
        final_epoch = mean(m_ep);
        earlystopping = true;
        
        X = ["Final_layer";"From_RBM1";"Epochs"];
        tr_acc = [tr_acc2;tr_acc1;NaN];
        te_acc = [te_acc2;te_acc1;NaN];
        tr_loss=[tr_loss2;tr_loss1;NaN];
        te_loss = [te_loss2;te_loss1;NaN];
        Epoch = [NaN;NaN;final_epoch];
        Classifier = table(X,tr_acc,te_acc,tr_loss,te_loss,Epoch);
            
        % Compute Detailed Train classifier performance
        % divide in finder dependend on the models' architecture
        addpath("plots_results/classf_perf6/details/1300n/");
        sourceDir = 'plots_results/classf_perf6/details/1300n/'; % already in 64x64 format
        fprintf(1,'Importing Data from runned simulations \n');
        loadData = dir([sourceDir '*.mat']);    
        l_acc1 = [];l_acc2 = [];l_acc3 = [];l_acc4 = [];l_acc5 = [];l_acc6 = [];
        xte_acc_2=[];xtr_acc_2=[];epochs = [];
        for i=1:length(loadData)
            load([sourceDir loadData(i).name],'acc1','acc2','acc3','acc4', 'acc5' ...
                ,'acc6','numhid2','final_epoch');
            xte_acc_2 = [xte_acc_2;te_acc_2];
            xtr_acc_2 = [xtr_acc_2;tr_acc_2];
            l_acc1 = [l_acc1;acc1];
            l_acc2 = [l_acc2;acc2];
            l_acc3 = [l_acc3;acc3];
            l_acc4 = [l_acc4;acc4];
            l_acc5 = [l_acc5;acc5];
            l_acc6 = [l_acc6;acc6];
            epochs= [epochs;final_epoch];
        end
        te_acc2  = mean(xte_acc_2);
        tr_acc2  = mean(xtr_acc_2);
        acc1 = mean(l_acc1);
        acc2 = mean(l_acc2);
        acc3 = mean(l_acc3);
        acc4 = mean(l_acc4);
        acc5 = mean(l_acc5);
        acc6 = mean(l_acc6);
        epoch = mean(epochs);
    %     filename = "sim_classDe_900";
    %     save(filename,'acc1','acc2','acc3','acc4', 'acc5' ...
    %             ,'acc6','acc7','acc8','acc9','acc10', 'acc11','acc12','epoch','earlystopping','te_acc2','tr_acc2');
    %  
    
        Output = ["1";"2";"3";"4";"5";"6"];
        Accuracy = [acc1;acc2;acc3;acc4;acc5;acc6];
        Classifier_Details = table(Output,Accuracy);
    
        filename = "Evals/sim6_1300n";
        save(filename,'Classifier','Classifier_Details','Id_BasedonS','Id_BasedonS_details','Id_BasedonS_PDFs');
    
            
    elseif geo_shape_class == 12
         %% now for the 12 class case
        sourceDir = 'plots_results/Id_basedOnGeoS/900/'; % already in 64x64 format
        fprintf(1,'Importing Data from runned simulations \n');
        loadData = dir([sourceDir '*.mat']);
        d_acc_shape =[];
        dacc_letter = [];
        d_acc_pletter = [];
        d_mode_l=zeros(size(loadData,1),6);
        d_mode_pl=zeros(size(loadData,1),6);
        acc1=zeros(size(loadData,1),6);
        acc2=zeros(size(loadData,1),6);
        l_pdf_a = [];l_pdf_h = [];l_pdf_m = [];l_pdf_u = [];l_pdf_t = [];l_pdf_x = [];     
        pl_pdf_a = [];pl_pdf_h = [];pl_pdf_m = [];pl_pdf_u = [];pl_pdf_t = [];pl_pdf_x = [];d_epoch=[];
        for i=1:length(loadData)
            load([sourceDir loadData(i).name],"accuracy_l","letter_pdr","pletter_pdr","accuracy_s","accuracy_pl","final_epoch","table_letter","table_pletter");
            dacc_letter = [dacc_letter;accuracy_l];
            d_acc_pletter = [d_acc_pletter;accuracy_pl];
            d_acc_shape = [d_acc_shape;accuracy_s];
            d_epoch = [d_epoch;final_epoch];
            d_mode_l(i,:) = table_letter.Mode;
            d_mode_pl(i,:)=table_pletter.pMode;
            acc1(i,:) = table_letter.Acc;
            acc2(i,:) = table_pletter.Acc;
            l_pdf_a = [l_pdf_a;letter_pdr(1,:)];
            l_pdf_h = [l_pdf_h;letter_pdr(2,:)];
            l_pdf_m = [l_pdf_m;letter_pdr(3,:)];
            l_pdf_u = [l_pdf_u;letter_pdr(4,:)];
            l_pdf_t = [l_pdf_t;letter_pdr(5,:)];
            l_pdf_x = [l_pdf_x;letter_pdr(6,:)];
            pl_pdf_a = [pl_pdf_a;pletter_pdr(1,:)];
            pl_pdf_h = [pl_pdf_h;pletter_pdr(2,:)];
            pl_pdf_m = [pl_pdf_m;pletter_pdr(3,:)];
            pl_pdf_u = [pl_pdf_u;pletter_pdr(4,:)];
            pl_pdf_t = [pl_pdf_t;pletter_pdr(5,:)];
            pl_pdf_x = [pl_pdf_x;pletter_pdr(6,:)];
        end
        d_mean_acc_s  = mean(d_acc_shape);
        d_mean_acc_l  = mean(dacc_letter);
        d_mean_acc_pl  = mean(d_acc_pletter);
        epoch = mean(d_epoch);
        Modes_L = mode(d_mode_l);
        Modes_pL = mode(d_mode_pl);
        letter_Acc = mean(acc1,1)';
        pletter_Acc = mean(acc2,1)';
        mean_l_pdf_a = mean(l_pdf_a,1);
        mean_l_pdf_h = mean(l_pdf_h,1);
        mean_l_pdf_m = mean(l_pdf_m,1);
        mean_l_pdf_u = mean(l_pdf_u,1);
        mean_l_pdf_t = mean(l_pdf_t,1);
        mean_l_pdf_x = mean(l_pdf_x,1);
        mean_pl_pdf_a = mean(pl_pdf_a,1);
        mean_pl_pdf_h = mean(pl_pdf_h,1);
        mean_pl_pdf_m = mean(pl_pdf_m,1);
        mean_pl_pdf_u = mean(pl_pdf_u,1);
        mean_pl_pdf_t = mean(pl_pdf_t,1);
        mean_pl_pdf_x = mean(pl_pdf_x,1);
        Targets = ["GeoShape";"Letter";"PLetter"];
        Accuracy = [d_mean_acc_s;d_mean_acc_l;d_mean_acc_pl];
        Modes = [[1,2,3,4,5,6];Modes_L;Modes_pL];
        Id_BasedonS = table(Targets,Accuracy,Modes);
        lTargets = ["(p/)A" ; "(p/)H"; "(p/)M"; "(p/)U"; "(p/)T"; "(p/)X"];
        Modes_L=Modes_L';
        Modes_pL=Modes_pL';
        Id_BasedonS_details = table(lTargets,letter_Acc,pletter_Acc,Modes_L,Modes_pL);
        
        pdf_Targets = ["A" ; "H"; "M"; "U"; "T"; "X";"pA" ; "pH"; "pM"; "pU"; "pT"; "pX"];
    %     l_pdf = [mean_l_pdf_a;mean_l_pdf_h;mean_l_pdf_m;mean_l_pdf_u;mean_l_pdf_t;mean_l_pdf_x];
    %     pl_pdf = [mean_pl_pdf_a;mean_pl_pdf_h;mean_pl_pdf_m;mean_pl_pdf_u;mean_pl_pdf_t;mean_pl_pdf_x];
        Models_Output_PDF = [mean_l_pdf_a;mean_l_pdf_h;mean_l_pdf_m;mean_l_pdf_u;mean_l_pdf_t;mean_l_pdf_x; ...
            mean_pl_pdf_a;mean_pl_pdf_h;mean_pl_pdf_m;mean_pl_pdf_u;mean_pl_pdf_t;mean_pl_pdf_x];
        Id_BasedonS_PDFs = table(pdf_Targets,Models_Output_PDF);

        % Compute Train classifier performance
        addpath("plots_results/classf_perf/train/900/");
        sourceDir = 'plots_results/classf_perf/train/900/'; % already in 64x64 format
        fprintf(1,'Importing Data from runned simulations \n');
        loadData = dir([sourceDir '*.mat']);
        xte_acc_1=[];xte_acc_2=[];xte_loss_1=[];xte_loss_2=[];xtr_acc_1=[];xtr_acc_2=[];xtr_loss_1=[];xtr_loss_2=[];m_ep = [];
        for i=1:length(loadData)
            load([sourceDir loadData(i).name],"te_acc_1","te_acc_2","te_loss_1","te_loss_2","tr_acc_1","tr_acc_2","tr_loss_1","tr_loss_2","final_epoch");
            xte_acc_1 = [xte_acc_1;te_acc_1];
            xte_acc_2 = [xte_acc_2;te_acc_2];
            xte_loss_1 = [xte_loss_1;te_loss_1];
            xte_loss_2 = [xte_loss_2;te_loss_2];
            xtr_acc_1 = [xtr_acc_1;tr_acc_1];
            xtr_acc_2 = [xtr_acc_2;tr_acc_2];
            xtr_loss_1 = [xtr_loss_1;tr_loss_1];
            xtr_loss_2 = [xtr_loss_2;tr_loss_2];
            m_ep=[m_ep;final_epoch];
        end
        te_acc1  = mean(xte_acc_1);
        te_acc2  = mean(xte_acc_2);
        te_loss1  = mean(xte_loss_1);
        te_loss2  = mean(xte_loss_2);
        tr_acc1  = mean(xtr_acc_1);
        tr_acc2  = mean(xtr_acc_2);
        tr_loss1  = mean(xtr_loss_1);
        tr_loss2  = mean(xtr_loss_2);
        final_epoch = mean(m_ep);
        earlystopping = true;
        
        X = ["Final_layer";"From_RBM1";"Epochs"];
        tr_acc = [tr_acc2;tr_acc1;NaN];
        te_acc = [te_acc2;te_acc1;NaN];
        tr_loss=[tr_loss2;tr_loss1;NaN];
        te_loss = [te_loss2;te_loss1;NaN];
        Epoch = [NaN;NaN;final_epoch];
        Classifier = table(X,tr_acc,te_acc,tr_loss,te_loss,Epoch);
    
        % save sim_class_900 te_acc1 te_acc2 te_loss1 te_loss2 tr_acc1 tr_acc2 tr_loss1 tr_loss2 final_epoch earlystopping;
        
        % Compute Detailed Train classifier performance
        % divide in finder dependend on the models' architecture
        addpath("plots_results/classf_perf/details/900/");
        sourceDir = 'plots_results/classf_perf/details/900/'; % already in 64x64 format
        fprintf(1,'Importing Data from runned simulations \n');
        loadData = dir([sourceDir '*.mat']);    
        l_acc1 = [];l_acc2 = [];l_acc3 = [];l_acc4 = [];l_acc5 = [];l_acc6 = [];
        l_acc7 = [];l_acc8 = [];l_acc9 = [];l_acc10 = [];l_acc11 = [];l_acc12 = [];
        xte_acc_2=[];xtr_acc_2=[];epochs = [];
        for i=1:length(loadData)
            load([sourceDir loadData(i).name],'acc1','acc2','acc3','acc4', 'acc5' ...
                ,'acc6','acc7','acc8','acc9','acc10', 'acc11','acc12','numhid2','final_epoch' ,'tr_acc_2','te_acc_2');
            xte_acc_2 = [xte_acc_2;te_acc_2];
            xtr_acc_2 = [xtr_acc_2;tr_acc_2];
            l_acc1 = [l_acc1;acc1];
            l_acc2 = [l_acc2;acc2];
            l_acc3 = [l_acc3;acc3];
            l_acc4 = [l_acc4;acc4];
            l_acc5 = [l_acc5;acc5];
            l_acc6 = [l_acc6;acc6];
            l_acc7 = [l_acc7;acc7];
            l_acc8 = [l_acc8;acc8];
            l_acc9 = [l_acc9;acc9];
            l_acc10 = [l_acc10;acc10];
            l_acc11= [l_acc11;acc11];
            l_acc12= [l_acc12;acc12];
            epochs= [epochs;final_epoch];
        end
        te_acc2  = mean(xte_acc_2);
        tr_acc2  = mean(xtr_acc_2);
        acc1 = mean(l_acc1);
        acc2 = mean(l_acc2);
        acc3 = mean(l_acc3);
        acc4 = mean(l_acc4);
        acc5 = mean(l_acc5);
        acc6 = mean(l_acc6);
        acc7 = mean(l_acc7);
        acc8 = mean(l_acc8);
        acc9 = mean(l_acc9);
        acc10 = mean(l_acc10);
        acc11 = mean(l_acc11);
        acc12 = mean(l_acc12);
        epoch = mean(epochs);
    %     filename = "sim_classDe_900";
    %     save(filename,'acc1','acc2','acc3','acc4', 'acc5' ...
    %             ,'acc6','acc7','acc8','acc9','acc10', 'acc11','acc12','epoch','earlystopping','te_acc2','tr_acc2');
    %  
    
        Output = ["1";"2";"3";"4";"5";"6";"7";"8";"9";"10";"11";"12"];
        Accuracy = [acc1;acc2;acc3;acc4;acc5;acc6;acc7;acc8;acc9;acc10;acc11;acc12];
        Classifier_Details = table(Output,Accuracy);
    
        filename = "Evals/sim_900";
        save(filename,'Classifier','Classifier_Details','Id_BasedonS','Id_BasedonS_details','Id_BasedonS_PDFs');
    end
end

%% All again if there is a third layer and
if numhid3 == true
    %% 12 class case
    if geo_shape_class == 12
        sourceDir = 'plots_results/Id_basedOnGeoS/three/900_300/'; % already in 64x64 format
        fprintf(1,'Importing Data from runned simulations \n');
        loadData = dir([sourceDir '*.mat']);
        d_acc_shape =[];
        dacc_letter = [];
        d_acc_pletter = [];
        d_mode_l=zeros(size(loadData,1),6);
        d_mode_pl=zeros(size(loadData,1),6);
        acc1=zeros(size(loadData,1),6);
        acc2=zeros(size(loadData,1),6);
        l_pdf_a = [];l_pdf_h = [];l_pdf_m = [];l_pdf_u = [];l_pdf_t = [];l_pdf_x = [];     
        pl_pdf_a = [];pl_pdf_h = [];pl_pdf_m = [];pl_pdf_u = [];pl_pdf_t = [];pl_pdf_x = [];d_epoch=[];
        for i=1:length(loadData)
            load([sourceDir loadData(i).name],"accuracy_l","letter_pdr","pletter_pdr","accuracy_s","accuracy_pl","final_epoch","table_letter","table_pletter");
            dacc_letter = [dacc_letter;accuracy_l];
            d_acc_pletter = [d_acc_pletter;accuracy_pl];
            d_acc_shape = [d_acc_shape;accuracy_s];
            d_epoch = [d_epoch;final_epoch];
            d_mode_l(i,:) = table_letter.Mode;
            d_mode_pl(i,:)=table_pletter.pMode;
            acc1(i,:) = table_letter.Acc;
            acc2(i,:) = table_pletter.Acc;
            l_pdf_a = [l_pdf_a;letter_pdr(1,:)];
            l_pdf_h = [l_pdf_h;letter_pdr(2,:)];
            l_pdf_m = [l_pdf_m;letter_pdr(3,:)];
            l_pdf_u = [l_pdf_u;letter_pdr(4,:)];
            l_pdf_t = [l_pdf_t;letter_pdr(5,:)];
            l_pdf_x = [l_pdf_x;letter_pdr(6,:)];
            pl_pdf_a = [pl_pdf_a;pletter_pdr(1,:)];
            pl_pdf_h = [pl_pdf_h;pletter_pdr(2,:)];
            pl_pdf_m = [pl_pdf_m;pletter_pdr(3,:)];
            pl_pdf_u = [pl_pdf_u;pletter_pdr(4,:)];
            pl_pdf_t = [pl_pdf_t;pletter_pdr(5,:)];
            pl_pdf_x = [pl_pdf_x;pletter_pdr(6,:)];
        end
        d_mean_acc_s  = mean(d_acc_shape);
        d_mean_acc_l  = mean(dacc_letter);
        d_mean_acc_pl  = mean(d_acc_pletter);
        epoch = mean(d_epoch);
        Modes_L = mode(d_mode_l);
        Modes_pL = mode(d_mode_pl);
        letter_Acc = mean(acc1,1)';
        pletter_Acc = mean(acc2,1)';
        mean_l_pdf_a = mean(l_pdf_a,1);
        mean_l_pdf_h = mean(l_pdf_h,1);
        mean_l_pdf_m = mean(l_pdf_m,1);
        mean_l_pdf_u = mean(l_pdf_u,1);
        mean_l_pdf_t = mean(l_pdf_t,1);
        mean_l_pdf_x = mean(l_pdf_x,1);
        mean_pl_pdf_a = mean(pl_pdf_a,1);
        mean_pl_pdf_h = mean(pl_pdf_h,1);
        mean_pl_pdf_m = mean(pl_pdf_m,1);
        mean_pl_pdf_u = mean(pl_pdf_u,1);
        mean_pl_pdf_t = mean(pl_pdf_t,1);
        mean_pl_pdf_x = mean(pl_pdf_x,1);
        Targets = ["GeoShape";"Letter";"PLetter"];
        d_Accuracy = [d_mean_acc_s;d_mean_acc_l;d_mean_acc_pl];
        d_Modes = [[1,2,3,4,5,6];Modes_L;Modes_pL];
        lTargets = ["(p/)A" ; "(p/)H"; "(p/)M"; "(p/)U"; "(p/)T"; "(p/)X"];
        pdf_Targets = ["A" ; "H"; "M"; "U"; "T"; "X";"pA" ; "pH"; "pM"; "pU"; "pT"; "pX"];
        Models_Output_PDF = [mean_l_pdf_a;mean_l_pdf_h;mean_l_pdf_m;mean_l_pdf_u;mean_l_pdf_t;mean_l_pdf_x; ...
            mean_pl_pdf_a;mean_pl_pdf_h;mean_pl_pdf_m;mean_pl_pdf_u;mean_pl_pdf_t;mean_pl_pdf_x];
        Modes_L=Modes_L';
        Modes_pL=Modes_pL';
        Table_Results = table(Targets,d_Accuracy,d_Modes);
        Table_Results_detail = table(lTargets,letter_Acc,pletter_Acc,Modes_L,Modes_pL);
        Table_pdfs = table(pdf_Targets,Models_Output_PDF);
    
        % Compute Train classifier performance
        addpath("plots_results/classf_perf/train/three/900_300/");
        sourceDir = 'plots_results/classf_perf/train/three/900_300/'; % already in 64x64 format
        fprintf(1,'Importing Data from runned simulations \n');
        loadData = dir([sourceDir '*.mat']);
        xte_acc_1=[];xte_acc_2=[];xte_acc_3=[];xte_loss_1=[];xte_loss_2=[];xte_loss_3=[];
        xtr_acc_1=[];xtr_acc_2=[];xtr_acc_3=[];xtr_loss_1=[];xtr_loss_2=[];xtr_loss_3=[];m_ep = [];
        for i=1:length(loadData)
            load([sourceDir loadData(i).name],"te_acc_1","te_acc_2","te_acc_3","te_loss_1","te_loss_2","te_loss_3","tr_acc_1","tr_acc_2","tr_acc_3","tr_loss_1","tr_loss_2","tr_loss_3","final_epoch");
            xte_acc_1 = [xte_acc_1;te_acc_1];
            xte_acc_2 = [xte_acc_2;te_acc_2];
            xte_acc_3 = [xte_acc_3;te_acc_3];
            xte_loss_1 = [xte_loss_1;te_loss_1];
            xte_loss_2 = [xte_loss_2;te_loss_2];
            xte_loss_3 = [xte_loss_3;te_loss_3];
            xtr_acc_1 = [xtr_acc_1;tr_acc_1];
            xtr_acc_2 = [xtr_acc_2;tr_acc_2];
            xtr_acc_3 = [xtr_acc_3;tr_acc_3];
            xtr_loss_1 = [xtr_loss_1;tr_loss_1];
            xtr_loss_2 = [xtr_loss_2;tr_loss_2];
            xtr_loss_3 = [xtr_loss_3;tr_loss_3];
            m_ep=[m_ep;final_epoch];
        end
        te_acc1  = mean(xte_acc_1);
        te_acc2  = mean(xte_acc_2);
        te_acc3  = mean(xte_acc_3);
        te_loss1  = mean(xte_loss_1);
        te_loss2  = mean(xte_loss_2);
        te_loss3  = mean(xte_loss_3);
        tr_acc1  = mean(xtr_acc_1);
        tr_acc2  = mean(xtr_acc_2);
        tr_acc3  = mean(xtr_acc_3);
        tr_loss1  = mean(xtr_loss_1);
        tr_loss2  = mean(xtr_loss_2);
        tr_loss3  = mean(xtr_loss_3);
        final_epoch = mean(m_ep);
        earlystopping = true;
           
        X = ["Final_layer";"From_RBM2";"From_RBM1";"Epochs"];
        tr_acc = [tr_acc3;tr_acc2;tr_acc1;NaN];
        te_acc = [te_acc3;te_acc2;te_acc1;NaN];
        tr_loss=[tr_loss3;tr_loss2;tr_loss1;NaN];
        te_loss = [te_loss3;te_loss2;te_loss1;NaN];
        Epoch = [NaN;NaN;NaN;final_epoch];          
    
        Table_class = table(X,tr_acc,te_acc,tr_loss,te_loss,Epoch);
    
        %save sim_class_300 te_acc1 te_acc2 te_loss1 te_loss2 tr_acc1 tr_acc2 tr_loss1 tr_loss2 final_epoch earlystopping;
        
        % Compute Detailed Train classifier performance
        % divide in finder dependend on the models' architecture
        addpath("plots_results/classf_perf/details/three/900_300/");
        sourceDir = 'plots_results/classf_perf/details/three/900_300/'; % already in 64x64 format
        fprintf(1,'Importing Data from runned simulations \n');
        loadData = dir([sourceDir '*.mat']);    
        l_acc1 = [];l_acc2 = [];l_acc3 = [];l_acc4 = [];l_acc5 = [];l_acc6 = [];
        l_acc7 = [];l_acc8 = [];l_acc9 = [];l_acc10 = [];l_acc11 = [];l_acc12 = [];
        epochs = [];
        for i=1:length(loadData)
            load([sourceDir loadData(i).name],'acc1','acc2','acc3','acc4', 'acc5' ...
                ,'acc6','acc7','acc8','acc9','acc10', 'acc11','acc12','numhid2','final_epoch');
            l_acc1 = [l_acc1;acc1];
            l_acc2 = [l_acc2;acc2];
            l_acc3 = [l_acc3;acc3];
            l_acc4 = [l_acc4;acc4];
            l_acc5 = [l_acc5;acc5];
            l_acc6 = [l_acc6;acc6];
            l_acc7 = [l_acc7;acc7];
            l_acc8 = [l_acc8;acc8];
            l_acc9 = [l_acc9;acc9];
            l_acc10 = [l_acc10;acc10];
            l_acc11= [l_acc11;acc11];
            l_acc12= [l_acc12;acc12];
            epochs= [epochs;final_epoch];
        end
        acc1 = mean(l_acc1);
        acc2 = mean(l_acc2);
        acc3 = mean(l_acc3);
        acc4 = mean(l_acc4);
        acc5 = mean(l_acc5);
        acc6 = mean(l_acc6);
        acc7 = mean(l_acc7);
        acc8 = mean(l_acc8);
        acc9 = mean(l_acc9);
        acc10 = mean(l_acc10);
        acc11 = mean(l_acc11);
        acc12 = mean(l_acc12);
        Output = ["1";"2";"3";"4";"5";"6";"7";"8";"9";"10";"11";"12"];
        Accuracy = [acc1;acc2;acc3;acc4;acc5;acc6;acc7;acc8;acc9;acc10;acc11;acc12];
        Table_class_details = table(Output,Accuracy);
    
        filename = "Evals/sim_900_300";
        save(filename,'Table_class','Table_class_details','Table_Results','Table_Results_detail','Table_pdfs');
    
    %         filename = "sim_classDe_300";
    %         save(filename,'acc1','acc2','acc3','acc4', 'acc5' ...
    %                 ,'acc6','acc7','acc8','acc9','acc10', 'acc11','acc12','epoch','earlystopping','te_acc2','tr_acc2');
    %      
    elseif geo_shape_class == 6
        %% for 6 for three Layers
        sourceDir = 'plots_results/Id_basedOnGeoS6/1300_100n/'; % already in 64x64 format
        fprintf(1,'Importing Data from runned simulations \n');
        loadData = dir([sourceDir '*.mat']);
        d_acc_shape =[];
        dacc_letter = [];
        d_acc_pletter = [];
        d_mode_l=zeros(size(loadData,1),6);
        d_mode_pl=zeros(size(loadData,1),6);
        acc1=zeros(size(loadData,1),6);
        acc2=zeros(size(loadData,1),6);
        l_pdf_a = [];l_pdf_h = [];l_pdf_m = [];l_pdf_u = [];l_pdf_t = [];l_pdf_x = [];     
        pl_pdf_a = [];pl_pdf_h = [];pl_pdf_m = [];pl_pdf_u = [];pl_pdf_t = [];pl_pdf_x = [];d_epoch=[];
        for i=1:length(loadData)
            load([sourceDir loadData(i).name],"accuracy_l","letter_pdr","pletter_pdr","accuracy_s","accuracy_pl","final_epoch","table_letter","table_pletter");
            dacc_letter = [dacc_letter;accuracy_l];
            d_acc_pletter = [d_acc_pletter;accuracy_pl];
            d_acc_shape = [d_acc_shape;accuracy_s];
            d_epoch = [d_epoch;final_epoch];
            d_mode_l(i,:) = table_letter.Mode;
            d_mode_pl(i,:)=table_pletter.pMode;
            acc1(i,:) = table_letter.Acc;
            acc2(i,:) = table_pletter.Acc;
            l_pdf_a = [l_pdf_a;letter_pdr(1,:)];
            l_pdf_h = [l_pdf_h;letter_pdr(2,:)];
            l_pdf_m = [l_pdf_m;letter_pdr(3,:)];
            l_pdf_u = [l_pdf_u;letter_pdr(4,:)];
            l_pdf_t = [l_pdf_t;letter_pdr(5,:)];
            l_pdf_x = [l_pdf_x;letter_pdr(6,:)];
            pl_pdf_a = [pl_pdf_a;pletter_pdr(1,:)];
            pl_pdf_h = [pl_pdf_h;pletter_pdr(2,:)];
            pl_pdf_m = [pl_pdf_m;pletter_pdr(3,:)];
            pl_pdf_u = [pl_pdf_u;pletter_pdr(4,:)];
            pl_pdf_t = [pl_pdf_t;pletter_pdr(5,:)];
            pl_pdf_x = [pl_pdf_x;pletter_pdr(6,:)];
        end
        d_mean_acc_s  = mean(d_acc_shape);
        d_mean_acc_l  = mean(dacc_letter);
        d_mean_acc_pl  = mean(d_acc_pletter);
        epoch = mean(d_epoch);
        Modes_L = mode(d_mode_l);
        Modes_pL = mode(d_mode_pl);
        letter_Acc = mean(acc1,1)';
        pletter_Acc = mean(acc2,1)';
        mean_l_pdf_a = mean(l_pdf_a,1);
        mean_l_pdf_h = mean(l_pdf_h,1);
        mean_l_pdf_m = mean(l_pdf_m,1);
        mean_l_pdf_u = mean(l_pdf_u,1);
        mean_l_pdf_t = mean(l_pdf_t,1);
        mean_l_pdf_x = mean(l_pdf_x,1);
        mean_pl_pdf_a = mean(pl_pdf_a,1);
        mean_pl_pdf_h = mean(pl_pdf_h,1);
        mean_pl_pdf_m = mean(pl_pdf_m,1);
        mean_pl_pdf_u = mean(pl_pdf_u,1);
        mean_pl_pdf_t = mean(pl_pdf_t,1);
        mean_pl_pdf_x = mean(pl_pdf_x,1);
        Targets = ["GeoShape";"Letter";"PLetter"];
        d_Accuracy = [d_mean_acc_s;d_mean_acc_l;d_mean_acc_pl];
        d_Modes = [[1,2,3,4,5,6];Modes_L;Modes_pL];
        lTargets = ["(p/)A" ; "(p/)H"; "(p/)M"; "(p/)U"; "(p/)T"; "(p/)X"];
        pdf_Targets = ["A" ; "H"; "M"; "U"; "T"; "X";"pA" ; "pH"; "pM"; "pU"; "pT"; "pX"];
        Models_Output_PDF = [mean_l_pdf_a;mean_l_pdf_h;mean_l_pdf_m;mean_l_pdf_u;mean_l_pdf_t;mean_l_pdf_x; ...
            mean_pl_pdf_a;mean_pl_pdf_h;mean_pl_pdf_m;mean_pl_pdf_u;mean_pl_pdf_t;mean_pl_pdf_x];
        Modes_L=Modes_L';
        Modes_pL=Modes_pL';
        Id_BasedonS = table(Targets,d_Accuracy,d_Modes);
        Id_BasedonS_details = table(lTargets,letter_Acc,pletter_Acc,Modes_L,Modes_pL);
        Id_BasedonS_PDFs = table(pdf_Targets,Models_Output_PDF);
    
        % Compute Train classifier performance
        addpath("plots_results/classf_perf6/train/three/1300_100n/");
        sourceDir = 'plots_results/classf_perf6/train/three/1300_100n/'; % already in 64x64 format
        fprintf(1,'Importing Data from runned simulations \n');
        loadData = dir([sourceDir '*.mat']);
        xte_acc_1=[];xte_acc_2=[];xte_acc_3=[];xte_loss_1=[];xte_loss_2=[];xte_loss_3=[];
        xtr_acc_1=[];xtr_acc_2=[];xtr_acc_3=[];xtr_loss_1=[];xtr_loss_2=[];xtr_loss_3=[];m_ep = [];
        for i=1:length(loadData)
            load([sourceDir loadData(i).name],"te_acc_1","te_acc_2","te_acc_3","te_loss_1","te_loss_2","te_loss_3","tr_acc_1","tr_acc_2","tr_acc_3","tr_loss_1","tr_loss_2","tr_loss_3","final_epoch");
            xte_acc_1 = [xte_acc_1;te_acc_1];
            xte_acc_2 = [xte_acc_2;te_acc_2];
            xte_acc_3 = [xte_acc_3;te_acc_3];
            xte_loss_1 = [xte_loss_1;te_loss_1];
            xte_loss_2 = [xte_loss_2;te_loss_2];
            xte_loss_3 = [xte_loss_3;te_loss_3];
            xtr_acc_1 = [xtr_acc_1;tr_acc_1];
            xtr_acc_2 = [xtr_acc_2;tr_acc_2];
            xtr_acc_3 = [xtr_acc_3;tr_acc_3];
            xtr_loss_1 = [xtr_loss_1;tr_loss_1];
            xtr_loss_2 = [xtr_loss_2;tr_loss_2];
            xtr_loss_3 = [xtr_loss_3;tr_loss_3];
            m_ep=[m_ep;final_epoch];
        end
        te_acc1  = mean(xte_acc_1);
        te_acc2  = mean(xte_acc_2);
        te_acc3  = mean(xte_acc_3);
        te_loss1  = mean(xte_loss_1);
        te_loss2  = mean(xte_loss_2);
        te_loss3  = mean(xte_loss_3);
        tr_acc1  = mean(xtr_acc_1);
        tr_acc2  = mean(xtr_acc_2);
        tr_acc3  = mean(xtr_acc_3);
        tr_loss1  = mean(xtr_loss_1);
        tr_loss2  = mean(xtr_loss_2);
        tr_loss3  = mean(xtr_loss_3);
        final_epoch = mean(m_ep);
        earlystopping = true;
           
        X = ["Final_layer";"From_RBM2";"From_RBM1";"Epochs"];
        tr_acc = [tr_acc3;tr_acc2;tr_acc1;NaN];
        te_acc = [te_acc3;te_acc2;te_acc1;NaN];
        tr_loss=[tr_loss3;tr_loss2;tr_loss1;NaN];
        te_loss = [te_loss3;te_loss2;te_loss1;NaN];
        Epoch = [NaN;NaN;NaN;final_epoch];          
    
        Table_class = table(X,tr_acc,te_acc,tr_loss,te_loss,Epoch);
    
        %save sim_class_300 te_acc1 te_acc2 te_loss1 te_loss2 tr_acc1 tr_acc2 tr_loss1 tr_loss2 final_epoch earlystopping;
        
        % Compute Detailed Train classifier performance
        % divide in finder dependend on the models' architecture
        addpath("plots_results/classf_perf6/details/three/1300_100n/");
        sourceDir = 'plots_results/classf_perf6/details/three/1300_100n/'; % already in 64x64 format
        fprintf(1,'Importing Data from runned simulations \n');
        loadData = dir([sourceDir '*.mat']);    
        l_acc1 = [];l_acc2 = [];l_acc3 = [];l_acc4 = [];l_acc5 = [];l_acc6 = [];
        epochs = [];
        for i=1:length(loadData)
            load([sourceDir loadData(i).name],'acc1','acc2','acc3','acc4', 'acc5' ...
                ,'acc6','numhid2','numhid3','final_epoch');
            l_acc1 = [l_acc1;acc1];
            l_acc2 = [l_acc2;acc2];
            l_acc3 = [l_acc3;acc3];
            l_acc4 = [l_acc4;acc4];
            l_acc5 = [l_acc5;acc5];
            l_acc6 = [l_acc6;acc6];
            epochs= [epochs;final_epoch];
        end
        acc1 = mean(l_acc1);
        acc2 = mean(l_acc2);
        acc3 = mean(l_acc3);
        acc4 = mean(l_acc4);
        acc5 = mean(l_acc5);
        acc6 = mean(l_acc6);
        Output = ["1";"2";"3";"4";"5";"6"];
        Accuracy = [acc1;acc2;acc3;acc4;acc5;acc6];
        Table_class_details = table(Output,Accuracy);
    
        filename = "Evals/sim6_1300_100n";
        save(filename,'Table_class','Table_class_details','Id_BasedonS','Id_BasedonS_details','Id_BasedonS_PDFs');
    end
end


    








