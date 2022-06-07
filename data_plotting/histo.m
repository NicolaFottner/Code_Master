%%% plottings histograms 
dd = strsplit(date,'-');
clean_date = strcat(dd(1),dd(2)); %without "-YYYY"
%load needed hyperparameters data
load info g_numcases 


%% %%%%%%%%%%%%%%%% for rbm 1 %%%%%%%%%%%%%%%%%%%%%%%%


load testolin/t_model.mat DN
vishid_1 = DN.L{1,1}.vishid;
hidbiases_1 = DN.L{1,1}.hidbiases;
visbiases_1 = DN.L{1,1}.visbiases;
clear DN;

%%% for weights
row_w =size(vishid_1,1);
col_w = size(vishid_1,2);
rbm1_w=zeros(numel(vishid_1),1);
for i=1:size(vishid_1,1)
    rbm1_w(1 + (i-1) * col_w : i * col_w)=vishid_1(i,:);
end
rbm1_bh = hidbiases_1';
rbm1_bv =visbiases_1';

% h_rbm1_w = histogram(rbm1_w);
% h_rbm1_bh = histogram(rbm1_bh);
% h_rbm1_bv = histogram(rbm1_bv);


% f = figure;
% subplot(2,3,1);
% h_w.Normalization = 'probability';
% title(clean_date);
% xlabel('RBM1 Weights'),ylabel('Pr/Frequency')
% %%% for biases
% subplot(2,3,2);
% h_bh.Normalization = 'probability';
% title(clean_date);
% xlabel('RBM1 HidBiases'),ylabel('Pr/Frequency')
% subplot(2,3,3);
% h_bv.Normalization = 'probability';
% title(clean_date);
% xlabel('RBM1 VisBiases'),ylabel('Pr/Frequency')

%%%%%%%%%%%%%%%%%% for rbm 2 %%%%%%%%%%%%%%%%%%%%%%%%

load g_rbm_2.mat vishid_2 hidbiases_2 visbiases_2;
%%% for weights
row_w =size(vishid_2,1);
col_w = size(vishid_2,2);
rbm2_w=zeros(numel(vishid_2),1);
for i=1:size(vishid_2,1)
    rbm2_w(1 + (i-1) * col_w : i * col_w)=vishid_2(i,:);
end
rbm2_bh = hidbiases_2';
rbm2_bv =visbiases_2';

% h_w = histogram(rbm2_w);
% h_bh = histogram(rbm2_bh);
% h_bv = histogram(rbm2_bv);
% .NORMALIZATION = probability



% subplot(2,3,4);
% % subplot(3,1,1);
% 
% h_w.Normalization = 'probability';
% title(clean_date + " w/ B_size: " + int2str(g_numcases));
% xlabel('RBM2 Weights'),ylabel('Pr/Frequency')
% 
% %%% forbiases
% subplot(2,3,5);
% % subplot(3,1,2);
% h_bh.Normalization = 'probability';
% title(clean_date + " w/ Bsize: " + int2str(g_numcases));
% xlabel('RBM2 HidBiases'),ylabel('Pr/Frequency')
% subplot(2,3,6);
% % subplot(3,1,3);
% h_bv.Normalization = 'probability';
% title(clean_date + " w/ Bsize: " + int2str(g_numcases));
% xlabel('RBM2 VisBiases'),ylabel('Pr/Frequency')
% file_name = "data_plotting/histograms.pdf";
% exportgraphics(f,file_name)

%% %%%%%%%%%%%%%%%% for rbm 3 %%%%%%%%%%%%%%%%%%%%%%%%

if numhid3 ~= 0
    load g_rbm_3.mat vishid_3 hidbiases_3 visbiases_3;
    row_w =size(vishid_3,1);
    col_w = size(vishid_3,2);
    rbm3_w=zeros(numel(vishid_3),1);
    for i=1:size(vishid_3,1)
        rbm3_w(1 + (i-1) * col_w : i * col_w)=vishid_3(i,:);
    end
    rbm3_bh = hidbiases_3';
    rbm3_bv =visbiases_3';  
end

%% gather all 

histo_rbm1.weights = rbm1_w;
histo_rbm1.bias_h = rbm1_bh;
histo_rbm1.bias_v = rbm1_bv;
histo_rbm2.weights = rbm2_w;
histo_rbm2.bias_h = rbm2_bh;
histo_rbm2.bias_v = rbm2_bv;
if numhid3 ~= 0 
    histo_rbm3.weights = rbm3_w;
    histo_rbm3.bias_h = rbm3_bh;
    histo_rbm3.bias_v = rbm3_bv;
end

histograms.rbm1 = histo_rbm1;
histograms.rbm2 = histo_rbm2;
if numhid3 ~= 0 
    histograms.rbm3 = histo_rbm3;
end
