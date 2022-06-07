
weights = W2;

dd = strsplit(date,'-'); clean_date = strcat(dd(1),dd(2)); %store date without "-YYYY"
c = clock;

%function [weights, accuracy, te_accuracy,tr_loss,te_loss] = pred_ce_effect(weights,jointdata,lettertarget,shapetarget)
addpath("testolin/")
addpath("data_plotting/")
addpath("data")
%load data
load data/CE_JointData.mat data target_l target_s
%load rbm data:
load t_model DN
vishid_1 = DN.L{1,1}.vishid;
hidbiases_1 = DN.L{1,1}.hidbiases;
clear DN
load g_rbm_2.mat vishid_2 hidbiases_2

%%% Create variables for Table -- visualization purpose
% table with accuracy values for shape
Targets = ["Letter" ; "PseudoLetter"; "All"];
Congruent_Case = zeros(3,1); 
Incongruent_Case =  zeros(3,1); 
Combined = zeros(3,1); 
% table with CE values for shape
CE = zeros(3,1); 
% table with CE values for letters
CE_l = zeros(3,1);
% table with accuracy values for letter
Congruent_Case_l = zeros(3,1); 
Incongruent_Case_l =  zeros(3,1); 
Combined_l = zeros(3,1); 


% pass data throught RBMs:
hid_out_1_d = 1./(1 + exp(-data*vishid_1 - repmat(hidbiases_1,size(data,1),1)));
rbms_pass = 1./(1 + exp(-hid_out_1_d*vishid_2 - repmat(hidbiases_2,size(hid_out_1_d,1),1)));

% add biases
ONES = ones(size(rbms_pass, 1), 1);  
rbms_pass = [rbms_pass ONES];

%create sets for congruent and incongruent cases
cong_cases = zeros(size(rbms_pass,1)/2,size(rbms_pass,2));
incong_cases = zeros(size(rbms_pass,1)/2,size(rbms_pass,2));

cong_ts = zeros(size(rbms_pass,1)/2,size(target_s,2));
cong_tl= zeros(size(rbms_pass,1)/2,size(target_s,2));
incong_ts= zeros(size(rbms_pass,1)/2,size(target_s,2));
incong_tl= zeros(size(rbms_pass,1)/2,size(target_s,2));

j1 = 1;j2 = 1;
for i=1:size(rbms_pass,1)
    idx_l = find(target_l(i,:));
    idx_s = find(target_s(i,:));
    if idx_l == 1 && idx_s == 6 || idx_l == 2 && idx_s == 4 || idx_l == 3 && idx_s == 5 || ...
            idx_l == 4 && idx_s == 1 || idx_l == 5 && idx_s == 2 || idx_l == 6 && idx_s == 3 || ...
                idx_l == 7 && idx_s == 6 || idx_l == 8 && idx_s == 4 || idx_l == 9 && idx_s == 5 || ...
                    idx_l == 10 && idx_s == 1 || idx_l == 11 && idx_s == 2 || idx_l == 12 && idx_s == 3 
        cong_cases(j1,:) = rbms_pass(i,:);
        cong_ts(j1,:) = target_s(i,:);
        cong_tl(j1,:) = target_l(i,:);
        j1 = j1 +1;
    else
        incong_cases(j2,:) = rbms_pass(i,:);
        incong_ts(j2,:) = target_s(i,:);
        incong_tl(j2,:) = target_l(i,:);
        j2 = j2 +1 ; 
    end
end

% Compute general prediction:
pred = rbms_pass*weights;
% [a,b] =max(pred,[],2); --- % a has also  values  > 1 ---- 
% i.e. a is unnormalized
[~, max_act] = max(pred,[],2); % max_act are indices of dim2 in pred of the highest value
[r1,~] = find(target_l'); % find which columns (rows in transpose) are 1
[r2,~] = find(target_s'); % find which columns (rows in transpose) are 1

acc_l = (max_act == r1);
acc_s = (max_act == r2);
accuracy_l = mean(acc_l);
accuracy_s = mean(acc_s);

fprintf(1,'\n Prediction / Assesment of the Congruency Effect\n');
fprintf(1,'\n ------- All targets ------- \n');
fprintf(1,'\n All cases -- on geo-shape \n Accuracy = %d ',accuracy_s);
fprintf(1,'\n If Decision based on letter? \n Accuracy = %d\n ',accuracy_l);
Combined(3) = accuracy_s;
Combined_l(3) = accuracy_l;
% Compute incong and cong predictions:
inc_pred = incong_cases*weights;
cong_pred = cong_cases*weights;

[~, max_act] = max(inc_pred,[],2); 
[r1,~] = find(incong_tl'); 
[r2,~] = find(incong_ts');
acc_l = (max_act == r1);
acc_s = (max_act == r2);
incong_acc_l = mean(acc_l);
incong_acc_s = mean(acc_s);

fprintf(1,'\n Incongrent cases & All targets -- geoshape\n Accuracy = %d ',incong_acc_s);
fprintf(1,'\n If decision based on letter?\n Accuracy = %d\n ',incong_acc_l);
Incongruent_Case(3) = incong_acc_s;
Incongruent_Case_l(3) = incong_acc_l;

[~, max_act] = max(cong_pred,[],2);
[r1,~] = find(cong_tl'); 
[r2,~] = find(cong_ts');
acc_l = (max_act == r1);
acc_s = (max_act == r2);
cong_acc_l = mean(acc_l);
cong_acc_s = mean(acc_s);

fprintf(1,'\n Congrent cases & All targets -- geoshape\n Accuracy = %d ',cong_acc_s);
fprintf(1,'\n If decision based on letter?\n Accuracy = %d\n ',cong_acc_l);
Congruent_Case(3) = cong_acc_s;
Congruent_Case_l(3) = cong_acc_l;
ce = [incong_acc_s * 100,cong_acc_s*100];
diff_ce = cong_acc_s*100 - incong_acc_s * 100;
fprintf(1,'\n\n General Congruency effect of magnitude: %d\n ',diff_ce);
CE(3) = diff_ce;
CE_l(3) = cong_acc_l*100 -incong_acc_l*100;


%%%%%%%%%%%%%%%%%%% GET DATA COMPARING LETTER AND PSEUDOLETTERS %%%%%%%%%%%

% Divide letters and pseudo letters up:
% comparing letters to pseudoletters

letter_cases = zeros(size(rbms_pass,1)/2,size(rbms_pass,2));
ps_letter_cases = zeros(size(rbms_pass,1)/2,size(rbms_pass,2));
letter_t_l = zeros(size(rbms_pass,1)/2,size(target_l,2));
letter_t_s = zeros(size(rbms_pass,1)/2,size(target_s,2));
ps_letter_t_l = zeros(size(rbms_pass,1)/2,size(target_l,2));
ps_letter_t_s = zeros(size(rbms_pass,1)/2,size(target_s,2));

j1 = 1;j2 = 1;
for i=1:size(rbms_pass,1)
    idx_l = find(target_l(i,:));
    if idx_l == 1 || idx_l == 2 || idx_l == 3 || idx_l == 4 ||idx_l == 5 || idx_l == 6
        letter_cases(j1,:) = rbms_pass(i,:);
        letter_t_s(j1,:) = target_s(i,:);
        letter_t_l(j1,:) = target_l(i,:);
        j1 = j1 +1;
    else
        ps_letter_cases(j2,:) = rbms_pass(i,:);
        ps_letter_t_s(j2,:) = target_s(i,:);
        ps_letter_t_l(j2,:) = target_l(i,:);
        j2 = j2 +1 ; 
    end
end

%%%%%%%%%%% COMPUTE FOR LETTERS
% divide data to congruent and incongruent for letters
cong_cases = zeros(size(letter_cases,1)/2,size(letter_cases,2));
incong_cases = zeros(size(letter_cases,1)/2,size(letter_cases,2));
cong_ts = zeros(size(letter_cases,1)/2,size(letter_t_s,2));
cong_tl= zeros(size(letter_cases,1)/2,size(letter_t_l,2));
incong_ts= zeros(size(letter_cases,1)/2,size(letter_t_s,2));
incong_tl= zeros(size(letter_cases,1)/2,size(letter_t_l,2));
j1 = 1;j2 = 1;
for i=1:size(letter_cases,1)
    idx_l = find(letter_t_l(i,:));
    idx_s = find(letter_t_s(i,:));
    if idx_l == 1 && idx_s == 6 || idx_l == 2 && idx_s == 4 || idx_l == 3 && idx_s == 5 || ...
            idx_l == 4 && idx_s == 1 || idx_l == 5 && idx_s == 2 || idx_l == 6 && idx_s == 3 || ...
                idx_l == 7 && idx_s == 6 || idx_l == 8 && idx_s == 4 || idx_l == 9 && idx_s == 5 || ...
                    idx_l == 10 && idx_s == 1 || idx_l == 11 && idx_s == 2 || idx_l == 12 && idx_s == 3 
        cong_cases(j1,:) = letter_cases(i,:);
        cong_ts(j1,:) = letter_t_s(i,:);
        cong_tl(j1,:) = letter_t_l(i,:);
        j1 = j1 +1;
    else
        incong_cases(j2,:) = letter_cases(i,:);
        incong_ts(j2,:) = letter_t_s(i,:);
        incong_tl(j2,:) = letter_t_l(i,:);
        j2 = j2 +1 ; 
    end
end
% General Predictions for letter
pred = letter_cases*weights;
[~, max_act] = max(pred,[],2); 
[r1,~] = find(letter_t_l'); 
[r2,~] = find(letter_t_s');
acc_l = (max_act == r1);
acc_s = (max_act == r2);
accuracy_l = mean(acc_l);
accuracy_s = mean(acc_s);
fprintf(1,'\n ------- LETTERS ------- \n');
fprintf(1,'\n All cases -- on geo-shape \n Accuracy= %d ',accuracy_s);
fprintf(1,'\n If decision based on letter? \n Accuracy= %d\n ',accuracy_l);
Combined(1) = accuracy_s;
Combined_l(1) = accuracy_l;
% Letter + Congruent
pred = cong_cases*weights;
[~, max_act] = max(pred,[],2); 
[r1,~] = find(cong_tl'); 
[r2,~] = find(cong_ts');
acc_l = (max_act == r1);
acc_s = (max_act == r2);
accuracy_l_c = mean(acc_l);
accuracy_s_c = mean(acc_s);
fprintf(1,'\n Congruent Cases -- on geo-shape \n Accuracy= %d ',accuracy_s_c);
fprintf(1,'\n If decision based on letter? \n Accuracy= %d\n ',accuracy_l_c);
Congruent_Case(1) = accuracy_s_c;
Congruent_Case_l(1) = accuracy_l_c;
% Letter + Incongruent
pred = incong_cases*weights;
[~, max_act] = max(pred,[],2); 
[r1,~] = find(incong_tl'); 
[r2,~] = find(incong_ts');
acc_l = (max_act == r1);
acc_s = (max_act == r2);
accuracy_l = mean(acc_l);
accuracy_s = mean(acc_s);
fprintf(1,'\n Incongruent Cases -- on geo-shape \n Accuracy= %d ',accuracy_s);
fprintf(1,'\n If decision based on letter? \n Accuracy= %d\n ',accuracy_l);
Incongruent_Case(1) = accuracy_s;
Incongruent_Case_l(1) = accuracy_l;
diff_ce = accuracy_s_c*100 - accuracy_s * 100;
fprintf(1,'\n\n Congruency effect for letters of magnitude: %d\n ',diff_ce);
CE(1) = diff_ce;
CE_l(1) = accuracy_l_c *100 - accuracy_l*100;

%%%%%%%%%%% COMPUTE FOR PSEUDO LETTERS
% divide data to congruent and incongruent for pseudoletters
cong_cases = zeros(size(letter_cases,1)/2,size(letter_cases,2));
incong_cases = zeros(size(letter_cases,1)/2,size(letter_cases,2));
cong_ts = zeros(size(letter_cases,1)/2,size(letter_t_s,2));
cong_tl= zeros(size(letter_cases,1)/2,size(letter_t_l,2));
incong_ts= zeros(size(letter_cases,1)/2,size(letter_t_s,2));
incong_tl= zeros(size(letter_cases,1)/2,size(letter_t_l,2));
j1 = 1;j2 = 1;
for i=1:size(ps_letter_cases,1)
    idx_l = find(ps_letter_t_l(i,:));
    idx_s = find(ps_letter_t_s(i,:));
    if idx_l == 1 && idx_s == 6 || idx_l == 2 && idx_s == 4 || idx_l == 3 && idx_s == 5 || ...
            idx_l == 4 && idx_s == 1 || idx_l == 5 && idx_s == 2 || idx_l == 6 && idx_s == 3 || ...
                idx_l == 7 && idx_s == 6 || idx_l == 8 && idx_s == 4 || idx_l == 9 && idx_s == 5 || ...
                    idx_l == 10 && idx_s == 1 || idx_l == 11 && idx_s == 2 || idx_l == 12 && idx_s == 3 
        cong_cases(j1,:) = ps_letter_cases(i,:);
        cong_ts(j1,:) = ps_letter_t_s(i,:);
        cong_tl(j1,:) = ps_letter_t_l(i,:);
        j1 = j1 +1;
    else
        incong_cases(j2,:) = ps_letter_cases(i,:);
        incong_ts(j2,:) = ps_letter_t_s(i,:);
        incong_tl(j2,:) = ps_letter_t_l(i,:);
        j2 = j2 +1 ; 
    end
end
% General Predictions for pseudoletters
pred = ps_letter_cases*weights;
[~, max_act] = max(pred,[],2); 
[r1,~] = find(ps_letter_t_l'); 
[r2,~] = find(ps_letter_t_s');
acc_l = (max_act == r1);
acc_s = (max_act == r2);
accuracy_l = mean(acc_l);
accuracy_s = mean(acc_s);
fprintf(1,'\n ------- PSEUDOLETTERS ------- \n ');
fprintf(1,'\n All cases -- on geo-shape \n Accuracy= %d ',accuracy_s);
Combined(2) = accuracy_s;
fprintf(1,'\n If decision based on pseudoletter? \n Accuracy= %d\n ',accuracy_l);
Combined_l(2) = accuracy_l;
% pseudoletter + Congruent
pred = cong_cases*weights;
[~, max_act] = max(pred,[],2); 
[r1,~] = find(cong_tl'); 
[r2,~] = find(cong_ts');
acc_l = (max_act == r1);
acc_s = (max_act == r2);
accuracy_l_c = mean(acc_l);
accuracy_s_c = mean(acc_s);
fprintf(1,'\n Congruent Cases -- on geo-shape \n Accuracy= %d ',accuracy_s_c);
Congruent_Case(2) = accuracy_s_c;
fprintf(1,'\n If decision based on pseudoletter? \n Accuracy= %d\n ',accuracy_l_c);
Congruent_Case_l(2) = accuracy_l_c;
% pseudoletter + Incongruent
pred = incong_cases*weights;
[~, max_act] = max(pred,[],2); 
[r1,~] = find(incong_tl'); 
[r2,~] = find(incong_ts');
acc_l = (max_act == r1);
acc_s = (max_act == r2);
accuracy_l = mean(acc_l);
accuracy_s = mean(acc_s);
fprintf(1,'\n Incongruent Cases -- on geo-shape \n Accuracy= %d ',accuracy_s);
Incongruent_Case(2) = accuracy_s;
Incongruent_Case_l(2) = accuracy_l;
fprintf(1,'\n If decision based on pseudoletter? \n Accuracy= %d\n ',accuracy_l);
diff_ce = accuracy_s_c*100 - accuracy_s * 100;
fprintf(1,'\n\n Congruency effect for pseudoletters of magnitude: %d\n ',diff_ce);
CE(2) = diff_ce;
CE_l(2) = accuracy_l_c*100 - accuracy_l;
Accuracy_Measurements_shape = table(Targets,Congruent_Case,Incongruent_Case,Combined);
Accuracy_Measurements_letter = table(Targets,Congruent_Case_l,Incongruent_Case_l,Combined_l);
CE_Measurements = table(Targets,CE);
CE_letter = table(Targets,CE_l);

filename = "plots_results/CE_pred" + "_" + clean_date + "_" + int2str(c(4)) + "h" + int2str(c(5))+"m";
save(filename,'Class_table','Accuracy_Measurements_shape','Accuracy_Measurements_letter' ...
    ,'CE_Measurements', 'CE_letter','final_epoch');



