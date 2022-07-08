% Computing performance -- identification of letters based on their
% geometrical couter part -- congruent shape

dd = strsplit(date,'-');c = clock;clean_date = strcat(dd(1),dd(2)); %without "-YYYY"

if numhid3 == 0
    strh3 ="";
else
    strh3 = "three/";
end
% load dataset for the 6 shapes

%load openCV_final_Shapes.mat data target_s %target_l is just their s-target
% load openCV_Shapes_xtra.mat data target_s
load openCV_shapePOS3.mat data target_s

s_data = zeros(size(data));
for i=1:size(data,1)
    s_data(i,:) = reshape(im2double(reshape(data(i,:),[40 40 1])), [1 1600]);
end
s_target = double(target_s);
clear data; clear target_s

% load dataset for the 6 letters
%load openCV_final_letters.mat data target_s target_l;
%load openCV_letters_xtra.mat data target_s target_l;


%load openCV_letters_tF.mat data target_s target_l;
load openCV_letterPOS1.mat data target_s target_l;


d_double = zeros(size(data));
for i=1:size(data,1)
    d_double(i,:) = reshape(im2double(reshape(data(i,:),[40 40 1])), [1 1600]);
end
clear data; data = d_double;
target_s= double(target_s);
target_l_save = target_l;
%% Prep Computation

% pass shapedata throught RBM^s:
hid_out_1_s = 1./(1 + exp(-s_data*vishid_1 - repmat(hidbiases_1,size(s_data,1),1)));
rbms_pass_s = 1./(1 + exp(-hid_out_1_s*vishid_2 - repmat(hidbiases_2,size(hid_out_1_s,1),1)));
if numhid3 ~= 0
    rbms_pass_s = 1./(1 + exp(-rbms_pass_s*vishid_3 - repmat(hidbiases_3,size(rbms_pass_s,1),1)));
end
% pass letterdata throught RBMs:
hid_out_1_l = 1./(1 + exp(-data*vishid_1 - repmat(hidbiases_1,size(data,1),1)));
rbms_pass_l = 1./(1 + exp(-hid_out_1_l*vishid_2 - repmat(hidbiases_2,size(hid_out_1_l,1),1)));
if numhid3 ~= 0
    rbms_pass_l = 1./(1 + exp(-rbms_pass_l*vishid_3 - repmat(hidbiases_3,size(rbms_pass_l,1),1)));
end

%% General Assesment
%%%%%
pred1 = rbms_pass_l*weights;

%pred1 = net2(rbms_pass_l',);




softmax_pred1 = softmax(dlarray(pred1','CB'));
pred1 = extractdata(softmax_pred1)';
[~, max_act_l] = max(pred1,[],2);
[r1,~] = find(target_s');
acc_l = (max_act_l == r1);
accuracy_l = mean(acc_l);
%loss?
l_loss = extractdata(crossentropy(softmax_pred1,target_s'));

%%%%%
pred2 = rbms_pass_s*weights;
softmax_pred2 = softmax(dlarray(pred2','CB'));
pred2 = extractdata(softmax_pred2)';
[~, max_act_s] = max(pred2,[],2);
[r2,~] = find(s_target'); 
acmax_cs = (max_act_s == r2);
accuracy_s = mean(acmax_cs);
%loss?
s_loss = extractdata(crossentropy(softmax_pred2,s_target'));

fprintf(1,'\n Identification of main geometrical shapes');
fprintf(1,'\n Accuracy = %d ',accuracy_s);
fprintf(1,'\n Identification of Letter based on their respetive geometrical shape');
fprintf(1,'\n Accuracy = %d ',accuracy_l);

%%%% Extended ASSESMENT
%% Letters
%(assement of identification based on their "congruent"shape)
max_a = [];max_h = [];max_m = [];max_u = [];max_t = [];max_x = [];
pred_a = [];pred_h = [];pred_m = [];pred_u = [];pred_t = [];pred_x = [];
for i=1:size(max_act_l,1)
    idx_l = find(target_l(i,:));
    if idx_l == 1 
        max_a = [max_a;max_act_l(i)];
        pred_a = [pred_a;pred1(i,:)];
    elseif idx_l == 2
        max_h= [max_h;max_act_l(i)];
        pred_h = [pred_h;pred1(i,:)];
    elseif idx_l == 3
        max_m= [max_m;max_act_l(i)];
        pred_m = [pred_m;pred1(i,:)];
    elseif idx_l == 4
        max_t= [max_t;max_act_l(i)];
        pred_t = [pred_t;pred1(i,:)];
    elseif idx_l == 5
        max_u= [max_u;max_act_l(i)];
        pred_u = [pred_u;pred1(i,:)];
    elseif idx_l == 6
        max_x= [max_x;max_act_l(i)];
        pred_x = [pred_x;pred1(i,:)];
    end
end

mode_A = mode(max_a);std_A = std(max_a);prD_A =  mean(pred_a,1);
mode_H = mode(max_h);std_H = std(max_h);prD_H =  mean(pred_h,1);
mode_M = mode(max_m);std_M = std(max_m);prD_M =  mean(pred_m,1);
mode_U = mode(max_u);std_U = std(max_u);prD_U =  mean(pred_u,1);
mode_T = mode(max_t);std_T = std(max_t);prD_T =  mean(pred_t,1);
mode_X = mode(max_x);std_X = std(max_x);prD_X =  mean(pred_x,1);
r1 = ones(size(rbms_pass_l,1)/6,1);r2 = ones(size(rbms_pass_l,1)/6,1)*2;r3 = ones(size(rbms_pass_l,1)/6,1)*3;
r4 = ones(size(rbms_pass_l,1)/6,1)*4;r5 = ones(size(rbms_pass_l,1)/6,1)*5;r6 = ones(size(rbms_pass_l,1)/6,1)*6;
acc1 = (max_a == r6);acc2 = (max_h == r4);acc3 = (max_m == r5);
acc4 = (max_u == r2);acc5 = (max_t == r1);acc6 = (max_x == r3);
acc1 = mean(acc1);acc2 = mean(acc2);acc3 = mean(acc3);
acc4 = mean(acc4);acc5 = mean(acc5);acc6 = mean(acc6);
% create Table for Letters:
Targets = ["A" ; "H"; "M"; "U"; "T"; "X"];
Mode = [mode_A;mode_H;mode_M;mode_U;mode_T;mode_X];
Std = [std_A;std_H;std_M;std_U;std_T;std_X];
Acc = [acc1;acc2;acc3;acc4;acc5;acc6];
table_letter = table(Targets,Mode,Std,Acc);
letter_pdr = [prD_A;prD_H;prD_M;prD_U;prD_T;prD_X];

clear data;

%% PSEUDO-LETTERS
% import data   
%load openCV_pletters_xtra.mat data target_l target_s

%load openCV_final_pletters.mat data target_l target_s
load openCV_pletters_tF.mat data target_s target_l;


d_double = zeros(size(data));
for i=1:size(data,1)
    d_double(i,:) = reshape(im2double(reshape(data(i,:),[40 40 1])), [1 1600]);
end
clear data; data = d_double;
target_ps = double(target_s);
% pass letterdata throught RBMs:
hid_out_1_pl = 1./(1 + exp(-data*vishid_1 - repmat(hidbiases_1,size(data,1),1)));
rbms_pass_pl = 1./(1 + exp(-hid_out_1_pl*vishid_2 - repmat(hidbiases_2,size(hid_out_1_pl,1),1)));
if numhid3 ~= 0
    rbms_pass_pl = 1./(1 + exp(-rbms_pass_pl*vishid_3 - repmat(hidbiases_3,size(rbms_pass_pl,1),1)));
end
% add biases
ONES = ones(size(rbms_pass_pl, 1), 1);  
rbms_pass_pl = [rbms_pass_pl ONES];

%%%%%
pred3 = rbms_pass_pl*weights;
softmax_pred3 = softmax(dlarray(pred3','CB'));
pred3 = extractdata(softmax_pred3)';
[~, max_act_pl] = max(pred3,[],2);
[r3,~] = find(target_ps');
acc_pl = (max_act_pl == r3);
accuracy_pl = mean(acc_pl);
%loss?
l_loss = extractdata(crossentropy(softmax_pred3,target_ps'));
fprintf(1,'\n Identification of PSEUDO-Letter based on their respetive geometrical shape');
fprintf(1,'\n Accuracy = %d ',accuracy_pl);

%%%% EXTENDED ASSESMENT
max_pa = [];max_ph = [];max_pm = [];max_pu = [];max_pt = [];max_px = [];
pred_pa = [];pred_ph = [];pred_pm = [];pred_pu = [];pred_pt = [];pred_px = [];

for i=1:size(max_act_pl,1)
    idx_l = find(target_l(i,:));
    if idx_l == 1 
        max_pa = [max_pa;max_act_pl(i)];
        pred_pa = [pred_pa;pred3(i,:)];
    elseif idx_l == 2
        max_ph= [max_ph;max_act_pl(i)];
        pred_ph = [pred_ph;pred3(i,:)];
    elseif idx_l == 3
        max_pm= [max_pm;max_act_pl(i)];
        pred_pm = [pred_pm;pred3(i,:)];
    elseif idx_l == 4
        max_pt= [max_pt;max_act_pl(i)];
        pred_pt = [pred_pt;pred3(i,:)];
    elseif idx_l == 5
        max_pu= [max_pu;max_act_pl(i)];
        pred_pu = [pred_pu;pred3(i,:)];
    elseif idx_l == 6
        max_px= [max_px;max_act_pl(i)];
        pred_px = [pred_px;pred3(i,:)];
    end
end
mode_pA = mode(max_pa);std_pA = std(max_pa);prD_pA =  mean(pred_pa,1);
mode_pH = mode(max_ph);std_pH = std(max_ph);prD_pH =  mean(pred_ph,1);
mode_pM = mode(max_pm);std_pM = std(max_pm);prD_pM =  mean(pred_pm,1);
mode_pU = mode(max_pu);std_pU = std(max_pu);prD_pU =  mean(pred_pu,1);
mode_pT = mode(max_pt);std_pT = std(max_pt);prD_pT =  mean(pred_pt,1);
mode_pX = mode(max_px);std_pX = std(max_px);prD_pX =  mean(pred_px,1);
r1 = ones(size(rbms_pass_pl,1)/6,1);r2 = ones(size(rbms_pass_pl,1)/6,1)*2;r3 = ones(size(rbms_pass_pl,1)/6,1)*3;
r4 = ones(size(rbms_pass_pl,1)/6,1)*4;r5 = ones(size(rbms_pass_pl,1)/6,1)*5;r6 = ones(size(rbms_pass_pl,1)/6,1)*6;
acc1 = (max_pa == r6);acc2 = (max_ph == r4);acc3 = (max_pm == r5);
acc4 = (max_pu == r2);acc5 = (max_pt == r1);acc6 = (max_px == r3);
acc1 = mean(acc1);acc2 = mean(acc2);acc3 = mean(acc3);
acc4 = mean(acc4);acc5 = mean(acc5);acc6 = mean(acc6);
% create Table for Letters:
pTargets = ["pA" ; "pH"; "pM"; "pU"; "pT"; "pX"];
pMode = [mode_pA;mode_pH;mode_pM;mode_pU;mode_pT;mode_pX];
pStd = [std_pA;std_pH;std_pM;std_pU;std_pT;std_pX];
Acc = [acc1;acc2;acc3;acc4;acc5;acc6];
table_pletter = table(pTargets,pMode,pStd,Acc);
pletter_pdr = [prD_pA;prD_pH;prD_pM;prD_pU;prD_pT;prD_pX];

%save the results/data
% filename = "plots_results/Id_basedOnGeoS6/" + clean_date + "_" + int2str(c(4)) + "h" + int2str(c(5))+"m_" + "Eval_L&pL";
% save(filename,'table_letter','table_pletter','letter_pdr','pletter_pdr','accuracy_s','accuracy_pl', 'accuracy_l','final_epoch');

Id_BasedOnGeoS.table_letter = table_letter;
Id_BasedOnGeoS.table_pletter = table_pletter;
Id_BasedOnGeoS.letter_pdr = letter_pdr;
Id_BasedOnGeoS.pletter_pdr = pletter_pdr;
Id_BasedOnGeoS.accuracy_s = accuracy_s;
Id_BasedOnGeoS.accuracy_pl = accuracy_pl;
Id_BasedOnGeoS.accuracy_l = accuracy_l;

%% Save Matrix for statistical analysis

if ii == 1
    subj_str = "Subject 1";
elseif ii == 2
    subj_str = "Subject 2 ";
elseif ii == 3
    subj_str = "Subject 3 ";
elseif ii == 4
    subj_str = "Subject 4 ";
elseif ii == 5
    subj_str = "Subject 5 ";
end
Accuracy = cat(1,acc_l,acc_pl);
str_target_l = letter_int2str(target_l_save);
str_target_pl = psletter_int2str(target_l);
Targets = cat(1,str_target_l,str_target_pl);
Subjects = repmat(subj_str,size(Accuracy,1),1);

space_holder = repmat("-----",size(Accuracy,1),1);
% letter   -- pred1
% psletter -- pred3
cross = cat(1,pred1(:,1),pred3(:,1));
elipse = cat(1,pred1(:,2),pred3(:,2));
hexa = cat(1,pred1(:,3),pred3(:,3));
rectangle = cat(1,pred1(:,4),pred3(:,4));
square = cat(1,pred1(:,5),pred3(:,5));
triangle = cat(1,pred1(:,6),pred3(:,6));

if ii == 1 
    matrix_2 = table(Subjects,Targets,Accuracy,space_holder,cross,elipse,hexa,rectangle, ...
        square,triangle);
else
    m = table(Subjects,Targets,Accuracy,space_holder,cross,elipse,hexa,rectangle, ...
        square,triangle);
    matrix_2 = cat(1,matrix_2,m);
end



function [str_letter] = letter_int2str(letter_id)
    str_letter = strings(size(letter_id,1),1);
    for i=1:size(letter_id,1)
        if find(letter_id(i,:))  == 1
            str_letter(i) = "A";
        elseif find(letter_id(i,:))  == 2
            str_letter(i) = "H";
        elseif find(letter_id(i,:))  == 3
            str_letter(i) = "M";
        elseif find(letter_id(i,:))  == 4 
            str_letter(i) = "T";
        elseif find(letter_id(i,:))  == 5
            str_letter(i) = "U";
        elseif find(letter_id(i,:))  == 6
            str_letter(i) = "X";
        end
    end
end

function [str_letter] = psletter_int2str(letter_id)
    str_letter = strings(size(letter_id,1),1);
    for i=1:size(letter_id,1)
        if find(letter_id(i,:))  == 1
            str_letter(i) = "psA";
        elseif find(letter_id(i,:))  == 2
            str_letter(i) = "psH";
        elseif find(letter_id(i,:))  == 3
            str_letter(i) = "psM";
        elseif find(letter_id(i,:))  == 4 
            str_letter(i) = "psT";
        elseif find(letter_id(i,:))  == 5
            str_letter(i) = "psU";
        elseif find(letter_id(i,:))  == 6
            str_letter(i) = "psX";
        end
    end
end



