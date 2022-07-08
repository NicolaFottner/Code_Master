% compute the Congruency effect and store result in a struct to be further
% store in a mat file for that specific simulation

load openCV_CE_data.mat cong_pl_d cong_pl_t cong_pl_s inc_pl_d inc_pl_t inc_pl_s 
load openCV_newL_CE.mat cong_l_d cong_l_t cong_l_s inc_l_d inc_l_t inc_l_s 

%% Prep and convert openCV_CE_data from uint8 to double:
shapedata = zeros(size(cong_l_d));
for i=1:size(cong_l_d,1)
    shapedata(i,:) = reshape(im2double(reshape(cong_l_d(i,:),[40 40 1])), [1 1600]);
end
cong_l_d = shapedata;
cong_l_t =  double(cong_l_t);
cong_l_s = double(cong_l_s);
%%%
shapedata = zeros(size(cong_pl_d));
for i=1:size(cong_pl_d,1)
    shapedata(i,:) = reshape(im2double(reshape(cong_pl_d(i,:),[40 40 1])), [1 1600]);
end
cong_pl_d = shapedata;
cong_pl_t =  double(cong_pl_t);
cong_pl_s = double(cong_pl_s);
%%%
shapedata = zeros(size(inc_l_d));
for i=1:size(inc_l_d,1)
    shapedata(i,:) = reshape(im2double(reshape(inc_l_d(i,:),[40 40 1])), [1 1600]);
end
inc_l_d = shapedata;
inc_l_t =  double(inc_l_t);
inc_l_s = double(inc_l_s);
%%%
shapedata = zeros(size(inc_pl_d));
for i=1:size(inc_pl_d,1)
    shapedata(i,:) = reshape(im2double(reshape(inc_pl_d(i,:),[40 40 1])), [1 1600]);
end
inc_pl_d = shapedata;
inc_pl_t =  double(inc_pl_t);
inc_pl_s = double(inc_pl_s);

%load rbm data
load t_model DN
vishid_1 = DN.L{1,1}.vishid;
hidbiases_1 = DN.L{1,1}.hidbiases;
clear DN
load g_rbm_2.mat vishid_2 hidbiases_2
%load rbm2_16J11h39.mat vishid_2 hidbiases_2

%%% Create variables for Table -- visualization purpose
% table with accuracy values for shape
Shape_decisionAcc = zeros(4,1); 
Letter_decisionAcc = zeros(4,1);

%% EVAL - Letter congruent
% pass data throught RBMs:
hid_out_1_d = 1./(1 + exp(-cong_l_d*vishid_1 - repmat(hidbiases_1,size(cong_l_d,1),1)));
rbms_pass = 1./(1 + exp(-hid_out_1_d*vishid_2 - repmat(hidbiases_2,size(hid_out_1_d,1),1)));
% Compute prediction:

pred = net2(rbms_pass');
pred = pred';
% [a,b] =max(pred,[],2); --- % a has also  values  > 1 ---- 
% i.e. a is unnormalized;
[~, max_act] = max(pred,[],2); % max_act are indices of dim2 in pred of the highest value
[r1,~] = find(cong_l_t'); % find which columns (rows in transpose) are 1
[r2,~] = find(cong_l_s'); % find which columns (rows in transpose) are 1
acc_l = (max_act == r1);
acc_s = (max_act == r2);
accuracy_l = mean(acc_l);
accuracy_s = mean(acc_s);
fprintf(1,'\n Prediction / Assesment of the Congruency Effect\n');
fprintf(1,'\n ------- Case : Congruent ------- \n');
fprintf(1,'\n ------- Target : Letter ------- \n');
fprintf(1,'\n If Decision based on letter? \n Accuracy = %d\n ',accuracy_l);
fprintf(1,'\n If Decision based on shape? \n Accuracy = %d\n ',accuracy_s);
Shape_decisionAcc(1) = accuracy_s;
Letter_decisionAcc(1) = accuracy_l;

%%%% Extended ASSESMENT
max_cr = [];max_el = [];max_he = [];max_re = [];max_sq = [];max_tr = [];
pred_cr = [];pred_el= [];pred_he = [];pred_re = [];pred_sq = [];pred_tr = [];
for i=1:size(max_act,1)
    idx_s = find(cong_l_s(i,:));
    if idx_s == 1 
        max_cr = [max_cr;max_act(i)];
        pred_cr = [pred_cr;pred(i,:)];
    elseif idx_s == 2
        max_el= [max_el;max_act(i)];
        pred_el = [pred_el;pred(i,:)];
    elseif idx_s == 3
        max_he= [max_he;max_act(i)];
        pred_he = [pred_he;pred(i,:)];
    elseif idx_s == 4
        max_re= [max_re;max_act(i)];
        pred_re = [pred_re;pred(i,:)];
    elseif idx_s == 5
        max_sq= [max_sq;max_act(i)];
        pred_sq = [pred_sq;pred(i,:)];
    elseif idx_s == 6
        max_tr= [max_tr;max_act(i)];
        pred_tr = [pred_tr;pred(i,:)];
    end
end
mode_cr = mode(max_cr);std_cr = std(max_cr);prD_cr =  mean(pred_cr,1);
mode_el = mode(max_el);std_el = std(max_el);prD_el =  mean(pred_el,1);
mode_he = mode(max_he);std_he = std(max_he);prD_he =  mean(pred_he,1);
mode_re = mode(max_re);std_re = std(max_re);prD_re =  mean(pred_re,1);
mode_sq = mode(max_sq);std_sq = std(max_sq);prD_sq =  mean(pred_sq,1);
mode_tr = mode(max_tr);std_tr = std(max_tr);prD_tr =  mean(pred_tr,1);
r1 = ones(size(rbms_pass,1)/6,1);r2 = ones(size(rbms_pass,1)/6,1)*2;r3 = ones(size(rbms_pass,1)/6,1)*3;
r4 = ones(size(rbms_pass,1)/6,1)*4;r5 = ones(size(rbms_pass,1)/6,1)*5;r6 = ones(size(rbms_pass,1)/6,1)*6;
acc1 = (max_cr == r1);% cross, ---- T
acc2 = (max_el == r2);% elipse ---- U
acc3 = (max_he == r3);% hexagon ---- X
acc4 = (max_re == r4);% rectangle ---- H
acc5 = (max_sq == r5); % square ---- M
acc6 = (max_tr == r6);% triangle ---- A
acc1 = mean(acc1);acc2 = mean(acc2);acc3 = mean(acc3);
acc4 = mean(acc4);acc5 = mean(acc5);acc6 = mean(acc6);
% create Table for Letters: (with old order:)
% Targets = ["Cross/T" ; "Elipse/U"; "Hexa/X"; "Rect/H"; "Square/M"; "Tria/A"];
% Acc = [acc1;acc2;acc3;acc4;acc5;acc6];
% Mode = [mode_cr;mode_el;mode_he;mode_re;mode_sq;mode_tr];
% Std = [std_cr;std_el;std_he;std_re;std_sq;std_tr];
%%% better order:
Targets = ["A/Tria";"H/Rect";"M/Squ";"U/Elips";"T/Cross";"X/Hexa"];
Acc = [acc6;acc4;acc5;acc2;acc1;acc3];
Mode = [mode_tr;mode_re;mode_sq;mode_el;mode_cr;mode_he];
Std = [mode_tr;mode_re;mode_sq;mode_el;mode_cr;mode_he];
cl_ce_details = table(Targets,Mode,Std,Acc);
cl_ce_pdr = [prD_tr;prD_re;prD_sq;prD_el;prD_cr;prD_he];

%% EVAL - Pseudo Letter congruent
% pass data throught RBMs:
hid_out_1_d = 1./(1 + exp(-cong_pl_d*vishid_1 - repmat(hidbiases_1,size(cong_pl_d,1),1)));
rbms_pass = 1./(1 + exp(-hid_out_1_d*vishid_2 - repmat(hidbiases_2,size(hid_out_1_d,1),1)));
% Compute prediction:
pred = net2(rbms_pass');
pred = pred';
% [a,b] =max(pred,[],2); --- % a has also  values  > 1 ---- 
% i.e. a is unnormalized
[~, max_act] = max(pred,[],2); % max_act are indices of dim2 in pred of the highest value
[r1,~] = find(cong_pl_t'); % find which columns (rows in transpose) are 1
[r2,~] = find(cong_pl_s'); % find which columns (rows in transpose) are 1
acc_l = (max_act == r1);
acc_s = (max_act == r2);
accuracy_l = mean(acc_l);
accuracy_s = mean(acc_s);
fprintf(1,'\n Prediction / Assesment of the Congruency Effect\n');
fprintf(1,'\n ------- Case : Congruent ------- \n');
fprintf(1,'\n ------- Target : Pseudo Letter ------- \n');
fprintf(1,'\n If Decision based on letter? \n Accuracy = %d\n ',accuracy_l);
fprintf(1,'\n If Decision based on shape? \n Accuracy = %d\n ',accuracy_s);

Shape_decisionAcc(2) = accuracy_s;
Letter_decisionAcc(2) = accuracy_l;

%%%% Extended ASSESMENT
max_cr = [];max_el = [];max_he = [];max_re = [];max_sq = [];max_tr = [];
pred_cr = [];pred_el = [];pred_he = [];pred_re = [];pred_sq = [];pred_tr = [];
for i=1:size(max_act,1)
    idx_s = find(cong_pl_s(i,:));
    if idx_s == 1 
        max_cr = [max_cr;max_act(i)];
        pred_cr = [pred_cr;pred(i,:)];
    elseif idx_s == 2
        max_el= [max_el;max_act(i)];
        pred_el = [pred_el;pred(i,:)];
    elseif idx_s == 3
        max_he= [max_he;max_act(i)];
        pred_he = [pred_he;pred(i,:)];
    elseif idx_s == 4
        max_re= [max_re;max_act(i)];
        pred_re = [pred_re;pred(i,:)];
    elseif idx_s == 5
        max_sq= [max_sq;max_act(i)];
        pred_sq = [pred_sq;pred(i,:)];
    elseif idx_s == 6
        max_tr= [max_tr;max_act(i)];
        pred_tr = [pred_tr;pred(i,:)];
    end
end
mode_cr = mode(max_cr);std_cr = std(max_cr);prD_cr =  mean(pred_cr,1);
mode_el = mode(max_el);std_el = std(max_el);prD_el =  mean(pred_el,1);
mode_he = mode(max_he);std_he = std(max_he);prD_he =  mean(pred_he,1);
mode_re = mode(max_re);std_re = std(max_re);prD_re =  mean(pred_re,1);
mode_sq = mode(max_sq);std_sq = std(max_sq);prD_sq =  mean(pred_sq,1);
mode_tr = mode(max_tr);std_tr = std(max_tr);prD_tr =  mean(pred_tr,1);
r1 = ones(size(rbms_pass,1)/6,1);r2 = ones(size(rbms_pass,1)/6,1)*2;r3 = ones(size(rbms_pass,1)/6,1)*3;
r4 = ones(size(rbms_pass,1)/6,1)*4;r5 = ones(size(rbms_pass,1)/6,1)*5;r6 = ones(size(rbms_pass,1)/6,1)*6;
acc1 = (max_cr == r1);% cross, ---- T
acc2 = (max_el == r2);% elipse ---- U
acc3 = (max_he == r3);% hexagon ---- X
acc4 = (max_re == r4);% rectangle ---- H
acc5 = (max_sq == r5); % square ---- M
acc6 = (max_tr == r6);% triangle ---- A
acc1 = mean(acc1);acc2 = mean(acc2);acc3 = mean(acc3);
acc4 = mean(acc4);acc5 = mean(acc5);acc6 = mean(acc6);
% create Table for Letters:
%Targets = ["pA" ; "pH"; "pM"; "pU"; "pT"; "pX"];
Targets = ["pA/Tria";"pH/Rect";"pM/Squ";"pU/Elips";"pT/Cross";"pX/Hexa"];
Acc = [acc6;acc4;acc5;acc2;acc1;acc3];
Mode = [mode_tr;mode_re;mode_sq;mode_el;mode_cr;mode_he];
Std = [mode_tr;mode_re;mode_sq;mode_el;mode_cr;mode_he];
cpl_ce_details = table(Targets,Mode,Std,Acc);
cpl_ce_pdr = [prD_tr;prD_re;prD_sq;prD_el;prD_cr;prD_he];


%% EVAL - Letter Incongruent
% pass data throught RBMs:
hid_out_1_d = 1./(1 + exp(-inc_l_d*vishid_1 - repmat(hidbiases_1,size(inc_l_d,1),1)));
rbms_pass = 1./(1 + exp(-hid_out_1_d*vishid_2 - repmat(hidbiases_2,size(hid_out_1_d,1),1)));
pred = net2(rbms_pass');
pred = pred';
% [a,b] =max(pred,[],2); --- % a has also  values  > 1 ---- 
% i.e. a is unnormalized
[~, max_act] = max(pred,[],2); % max_act are indices of dim2 in pred of the highest value
[r1,~] = find(inc_l_t'); % find which columns (rows in transpose) are 1
[r2,~] = find(inc_l_s'); % find which columns (rows in transpose) are 1
acc_l = (max_act == r1);
acc_s = (max_act == r2);
accuracy_l = mean(acc_l);
accuracy_s = mean(acc_s);
fprintf(1,'\n Prediction / Assesment of the Congruency Effect\n');
fprintf(1,'\n ------- Case : Incongruent ------- \n');
fprintf(1,'\n ------- Target : Letter ------- \n');
fprintf(1,'\n If Decision based on letter? \n Accuracy = %d\n ',accuracy_l);
fprintf(1,'\n If Decision based on shape? \n Accuracy = %d\n ',accuracy_s);

Shape_decisionAcc(3) = accuracy_s;
Letter_decisionAcc(3) = accuracy_l;

%%%% Extended ASSESMENT
max_cr = [];max_el = [];max_he = [];max_re = [];max_sq = [];max_tr = [];
pred_cr = [];pred_el = [];pred_he = [];pred_re = [];pred_sq = [];pred_tr = [];
% for more details, for each incongruent case
max_cr_m = [];max_cr_x=[];max_el_h = [];max_el_a = [];max_he_m = [];max_he_t=[];
max_re_a = [];max_re_u=[];max_sq_t = [];max_sq_x=[];max_tr_h= [];max_tr_u = [];
pred_cr_m = [];pred_cr_x = [];pred_el_a= [];pred_el_h=[];pred_he_m = [];pred_he_t=[];
pred_re_a = [];pred_re_u = [];pred_sq_t = [];pred_sq_x=[];pred_tr_h = [];pred_tr_u=[];
msgtext = 'Invalid Incongruent Data/case indexing';
for i=1:size(max_act,1)
    idx_s = find(inc_l_s(i,:));
    idx_l = find(inc_l_t(i,:));
    if idx_s == 1 
        if idx_l == 3
            max_cr_m = [max_cr_m;max_act(i)];
            pred_cr_m = [pred_cr_m;pred(i,:)];
        elseif idx_l == 6
            max_cr_x = [max_cr_x;max_act(i)];
            pred_cr_x = [pred_cr_x;pred(i,:)];
        else 
            error(msgtext);
        end
        max_cr = [max_cr;max_act(i)];
        pred_cr = [pred_cr;pred(i,:)];
    elseif idx_s == 2
        if idx_l == 1
            max_el_a = [max_el_a;max_act(i)];
            pred_el_a = [pred_el_a;pred(i,:)];
        elseif idx_l == 2
            max_el_h = [max_el_h;max_act(i)];
            pred_el_h = [pred_el_h;pred(i,:)];
        else
            error(msgtext);
        end
        max_el= [max_el;max_act(i)];
        pred_el = [pred_el;pred(i,:)];
    elseif idx_s == 3
        if idx_l == 3
            max_he_m = [max_he_m;max_act(i)];
            pred_he_m = [pred_he_m;pred(i,:)];
        elseif idx_l == 4
            max_he_t = [max_he_t;max_act(i)];
            pred_he_t = [pred_he_t;pred(i,:)];
        else
            error(msgtext);
        end
        max_he= [max_he;max_act(i)];
        pred_he = [pred_he;pred(i,:)];
    elseif idx_s == 4
        if idx_l == 1
            max_re_a = [max_re_a;max_act(i)];
            pred_re_a = [pred_re_a;pred(i,:)];
        elseif idx_l == 5
            max_re_u = [max_re_u;max_act(i)];
            pred_re_u = [pred_re_u;pred(i,:)];
        else
            error(msgtext);
        end
        max_re= [max_re;max_act(i)];
        pred_re = [pred_re;pred(i,:)];
    elseif idx_s == 5
        if idx_l == 4
            max_sq_t = [max_sq_t;max_act(i)];
            pred_sq_t = [pred_sq_t;pred(i,:)];
        elseif idx_l == 6
            max_sq_x = [max_sq_x;max_act(i)];
            pred_sq_x = [pred_sq_x;pred(i,:)];
        else
            error(msgtext);
        end
        max_sq= [max_sq;max_act(i)];
        pred_sq = [pred_sq;pred(i,:)];
    elseif idx_s == 6
        if idx_l == 2
            max_tr_h = [max_tr_h;max_act(i)];
            pred_tr_h = [pred_tr_h;pred(i,:)];
        elseif idx_l == 5
            max_tr_u = [max_tr_u;max_act(i)];
            pred_tr_u = [pred_tr_u;pred(i,:)];
        else
            error(msgtext);
        end
        max_tr= [max_tr;max_act(i)];
        pred_tr = [pred_tr;pred(i,:)];
    end
end
mode_cr = mode(max_cr);std_cr = std(max_cr);prD_cr =  mean(pred_cr,1);
mode_el = mode(max_el);std_el = std(max_el);prD_el =  mean(pred_el,1);
mode_he = mode(max_he);std_he = std(max_he);prD_he =  mean(pred_he,1);
mode_re = mode(max_re);std_re = std(max_re);prD_re =  mean(pred_re,1);
mode_sq = mode(max_sq);std_sq = std(max_sq);prD_sq =  mean(pred_sq,1);
mode_tr = mode(max_tr);std_tr = std(max_tr);prD_tr =  mean(pred_tr,1);
r1 = ones(size(rbms_pass,1)/6,1);r2 = ones(size(rbms_pass,1)/6,1)*2;r3 = ones(size(rbms_pass,1)/6,1)*3;
r4 = ones(size(rbms_pass,1)/6,1)*4;r5 = ones(size(rbms_pass,1)/6,1)*5;r6 = ones(size(rbms_pass,1)/6,1)*6;
acc1 = (max_cr == r1);% cross, ---- T
acc2 = (max_el == r2);% elipse ---- U
acc3 = (max_he == r3);% hexagon ---- X
acc4 = (max_re == r4);% rectangle ---- H
acc5 = (max_sq == r5); % square ---- M
acc6 = (max_tr == r6);% triangle ---- A
acc1 = mean(acc1);acc2 = mean(acc2);acc3 = mean(acc3);
acc4 = mean(acc4);acc5 = mean(acc5);acc6 = mean(acc6);
% create Table for Letters:
Targets = ["A/Tria";"H/Rect";"M/Squ";"U/Elips";"T/Cross";"X/Hexa"];
Acc = [acc6;acc4;acc5;acc2;acc1;acc3];
Mode = [mode_tr;mode_re;mode_sq;mode_el;mode_cr;mode_he];
Std = [mode_tr;mode_re;mode_sq;mode_el;mode_cr;mode_he];
incl_ce_details = table(Targets,Mode,Std,Acc);
incl_ce_pdr = [prD_tr;prD_re;prD_sq;prD_el;prD_cr;prD_he];

%%%% Extra Extended for Incongruent Case
r1 = ones(size(rbms_pass,1)/12,1);r2 = ones(size(rbms_pass,1)/12,1)*2;r3 = ones(size(rbms_pass,1)/12,1)*3;
r4 = ones(size(rbms_pass,1)/12,1)*4;r5 = ones(size(rbms_pass,1)/12,1)*5;r6 = ones(size(rbms_pass,1)/12,1)*6;
accT_h = mean(max_tr_h==r6);accT_u = mean(max_tr_u == r6);
accR_a = mean(max_re_a == r4);accR_u = mean(max_re_u == r4);
accS_t = mean(max_sq_t == r5);accS_x = mean(max_sq_x == r5);
accE_h = mean(max_el_h == r2);accE_a = mean(max_el_a == r2);
accC_m = mean(max_cr_m == r1);accC_x = mean(max_cr_x == r1);
accH_m = mean(max_he_m == r3);accH_t = mean(max_he_t == r3);
pred_cr_m =  mean(pred_cr_m,1);pred_cr_x =  mean(pred_cr_x,1);
pred_el_a =  mean(pred_el_a,1);pred_el_h =  mean(pred_el_h,1);
pred_he_m =  mean(pred_he_m,1);pred_he_t =  mean(pred_he_t,1);
pred_re_a =  mean(pred_re_a,1);pred_re_u =  mean(pred_re_u,1);
pred_sq_t =  mean(pred_sq_t,1);pred_sq_x =  mean(pred_sq_x,1);
pred_tr_h =  mean(pred_tr_h,1);pred_tr_u =  mean(pred_tr_u,1);
mode_cr_m = mode(max_cr_m);mode_cr_x = mode(max_cr_x);
mode_el_a = mode(max_el_a);mode_el_h = mode(max_el_h);
mode_he_m = mode(max_he_m);mode_he_t = mode(max_he_t);
mode_re_a = mode(max_re_a); mode_re_u = mode(max_re_u);
mode_sq_t = mode(max_sq_t);mode_sq_x = mode(max_sq_x);
mode_tr_h = mode(max_tr_h);mode_tr_u = mode(max_tr_u);

Letter_Acc = [accT_h;accT_u;accR_a;accR_u;accS_t;accS_x;accE_a;accE_h;accC_m;accC_x;accH_m;accH_t];
Letter_Mode = [mode_tr_h;mode_tr_u;mode_re_a;mode_re_u;mode_sq_t;mode_sq_x;...
    mode_el_a;mode_el_h;mode_cr_m;mode_cr_x;mode_he_m;mode_he_t];

inc_l_pdr_detailed = [pred_tr_h;pred_tr_u;pred_re_a;pred_re_u;pred_sq_t;pred_sq_x;pred_el_a;pred_el_h; ...
    pred_cr_m;pred_cr_x;pred_he_m;pred_he_t];

%% EVAL - Pseudo-Letter Incongruent
% pass data throught RBMs:
hid_out_1_d = 1./(1 + exp(-inc_pl_d*vishid_1 - repmat(hidbiases_1,size(inc_pl_d,1),1)));
rbms_pass = 1./(1 + exp(-hid_out_1_d*vishid_2 - repmat(hidbiases_2,size(hid_out_1_d,1),1)));
pred = net2(rbms_pass');
pred = pred';
% [a,b] =max(pred,[],2); --- % a has also  values  > 1 ---- 
% i.e. a is unnormalized
[~, max_act] = max(pred,[],2); % max_act are indices of dim2 in pred of the highest value
[r1,~] = find(inc_pl_t'); % find which columns (rows in transpose) are 1
[r2,~] = find(inc_pl_s'); % find which columns (rows in transpose) are 1
acc_l = (max_act == r1);
acc_s = (max_act == r2);
accuracy_l = mean(acc_l);
accuracy_s = mean(acc_s);
fprintf(1,'\n Prediction / Assesment of the Congruency Effect\n');
fprintf(1,'\n ------- Case : Incongruent ------- \n');
fprintf(1,'\n ------- Target : Pseudo Letter ------- \n');
fprintf(1,'\n If Decision based on letter? \n Accuracy = %d\n ',accuracy_l);
fprintf(1,'\n If Decision based on shape? \n Accuracy = %d\n ',accuracy_s);

Shape_decisionAcc(4) = accuracy_s;
Letter_decisionAcc(4) = accuracy_l;

%%%% Extended ASSESMENT
max_cr = [];max_el = [];max_he = [];max_re = [];max_sq = [];max_tr = [];
pred_cr = [];pred_el = [];pred_he = [];pred_re = [];pred_sq = [];pred_tr = [];
% for more details, for each incongruent case
max_cr_m = [];max_cr_x=[];max_el_h = [];max_el_a = [];max_he_m = [];max_he_t=[];
max_re_a = [];max_re_u=[];max_sq_t = [];max_sq_x=[];max_tr_h= [];max_tr_u = [];
pred_cr_m = [];pred_cr_x = [];pred_el_a= [];pred_el_h=[];pred_he_m = [];pred_he_t=[];
pred_re_a = [];pred_re_u = [];pred_sq_t = [];pred_sq_x=[];pred_tr_h = [];pred_tr_u=[];
for i=1:size(max_act,1)
    idx_s = find(inc_l_s(i,:));
    idx_l = find(inc_l_t(i,:));
    if idx_s == 1 
        if idx_l == 3
            max_cr_m = [max_cr_m;max_act(i)];
            pred_cr_m = [pred_cr_m;pred(i,:)];
        elseif idx_l == 6
            max_cr_x = [max_cr_x;max_act(i)];
            pred_cr_x = [pred_cr_x;pred(i,:)];
        else 
            error(msgtext);
        end
        max_cr = [max_cr;max_act(i)];
        pred_cr = [pred_cr;pred(i,:)];
    elseif idx_s == 2
        if idx_l == 1
            max_el_a = [max_el_a;max_act(i)];
            pred_el_a = [pred_el_a;pred(i,:)];
        elseif idx_l == 2
            max_el_h = [max_el_h;max_act(i)];
            pred_el_h = [pred_el_h;pred(i,:)];
        else
            error(msgtext);
        end
        max_el= [max_el;max_act(i)];
        pred_el = [pred_el;pred(i,:)];
    elseif idx_s == 3
        if idx_l == 3
            max_he_m = [max_he_m;max_act(i)];
            pred_he_m = [pred_he_m;pred(i,:)];
        elseif idx_l == 4
            max_he_t = [max_he_t;max_act(i)];
            pred_he_t = [pred_he_t;pred(i,:)];
        else
            error(msgtext);
        end
        max_he= [max_he;max_act(i)];
        pred_he = [pred_he;pred(i,:)];
    elseif idx_s == 4
        if idx_l == 1
            max_re_a = [max_re_a;max_act(i)];
            pred_re_a = [pred_re_a;pred(i,:)];
        elseif idx_l == 5
            max_re_u = [max_re_u;max_act(i)];
            pred_re_u = [pred_re_u;pred(i,:)];
        else
            error(msgtext);
        end
        max_re= [max_re;max_act(i)];
        pred_re = [pred_re;pred(i,:)];
    elseif idx_s == 5
        if idx_l == 4
            max_sq_t = [max_sq_t;max_act(i)];
            pred_sq_t = [pred_sq_t;pred(i,:)];
        elseif idx_l == 6
            max_sq_x = [max_sq_x;max_act(i)];
            pred_sq_x = [pred_sq_x;pred(i,:)];
        else
            error(msgtext);
        end
        max_sq= [max_sq;max_act(i)];
        pred_sq = [pred_sq;pred(i,:)];
    elseif idx_s == 6
        if idx_l == 2
            max_tr_h = [max_tr_h;max_act(i)];
            pred_tr_h = [pred_tr_h;pred(i,:)];
        elseif idx_l == 5
            max_tr_u = [max_tr_u;max_act(i)];
            pred_tr_u = [pred_tr_u;pred(i,:)];
        else
            error(msgtext);
        end
        max_tr= [max_tr;max_act(i)];
        pred_tr = [pred_tr;pred(i,:)];
    end
end
mode_cr = mode(max_cr);std_cr = std(max_cr);prD_cr =  mean(pred_cr,1);
mode_el = mode(max_el);std_el = std(max_el);prD_el =  mean(pred_el,1);
mode_he = mode(max_he);std_he = std(max_he);prD_he =  mean(pred_he,1);
mode_re = mode(max_re);std_re = std(max_re);prD_re =  mean(pred_re,1);
mode_sq = mode(max_sq);std_sq = std(max_sq);prD_sq =  mean(pred_sq,1);
mode_tr = mode(max_tr);std_tr = std(max_tr);prD_tr =  mean(pred_tr,1);
r1 = ones(size(rbms_pass,1)/6,1);r2 = ones(size(rbms_pass,1)/6,1)*2;r3 = ones(size(rbms_pass,1)/6,1)*3;
r4 = ones(size(rbms_pass,1)/6,1)*4;r5 = ones(size(rbms_pass,1)/6,1)*5;r6 = ones(size(rbms_pass,1)/6,1)*6;
acc1 = (max_cr == r1);% cross, ---- T
acc2 = (max_el == r2);% elipse ---- U
acc3 = (max_he == r3);% hexagon ---- X
acc4 = (max_re == r4);% rectangle ---- H
acc5 = (max_sq == r5); % square ---- M
acc6 = (max_tr == r6);% triangle ---- A
acc1 = mean(acc1);acc2 = mean(acc2);acc3 = mean(acc3);
acc4 = mean(acc4);acc5 = mean(acc5);acc6 = mean(acc6);
% create Table for Letters:
Targets = ["pA/Tria";"pH/Rect";"pM/Squ";"pU/Elips";"pT/Cross";"pX/Hexa"];
Acc = [acc6;acc4;acc5;acc2;acc1;acc3];
Mode = [mode_tr;mode_re;mode_sq;mode_el;mode_cr;mode_he];
Std = [mode_tr;mode_re;mode_sq;mode_el;mode_cr;mode_he];
incpl_ce_details = table(Targets,Mode,Std,Acc);
incpl_ce_pdr = [prD_tr;prD_re;prD_sq;prD_el;prD_cr;prD_he];
%%%% Extra Extended for Incongruent Case
r1 = ones(size(rbms_pass,1)/12,1);r2 = ones(size(rbms_pass,1)/12,1)*2;r3 = ones(size(rbms_pass,1)/12,1)*3;
r4 = ones(size(rbms_pass,1)/12,1)*4;r5 = ones(size(rbms_pass,1)/12,1)*5;r6 = ones(size(rbms_pass,1)/12,1)*6;
accT_h = mean(max_tr_h==r6);accT_u = mean(max_tr_u == r6);
accR_a = mean(max_re_a == r4);accR_u = mean(max_re_u == r4);
accS_t = mean(max_sq_t == r5);accS_x = mean(max_sq_x == r5);
accE_h = mean(max_el_h == r2);accE_a = mean(max_el_a == r2);
accC_m = mean(max_cr_m == r1);accC_x = mean(max_cr_x == r1);
accH_m = mean(max_he_m == r3);accH_t = mean(max_he_t == r3);
pred_cr_m =  mean(pred_cr_m,1);pred_cr_x =  mean(pred_cr_x,1);
pred_el_a =  mean(pred_el_a,1);pred_el_h =  mean(pred_el_h,1);
pred_he_m =  mean(pred_he_m,1);pred_he_t =  mean(pred_he_t,1);
pred_re_a =  mean(pred_re_a,1);pred_re_u =  mean(pred_re_u,1);
pred_sq_t =  mean(pred_sq_t,1);pred_sq_x =  mean(pred_sq_x,1);
pred_tr_h =  mean(pred_tr_h,1);pred_tr_u =  mean(pred_tr_u,1);
mode_cr_m = mode(max_cr_m);mode_cr_x = mode(max_cr_x);
mode_el_a = mode(max_el_a);mode_el_h = mode(max_el_h);
mode_he_m = mode(max_he_m);mode_he_t = mode(max_he_t);
mode_re_a = mode(max_re_a); mode_re_u = mode(max_re_u);
mode_sq_t = mode(max_sq_t);mode_sq_x = mode(max_sq_x);
mode_tr_h = mode(max_tr_h);mode_tr_u = mode(max_tr_u);

psLetter_Acc = [accT_h;accT_u;accR_a;accR_u;accS_t;accS_x;accE_a;accE_h;accC_m;accC_x;accH_m;accH_t];
psLetter_Mode = [mode_tr_h;mode_tr_u;mode_re_a;mode_re_u;mode_sq_t;mode_sq_x;...
    mode_el_a;mode_el_h;mode_cr_m;mode_cr_x;mode_he_m;mode_he_t];

inc_pl_pdr_detailed = [pred_tr_h;pred_tr_u;pred_re_a;pred_re_u;pred_sq_t;pred_sq_x;pred_el_a;pred_el_h; ...
    pred_cr_m;pred_cr_x;pred_he_m;pred_he_t];

%% store the evaluation data
Targets = ["Cong_Letter" ; "Cong_Ps-Letter"; "Inc_Letter";"Inc_Ps-Letter"];
CE_eval.overall = table(Targets,Shape_decisionAcc,Letter_decisionAcc);
CE_eval.CEs_Letter = Shape_decisionAcc(1)*100 - Shape_decisionAcc(3)*100;
CE_eval.CEs_psLetter = Shape_decisionAcc(2)*100 - Shape_decisionAcc(4)*100;

Targets = ["A/Tria";"H/Rect";"M/Squ";"U/Elips";"T/Cross";"X/Hexa"];
letter_acc = cl_ce_details.Acc; letter_mode = cl_ce_details.Mode;
psLetter_acc = cpl_ce_details.Acc; psLetter_mode = cpl_ce_details.Mode;
CE_eval.Congruent = table(Targets,letter_acc,letter_mode,psLetter_acc,psLetter_mode);

Targets = ["Tria_H";"Tria_U";"Rect_A";"Rect_U";"Squa_T";"Squa_X"; ...
    "Elips_A";"Elips_H";"Cross_M";"Cross_X";"Hexa_M";"Hexa_T"];
inc_whole = table(Targets,Letter_Acc,Letter_Mode,psLetter_Acc,psLetter_Mode);
CE_eval.Incongruent = inc_whole;

details.cong_l = cl_ce_details;
details.cong_pl = cpl_ce_details;
details.inc_l = incl_ce_details;
details.inc_pl = incpl_ce_details;
% for pdf:
% rows = targets (A,H,M,...) && columns = possible classes
details.pdr_cong_l = cl_ce_pdr;
details.pdr_cong_pl = cpl_ce_pdr;
details.pdr_inc_l = incl_ce_pdr;
details.pdr_inc_pl = incpl_ce_pdr;
details.detailed_inc_pdr_l = inc_l_pdr_detailed;
details.detailed_inc_pdr_pl = inc_pl_pdr_detailed;

CE_eval.detail = details;

%%% not relevant as far as I understood:
% CEl_Letter = Letter_decisionAcc(1)*100 - Letter_decisionAcc(3)*100;
% CEl_psLetter = Letter_decisionAcc(2)*100 - Letter_decisionAcc(4)*100;

