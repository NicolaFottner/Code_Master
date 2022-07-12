%%%
%   This method changed to its illiterate counterpart in which the CE was computed based
%   on the outer shapes identity: (triangle-A) <-> (triangle-H)
%   Here, it is measure based on the letter identity: (A-tr) <-> (A-sq)
%
%   Therefore, for the incongruent cases, the accuracies represent the 
%   models' hit rate on the inner target.
%%%

assert(literate);addpath("data/");
load openCV_CE_data.mat cong_pl_d cong_pl_t cong_pl_s inc_pl_d inc_pl_t inc_pl_s 
load openCV_newL_CE.mat cong_l_d cong_l_t cong_l_s inc_l_d inc_l_t inc_l_s 

%% Prep and convert openCV_CE_data from uint8 to double:
xdata = zeros(size(cong_l_d));
for i=1:size(cong_l_d,1)
    xdata(i,:) = reshape(im2double(reshape(cong_l_d(i,:),[40 40 1])), [1 1600]);
end
cong_l_d = xdata;xdata = zeros(size(cong_pl_d));
for i=1:size(cong_pl_d,1)
    xdata(i,:) = reshape(im2double(reshape(cong_pl_d(i,:),[40 40 1])), [1 1600]);
end
cong_pl_d = xdata;xdata = zeros(size(inc_l_d));
for i=1:size(inc_l_d,1)
    xdata(i,:) = reshape(im2double(reshape(inc_l_d(i,:),[40 40 1])), [1 1600]);
end
inc_l_d = xdata;xdata = zeros(size(inc_pl_d));
for i=1:size(inc_pl_d,1)
    xdata(i,:) = reshape(im2double(reshape(inc_pl_d(i,:),[40 40 1])), [1 1600]);
end
inc_pl_d = xdata;

%%% concatenate the targets (might be unused)
% cong_l_t changes from only targets for letter, for both letter and shape
xcong_l_t = cat(2,double(cong_l_t),double(cong_l_s));
xcong_pl_t =  cat(2,double(cong_pl_t),double(cong_pl_s));
xinc_l_t =  cat(2,double(inc_l_t),double(inc_l_s));
xinc_pl_t =  cat(2,double(inc_pl_t),double(inc_pl_s));

z = zeros(size(cong_l_t));
cong_l_t =  cat(2,double(cong_l_t),z);
cong_l_s = cat(2,z,double(cong_l_s));
cong_pl_t =  cat(2,double(cong_pl_t),z);
cong_pl_s = cat(2,z,double(cong_pl_s));
z = zeros(size(inc_l_t));
inc_l_t =  cat(2,double(inc_l_t),z);
inc_l_s = cat(2,z,double(inc_l_s));
inc_pl_t =  cat(2,double(inc_pl_t),z);
inc_pl_s = cat(2,z,double(inc_pl_s));

%load rbm data
load t_model DN
vishid_1 = DN.L{1,1}.vishid;
hidbiases_1 = DN.L{1,1}.hidbiases;
clear DN
if literate
    load l_rbm_2.mat vishid_2 hidbiases_2
else
    load g_rbm_2.mat vishid_2 hidbiases_2
end%load rbm2_16J11h39.mat vishid_2 hidbiases_2

%%% Create variables for Table -- visualization purpose
% table with accuracy values for shape
Shape_decisionAcc = zeros(4,1); 
Letter_decisionAcc = zeros(4,1);


%% EVAL - Letter congruent
% pass data throught RBMs:
hid_out_1_d = 1./(1 + exp(-cong_l_d*vishid_1 - repmat(hidbiases_1,size(cong_l_d,1),1)));
rbms_pass = 1./(1 + exp(-hid_out_1_d*vishid_2 - repmat(hidbiases_2,size(hid_out_1_d,1),1)));
% Compute prediction:
pred = net(rbms_pass');
[~, max_act] = max(pred,[],2); % max_act are indices of dim2 in pred of the highest value
[r1,~] = find(cong_l_t'); % letter id
[r2,~] = find(cong_l_s'); % shape id
acc_l = (max_act == r1);
acc_s = (max_act == r2);
accuracy_l = mean(acc_l);
accuracy_s = mean(acc_s);
fprintf(1,'\n Prediction / Assesment of the Congruency Effect\n');
fprintf(1,'\n ------- Case : Congruent ------- \n');
fprintf(1,'\n ------- Target : Letter ------- \n');
fprintf(1,'\n If Accuracy in classifying the inner-letter correctly = %d\n ',accuracy_l);
fprintf(1,'\n If Accuracy in classifying the outer-shape correctly = %d\n ',accuracy_s);
Shape_decisionAcc(1) = accuracy_s;
Letter_decisionAcc(1) = accuracy_l;


%% EXTENDED EVAL FOR INNER:
% this is differnt to the illiterate counterpart/method

max_a = [];max_h = [];max_m = [];max_t = [];max_u = [];max_x = [];
pred_a = [];pred_h= [];pred_m = [];pred_t = [];pred_u = [];pred_x = [];
for i=1:size(max_act,1)
    idx_s = find(cong_l_t(i,:));
    if idx_s == 1 
        max_a = [max_a;max_act(i)];
        pred_a = [pred_a;pred(i,:)];
    elseif idx_s == 2
        max_h= [max_h;max_act(i)];
        pred_h = [pred_h;pred(i,:)];
    elseif idx_s == 3
        max_m= [max_m;max_act(i)];
        pred_m = [pred_m;pred(i,:)];
    elseif idx_s == 4
        max_t= [max_t;max_act(i)];
        pred_t = [pred_t;pred(i,:)];
    elseif idx_s == 5
        max_u= [max_u;max_act(i)];
        pred_u = [pred_u;pred(i,:)];
    elseif idx_s == 6
        max_x= [max_x;max_act(i)];
        pred_x = [pred_x;pred(i,:)];
    end
end
mode_a = mode(max_a);std_a = std(max_a);prD_a =  mean(pred_a,1);
mode_h = mode(max_h);std_h = std(max_h);prD_h =  mean(pred_h,1);
mode_m = mode(max_m);std_m = std(max_m);prD_m =  mean(pred_m,1);
mode_t = mode(max_t);std_t = std(max_t);prD_t =  mean(pred_t,1);
mode_u = mode(max_u);std_u = std(max_u);prD_u =  mean(pred_u,1);
mode_x = mode(max_x);std_x = std(max_x);prD_x =  mean(pred_x,1);
r1 = ones(size(rbms_pass,1)/6,1)*1;r2 = ones(size(rbms_pass,1)/6,1)*2;r3 = ones(size(rbms_pass,1)/6,1)*3;
r4 = ones(size(rbms_pass,1)/6,1)*4;r5 = ones(size(rbms_pass,1)/6,1)*5;r6 = ones(size(rbms_pass,1)/6,1)*6;
acc1 = (max_a == r1);% A
acc2 = (max_h == r2);% H
acc3 = (max_m == r3);% M
acc4 = (max_t == r4);% T
acc5 = (max_u == r5); % U
acc6 = (max_x == r6);% X
acc1 = mean(acc1);acc2 = mean(acc2);acc3 = mean(acc3);
acc4 = mean(acc4);acc5 = mean(acc5);acc6 = mean(acc6);
% create Table for Letters: (with old order:)
% Targets = ["Cross/T" ; "Elipse/U"; "Hexa/X"; "Rect/H"; "Square/M"; "Tria/A"];
% Acc = [acc1;acc2;acc3;acc4;acc5;acc6];
% Mode = [mode_cr;mode_el;mode_he;mode_re;mode_sq;mode_tr];
% Std = [std_cr;std_el;std_he;std_re;std_sq;std_tr];
%%% better order:
Targets = ["A/Tria";"H/Rect";"M/Squ";"U/Elips";"T/Cross";"X/Hexa"];
Acc = [acc1;acc2;acc3;acc5;acc4;acc6];
Mode = [mode_a;mode_h;mode_m;mode_u;mode_t;mode_x];
Std = [std_a;std_h;std_m;std_u;std_t;std_x];
cl_ce_details_inner = table(Targets,Mode,Std,Acc);
cl_ce_pdr_inner = [prD_a;prD_h;prD_m;prD_u;prD_t;prD_x];

%% EVAL - Pseudo Letter congruent
% pass data throught RBMs:
hid_out_1_d = 1./(1 + exp(-cong_pl_d*vishid_1 - repmat(hidbiases_1,size(cong_pl_d,1),1)));
rbms_pass = 1./(1 + exp(-hid_out_1_d*vishid_2 - repmat(hidbiases_2,size(hid_out_1_d,1),1)));
% Compute prediction:
pred = net(rbms_pass');
[~, max_act] = max(pred,[],2); % max_act are indices of dim2 in pred of the highest value
[r1,~] = find(cong_pl_t'); % find which columns (rows in transpose) are 1
[r2,~] = find(cong_pl_s'); % find which columns (rows in transpose) are 1
acc_l = (max_act == r1);
acc_s = (max_act == r2);
accuracy_l = mean(acc_l);
accuracy_s = mean(acc_s);
fprintf(1,'\n Prediction / Assesment of the Congruency Effect\n');
fprintf(1,'\n ------- Case : Congruent ------- \n');
fprintf(1,'\n ------- Target : Letter ------- \n');
fprintf(1,'\n If Accuracy in classifying the inner-letter correctly = %d\n ',accuracy_l);
fprintf(1,'\n If Accuracy in classifying the outer-shape correctly = %d\n ',accuracy_s);

Shape_decisionAcc(2) = accuracy_s;
Letter_decisionAcc(2) = accuracy_l;

%%%% Extended ASSESMENT
max_a = [];max_h = [];max_m = [];max_t = [];max_u = [];max_x = [];
pred_a = [];pred_h = [];pred_m = [];pred_t = [];pred_u = [];pred_x = [];
for i=1:size(max_act,1)
    idx_s = find(cong_pl_t(i,:));
    if idx_s == 1 
        max_a = [max_a;max_act(i)];
        pred_a = [pred_a;pred(i,:)];
    elseif idx_s == 2
        max_h= [max_h;max_act(i)];
        pred_h = [pred_h;pred(i,:)];
    elseif idx_s == 3
        max_m= [max_m;max_act(i)];
        pred_m = [pred_m;pred(i,:)];
    elseif idx_s == 4
        max_t= [max_t;max_act(i)];
        pred_t = [pred_t;pred(i,:)];
    elseif idx_s == 5
        max_u= [max_u;max_act(i)];
        pred_u = [pred_u;pred(i,:)];
    elseif idx_s == 6
        max_x= [max_x;max_act(i)];
        pred_x = [pred_x;pred(i,:)];
    end
end
mode_a = mode(max_a);std_a = std(max_a);prD_a =  mean(pred_a,1);
mode_h = mode(max_h);std_h = std(max_h);prD_h =  mean(pred_h,1);
mode_m = mode(max_m);std_m = std(max_m);prD_m =  mean(pred_m,1);
mode_t = mode(max_t);std_t = std(max_t);prD_t =  mean(pred_t,1);
mode_u = mode(max_u);std_u = std(max_u);prD_u =  mean(pred_u,1);
mode_x = mode(max_x);std_x = std(max_x);prD_x =  mean(pred_x,1);
r1 = ones(size(rbms_pass,1)/6,1);r2 = ones(size(rbms_pass,1)/6,1)*2;r3 = ones(size(rbms_pass,1)/6,1)*3;
r4 = ones(size(rbms_pass,1)/6,1)*4;r5 = ones(size(rbms_pass,1)/6,1)*5;r6 = ones(size(rbms_pass,1)/6,1)*6;
acc1 = (max_a == r1);% A
acc2 = (max_h == r2);% H
acc3 = (max_m == r3);% M
acc4 = (max_t == r4);% T
acc5 = (max_u == r5); % U
acc6 = (max_x == r6);% X
acc1 = mean(acc1);acc2 = mean(acc2);acc3 = mean(acc3);
acc4 = mean(acc4);acc5 = mean(acc5);acc6 = mean(acc6);
% create Table for Letters:
%Targets = ["pA" ; "pH"; "pM"; "pU"; "pT"; "pX"];
Targets = ["pA/Tria";"pH/Rect";"pM/Squ";"pU/Elips";"pT/Cross";"pX/Hexa"];
Acc = [acc1;acc2;acc3;acc5;acc4;acc6];
Mode = [mode_a;mode_h;mode_m;mode_u;mode_t;mode_x];
Std = [std_a;std_h;std_m;std_u;std_t;std_x];
cpl_ce_details = table(Targets,Mode,Std,Acc);
cpl_ce_pdr = [prD_a;prD_h;prD_m;prD_u;prD_t;prD_x];


%% EVAL - Letter Incongruent
% pass data throught RBMs:
hid_out_1_d = 1./(1 + exp(-inc_l_d*vishid_1 - repmat(hidbiases_1,size(inc_l_d,1),1)));
rbms_pass = 1./(1 + exp(-hid_out_1_d*vishid_2 - repmat(hidbiases_2,size(hid_out_1_d,1),1)));
% Compute prediction:
pred = net(rbms_pass');
[~, max_act] = max(pred,[],2);
[r1,~] = find(inc_l_t');
[r2,~] = find(inc_l_s');
acc_l = (max_act == r1);
acc_s = (max_act == r2);
accuracy_l = mean(acc_l);
accuracy_s = mean(acc_s);
fprintf(1,'\n Prediction / Assesment of the Congruency Effect\n');
fprintf(1,'\n ------- Case : Congruent ------- \n');
fprintf(1,'\n ------- Target : Letter ------- \n');
fprintf(1,'\n If Accuracy in classifying the inner-letter correctly = %d\n ',accuracy_l);
fprintf(1,'\n If Accuracy in classifying the outer-shape correctly = %d\n ',accuracy_s);

Shape_decisionAcc(3) = accuracy_s;
Letter_decisionAcc(3) = accuracy_l;

%%%% Extended ASSESMENT
max_a = [];max_h = [];max_m = [];max_t = [];max_u = [];max_x = [];
pred_a = [];pred_h = [];pred_m = [];pred_t = [];pred_u = [];pred_x = [];
% for more details, for each incongruent case
max_m_cr = [];max_x_cr=[];max_h_el = [];max_a_el = [];max_m_he = [];max_t_he=[];
max_a_re = [];max_u_re=[];max_t_sq = [];max_x_sq=[];max_h_tr= [];max_u_tr = [];
pred_m_cr = [];pred_x_cr = [];pred_a_el= [];pred_h_el=[];pred_m_he = [];pred_t_he=[];
pred_a_re = [];pred_u_re = [];pred_t_sq = [];pred_x_sq=[];pred_h_tr = [];pred_u_tr=[];
msgtext = 'Invalid Incongruent Data/case indexing';

for i=1:size(max_act,1)
    idx_s = find(inc_l_s(i,:));
    idx_l = find(inc_l_t(i,:));
    if idx_s == 7 
        if idx_l == 3
            max_m_cr = [max_m_cr;max_act(i)];
            pred_m_cr = [pred_m_cr;pred(i,:)];
        elseif idx_l == 6
            max_x_cr = [max_x_cr;max_act(i)];
            pred_x_cr = [pred_x_cr;pred(i,:)];
        else 
            error(msgtext);
        end
        max_a = [max_a;max_act(i)];
        pred_a = [pred_a;pred(i,:)];
    elseif idx_s == 8
        if idx_l == 1
            max_a_el = [max_a_el;max_act(i)];
            pred_a_el = [pred_a_el;pred(i,:)];
        elseif idx_l == 2
            max_h_el = [max_h_el;max_act(i)];
            pred_h_el = [pred_h_el;pred(i,:)];
        else
            error(msgtext);
        end
        max_h= [max_h;max_act(i)];
        pred_h = [pred_h;pred(i,:)];
    elseif idx_s == 9
        if idx_l == 3
            max_m_he = [max_m_he;max_act(i)];
            pred_m_he = [pred_m_he;pred(i,:)];
        elseif idx_l == 4
            max_t_he = [max_t_he;max_act(i)];
            pred_t_he = [pred_t_he;pred(i,:)];
        else
            error(msgtext);
        end
        max_m= [max_m;max_act(i)];
        pred_m = [pred_m;pred(i,:)];
    elseif idx_s == 10
        if idx_l == 1
            max_a_re = [max_a_re;max_act(i)];
            pred_a_re = [pred_a_re;pred(i,:)];
        elseif idx_l == 5
            max_u_re = [max_u_re;max_act(i)];
            pred_u_re = [pred_u_re;pred(i,:)];
        else
            error(msgtext);
        end
        max_t= [max_t;max_act(i)];
        pred_t = [pred_t;pred(i,:)];
    elseif idx_s == 11
        if idx_l == 4
            max_t_sq = [max_t_sq;max_act(i)];
            pred_t_sq = [pred_t_sq;pred(i,:)];
        elseif idx_l == 6
            max_x_sq = [max_x_sq;max_act(i)];
            pred_x_sq = [pred_x_sq;pred(i,:)];
        else
            error(msgtext);
        end
        max_u= [max_u;max_act(i)];
        pred_u = [pred_u;pred(i,:)];
    elseif idx_s == 12
        if idx_l == 2
            max_h_tr = [max_h_tr;max_act(i)];
            pred_h_tr = [pred_h_tr;pred(i,:)];
        elseif idx_l == 5
            max_u_tr = [max_u_tr;max_act(i)];
            pred_u_tr = [pred_u_tr;pred(i,:)];
        else
            error(msgtext);
        end
        max_x= [max_x;max_act(i)];
        pred_x = [pred_x;pred(i,:)];
    end
end
mode_a = mode(max_a);std_a = std(max_a);prD_a =  mean(pred_a,1);
mode_h = mode(max_h);std_h = std(max_h);prD_h =  mean(pred_h,1);
mode_m = mode(max_m);std_m = std(max_m);prD_m =  mean(pred_m,1);
mode_t = mode(max_t);std_t = std(max_t);prD_t =  mean(pred_t,1);
mode_u = mode(max_u);std_u = std(max_u);prD_u =  mean(pred_u,1);
mode_x = mode(max_x);std_x = std(max_x);prD_x =  mean(pred_x,1);
r1 = ones(size(rbms_pass,1)/6,1);r2 = ones(size(rbms_pass,1)/6,1)*2;r3 = ones(size(rbms_pass,1)/6,1)*3;
r4 = ones(size(rbms_pass,1)/6,1)*4;r5 = ones(size(rbms_pass,1)/6,1)*5;r6 = ones(size(rbms_pass,1)/6,1)*6;
acc1 = (max_a == r1);% A
acc2 = (max_h == r2);% H
acc3 = (max_m == r3);% M
acc4 = (max_t == r4);% T
acc5 = (max_u == r5); % U
acc6 = (max_x == r6);% X
acc1 = mean(acc1);acc2 = mean(acc2);acc3 = mean(acc3);
acc4 = mean(acc4);acc5 = mean(acc5);acc6 = mean(acc6);
% create Table for Letters:
Targets = ["A/Tria";"H/Rect";"M/Squ";"U/Elips";"T/Cross";"X/Hexa"];
Acc = [acc1;acc2;acc3;acc5;acc4;acc6];
Mode = [mode_a;mode_h;mode_m;mode_u;mode_t;mode_x];
Std = [std_a;std_h;std_m;std_u;std_t;std_x];
incl_ce_details = table(Targets,Mode,Std,Acc);
incl_ce_pdr = [prD_a;prD_h;prD_m;prD_u;prD_t;prD_x];

%%%% Extra Extended for Incongruent Case
r1 = ones(size(rbms_pass,1)/12,1);r2 = ones(size(rbms_pass,1)/12,1)*2;r3 = ones(size(rbms_pass,1)/12,1)*3;
r4 = ones(size(rbms_pass,1)/12,1)*4;r5 = ones(size(rbms_pass,1)/12,1)*5;r6 = ones(size(rbms_pass,1)/12,1)*6;
acc_hTr = mean(max_h_tr == r2);acc_uTr = mean(max_u_tr == r5);
acc_aRec = mean(max_a_re == r1);acc_uRec = mean(max_u_re == r5);
acc_tSq = mean(max_t_sq == r4);acc_xSq = mean(max_x_sq == r6);
acc_hEl = mean(max_h_el == r2);acc_aEl = mean(max_a_el == r1);
acc_mCr = mean(max_m_cr == r3);acc_xCr = mean(max_x_cr == r6);
acc_mHx = mean(max_m_he == r3);acc_tHx = mean(max_t_he == r4);

pred_m_cr =  mean(pred_m_cr,1);pred_x_cr =  mean(pred_x_cr,1);
pred_a_el =  mean(pred_a_el,1);pred_h_el =  mean(pred_h_el,1);
pred_m_he =  mean(pred_m_he,1);pred_t_he =  mean(pred_t_he,1);
pred_a_re =  mean(pred_a_re,1);pred_u_re =  mean(pred_u_re,1);
pred_t_sq =  mean(pred_t_sq,1);pred_x_sq =  mean(pred_x_sq,1);
pred_h_tr =  mean(pred_h_tr,1);pred_u_tr =  mean(pred_u_tr,1);
mode_mCr = mode(max_m_cr);mode_xCr = mode(max_x_cr);
mode_aEl = mode(max_a_el);mode_hEl = mode(max_h_el);
mode_mHx = mode(max_m_he);mode_tHx = mode(max_t_he);
mode_aRe = mode(max_a_re); mode_uRe = mode(max_u_re);
mode_tSq = mode(max_t_sq);mode_xSq = mode(max_x_sq);
mode_hTr = mode(max_h_tr);mode_uTr = mode(max_u_tr);

Letter_Acc = [acc_aEl;acc_aRec;acc_hEl;acc_hTr;acc_mCr;acc_mHx;acc_uTr;acc_uRec; ...
                acc_tHx;acc_tSq;acc_xCr;acc_xSq];
Letter_Mode=[mode_aEl;mode_aRe;mode_hEl;mode_hTr;mode_mCr;mode_mHx;mode_uTr;mode_uRe; ...
                mode_tHx;mode_tSq;mode_xCr;mode_xSq];
inc_l_pdr_detailed = [pred_a_el;pred_a_re;pred_h_el;pred_h_tr;pred_m_cr;pred_m_he;pred_u_tr;pred_u_re; ...
                        pred_t_he;pred_t_sq;pred_x_cr;pred_x_sq];

%% EVAL - Pseudo-Letter Incongruent
% pass data throught RBMs:
hid_out_1_d = 1./(1 + exp(-inc_pl_d*vishid_1 - repmat(hidbiases_1,size(inc_pl_d,1),1)));
rbms_pass = 1./(1 + exp(-hid_out_1_d*vishid_2 - repmat(hidbiases_2,size(hid_out_1_d,1),1)));
% Compute prediction:
pred = net(rbms_pass');
[~, max_act] = max(pred,[],2); %
[r1,~] = find(inc_pl_t');
[r2,~] = find(inc_pl_s'); 
acc_l = (max_act == r1);
acc_s = (max_act == r2);
accuracy_l = mean(acc_l);
accuracy_s = mean(acc_s);
fprintf(1,'\n Prediction / Assesment of the Congruency Effect\n');
fprintf(1,'\n ------- Case : Congruent ------- \n');
fprintf(1,'\n ------- Target : Letter ------- \n');
fprintf(1,'\n If Accuracy in classifying the inner-letter correctly = %d\n ',accuracy_l);
fprintf(1,'\n If Accuracy in classifying the outer-shape correctly = %d\n ',accuracy_s);

Shape_decisionAcc(4) = accuracy_s;
Letter_decisionAcc(4) = accuracy_l;

%%%% Extended ASSESMENT
max_a = [];max_h = [];max_m = [];max_t = [];max_u = [];max_x = [];
pred_a = [];pred_h = [];pred_m = [];pred_t = [];pred_u = [];pred_x = [];
% for more details, for each incongruent case
max_m_cr = [];max_x_cr=[];max_h_el = [];max_a_el = [];max_m_he = [];max_t_he=[];
max_a_re = [];max_u_re=[];max_t_sq = [];max_x_sq=[];max_h_tr= [];max_u_tr = [];
pred_m_cr = [];pred_x_cr = [];pred_a_el= [];pred_h_el=[];pred_m_he = [];pred_t_he=[];
pred_a_re = [];pred_u_re = [];pred_t_sq = [];pred_x_sq=[];pred_h_tr = [];pred_u_tr=[];
for i=1:size(max_act,1)
    idx_s = find(inc_l_s(i,:));
    idx_l = find(inc_l_t(i,:));
    if idx_s == 7 
        if idx_l == 3
            max_m_cr = [max_m_cr;max_act(i)];
            pred_m_cr = [pred_m_cr;pred(i,:)];
        elseif idx_l == 6
            max_x_cr = [max_x_cr;max_act(i)];
            pred_x_cr = [pred_x_cr;pred(i,:)];
        else 
            error(msgtext);
        end
        max_a = [max_a;max_act(i)];
        pred_a = [pred_a;pred(i,:)];
    elseif idx_s == 8
        if idx_l == 1
            max_a_el = [max_a_el;max_act(i)];
            pred_a_el = [pred_a_el;pred(i,:)];
        elseif idx_l == 2
            max_h_el = [max_h_el;max_act(i)];
            pred_h_el = [pred_h_el;pred(i,:)];
        else
            error(msgtext);
        end
        max_h= [max_h;max_act(i)];
        pred_h = [pred_h;pred(i,:)];
    elseif idx_s == 9
        if idx_l == 3
            max_m_he = [max_m_he;max_act(i)];
            pred_m_he = [pred_m_he;pred(i,:)];
        elseif idx_l == 4
            max_t_he = [max_t_he;max_act(i)];
            pred_t_he = [pred_t_he;pred(i,:)];
        else
            error(msgtext);
        end
        max_m= [max_m;max_act(i)];
        pred_m = [pred_m;pred(i,:)];
    elseif idx_s == 10
        if idx_l == 1
            max_a_re = [max_a_re;max_act(i)];
            pred_a_re = [pred_a_re;pred(i,:)];
        elseif idx_l == 5
            max_u_re = [max_u_re;max_act(i)];
            pred_u_re = [pred_u_re;pred(i,:)];
        else
            error(msgtext);
        end
        max_t= [max_t;max_act(i)];
        pred_t = [pred_t;pred(i,:)];
    elseif idx_s == 11
        if idx_l == 4
            max_t_sq = [max_t_sq;max_act(i)];
            pred_t_sq = [pred_t_sq;pred(i,:)];
        elseif idx_l == 6
            max_x_sq = [max_x_sq;max_act(i)];
            pred_x_sq = [pred_x_sq;pred(i,:)];
        else
            error(msgtext);
        end
        max_u= [max_u;max_act(i)];
        pred_u = [pred_u;pred(i,:)];
    elseif idx_s == 12
        if idx_l == 2
            max_h_tr = [max_h_tr;max_act(i)];
            pred_h_tr = [pred_h_tr;pred(i,:)];
        elseif idx_l == 5
            max_u_tr = [max_u_tr;max_act(i)];
            pred_u_tr = [pred_u_tr;pred(i,:)];
        else
            error(msgtext);
        end
        max_x= [max_x;max_act(i)];
        pred_x = [pred_x;pred(i,:)];
    end
end
mode_a = mode(max_a);std_a = std(max_a);prD_a =  mean(pred_a,1);
mode_h = mode(max_h);std_h = std(max_h);prD_h =  mean(pred_h,1);
mode_m = mode(max_m);std_m = std(max_m);prD_m =  mean(pred_m,1);
mode_t = mode(max_t);std_t = std(max_t);prD_t =  mean(pred_t,1);
mode_u = mode(max_u);std_u = std(max_u);prD_u =  mean(pred_u,1);
mode_x = mode(max_x);std_x = std(max_x);prD_x =  mean(pred_x,1);
r1 = ones(size(rbms_pass,1)/6,1);r2 = ones(size(rbms_pass,1)/6,1)*2;r3 = ones(size(rbms_pass,1)/6,1)*3;
r4 = ones(size(rbms_pass,1)/6,1)*4;r5 = ones(size(rbms_pass,1)/6,1)*5;r6 = ones(size(rbms_pass,1)/6,1)*6;
acc1 = (max_a == r1);% A
acc2 = (max_h == r2);% H
acc3 = (max_m == r3);% M
acc4 = (max_t == r4);% T
acc5 = (max_u == r5); % U
acc6 = (max_x == r6);% X
acc1 = mean(acc1);acc2 = mean(acc2);acc3 = mean(acc3);
acc4 = mean(acc4);acc5 = mean(acc5);acc6 = mean(acc6);
% create Table for Letters:
Targets = ["pA/Tria";"pH/Rect";"pM/Squ";"pU/Elips";"pT/Cross";"pX/Hexa"];
Acc = [acc1;acc2;acc3;acc5;acc4;acc6];
Mode = [mode_a;mode_h;mode_m;mode_u;mode_t;mode_x];
Std = [mode_x;mode_t;mode_u;mode_h;mode_a;mode_m];
incpl_ce_details = table(Targets,Mode,Std,Acc);
incpl_ce_pdr = [prD_a;prD_h;prD_m;prD_u;prD_t;prD_x];

%%% Extra Extended for Incongruent Case
r1 = ones(size(rbms_pass,1)/12,1);r2 = ones(size(rbms_pass,1)/12,1)*2;r3 = ones(size(rbms_pass,1)/12,1)*3;
r4 = ones(size(rbms_pass,1)/12,1)*4;r5 = ones(size(rbms_pass,1)/12,1)*5;r6 = ones(size(rbms_pass,1)/12,1)*6;
acc_hTr = mean(max_h_tr == r2);acc_uTr = mean(max_u_tr == r5);
acc_aRec = mean(max_a_re == r1);acc_uRec = mean(max_u_re == r5);
acc_tSq = mean(max_t_sq == r4);acc_xSq = mean(max_x_sq == r6);
acc_hEl = mean(max_h_el == r2);acc_aEl = mean(max_a_el == r1);
acc_mCr = mean(max_m_cr == r3);acc_xCr = mean(max_x_cr == r6);
acc_mHx = mean(max_m_he == r3);acc_tHx = mean(max_t_he == r4);
pred_m_cr =  mean(pred_m_cr,1);pred_x_cr =  mean(pred_x_cr,1);
pred_a_el =  mean(pred_a_el,1);pred_h_el =  mean(pred_h_el,1);
pred_m_he =  mean(pred_m_he,1);pred_t_he =  mean(pred_t_he,1);
pred_a_re =  mean(pred_a_re,1);pred_u_re =  mean(pred_u_re,1);
pred_t_sq =  mean(pred_t_sq,1);pred_x_sq =  mean(pred_x_sq,1);
pred_h_tr =  mean(pred_h_tr,1);pred_u_tr =  mean(pred_u_tr,1);
mode_mCr = mode(max_m_cr);mode_xCr = mode(max_x_cr);
mode_aEl = mode(max_a_el);mode_hEl = mode(max_h_el);
mode_mHx = mode(max_m_he);mode_tHx = mode(max_t_he);
mode_aRe = mode(max_a_re); mode_uRe = mode(max_u_re);
mode_tSq = mode(max_t_sq);mode_xSq = mode(max_x_sq);
mode_hTr = mode(max_h_tr);mode_uTr = mode(max_u_tr);

psLetter_Acc = [acc_aEl;acc_aRec;acc_hEl;acc_hTr;acc_mCr;acc_mHx;acc_uTr;acc_uRec; ...
                acc_tHx;acc_tSq;acc_xCr;acc_xSq];
psLetter_Mode=[mode_aEl;mode_aRe;mode_hEl;mode_hTr;mode_mCr;mode_mHx;mode_uTr;mode_uRe; ...
                mode_tHx;mode_tSq;mode_xCr;mode_xSq];
inc_pl_pdr_detailed = [pred_a_el;pred_a_re;pred_h_el;pred_h_tr;pred_m_cr;pred_m_he;pred_u_tr;pred_u_re; ...
                        pred_t_he;pred_t_sq;pred_x_cr;pred_x_sq];
%% store the evaluation data
Targets = ["Cong_Letter" ; "Cong_Ps-Letter"; "Inc_Letter";"Inc_Ps-Letter"];
CE_eval.overall = table(Targets,Shape_decisionAcc,Letter_decisionAcc);
CE_eval.CEs_Letter = Shape_decisionAcc(1)*100 - Shape_decisionAcc(3)*100;
CE_eval.CEs_psLetter = Shape_decisionAcc(2)*100 - Shape_decisionAcc(4)*100;

Targets = ["A/Tria";"H/Rect";"M/Squ";"U/Elips";"T/Cross";"X/Hexa"];
letter_acc = cl_ce_details_inner.Acc; letter_mode = cl_ce_details_inner.Mode;
psLetter_acc = cpl_ce_details.Acc; psLetter_mode = cpl_ce_details.Mode;
CE_eval.Congruent = table(Targets,letter_acc,letter_mode,psLetter_acc,psLetter_mode);

Targets = ["A_Elips";"A_Rec";"H_Elips";"H_Tria";"M_Cross";"M_Hexa"; ...
                "U_Tria";"U_Rec";"T_Hexa";"T_Squa";"X_Cross";"X_Squa"];
inc_whole = table(Targets,Letter_Acc,Letter_Mode,psLetter_Acc,psLetter_Mode);
CE_eval.Incongruent = inc_whole;

details.cong_l = cl_ce_details_inner;
details.cong_pl = cpl_ce_details;
details.inc_l = incl_ce_details;
details.inc_pl = incpl_ce_details;
% for pdf:
% rows = targets (A,H,M,...) && columns = possible classes
details.pdr_cong_l = cl_ce_pdr_inner;
details.pdr_cong_pl = cpl_ce_pdr;
details.pdr_inc_l = incl_ce_pdr;
details.pdr_inc_pl = incpl_ce_pdr;
details.detailed_inc_pdr_l = inc_l_pdr_detailed;
details.detailed_inc_pdr_pl = inc_pl_pdr_detailed;

CE_eval.detail = details;

%%% not relevant as far as I understood:
% CEl_Letter = Letter_decisionAcc(1)*100 - Letter_decisionAcc(3)*100;
% CEl_psLetter = Letter_decisionAcc(2)*100 - Letter_decisionAcc(4)*100;

