%%%
%   This method changed to its illiterate counterpart in which the CE was computed based
%   on the outer shapes identity: (triangle-A) <-> (triangle-H)
%   Here, it is measure based on the letter identity: (A-tr) <-> (A-sq)
%   so inner target being letter identity
%
%   Therefore, for the incongruent cases, the accuracies represent the 
%   models' hit rate on the inner target.
%%%

assert(literate);
addpath("data/new04Jl/");
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
%%% create lable array with only 1 target to ease evaluation
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
% load rbm data
load t_model DN
vishid_1 = DN.L{1,1}.vishid;
hidbiases_1 = DN.L{1,1}.hidbiases;
clear DN
load l_rbm_2.mat vishid_2 hidbiases_2

%% EVAL - Letter congruent
hid_out_1_d = 1./(1 + exp(-cong_l_d*vishid_1 - repmat(hidbiases_1,size(cong_l_d,1),1)));
rbms_pass = 1./(1 + exp(-hid_out_1_d*vishid_2 - repmat(hidbiases_2,size(hid_out_1_d,1),1)));
% Compute prediction:
ONES = ones(size(rbms_pass, 1), 1);rbms_pass = [rbms_pass ONES]; 
pred_cl = rbms_pass*weights;
softmax_pred_cl = softmax(dlarray(pred_cl','CB'));
pred_cl = extractdata(softmax_pred_cl)';
[~, max_act] = max(pred_cl,[],2); 
[r2,~] = find(cong_l_s');
acc_cong_l_s = (max_act == r2);
[r1,~] = find(cong_l_t');
acc_cong_l_t = (max_act == r1);

%% EVAL - Pseudo Letter congruent
hid_out_1_d = 1./(1 + exp(-cong_pl_d*vishid_1 - repmat(hidbiases_1,size(cong_pl_d,1),1)));
rbms_pass = 1./(1 + exp(-hid_out_1_d*vishid_2 - repmat(hidbiases_2,size(hid_out_1_d,1),1)));
ONES = ones(size(rbms_pass, 1), 1);  
rbms_pass = [rbms_pass ONES];
pred_cpl = rbms_pass*weights;
softmax_pred_cpl = softmax(dlarray(pred_cpl','CB'));
pred_cpl = extractdata(softmax_pred_cpl)';
[~, max_act] = max(pred_cpl,[],2); 
[r2,~] = find(cong_pl_s'); 
acc_cong_pl_s = (max_act == r2);
[r1,~] = find(cong_pl_t');
acc_cong_pl_t = (max_act == r1);

%% EVAL - Letter Incongruent
hid_out_1_d = 1./(1 + exp(-inc_l_d*vishid_1 - repmat(hidbiases_1,size(inc_l_d,1),1)));
rbms_pass = 1./(1 + exp(-hid_out_1_d*vishid_2 - repmat(hidbiases_2,size(hid_out_1_d,1),1)));
ONES = ones(size(rbms_pass, 1), 1);  
rbms_pass = [rbms_pass ONES];
pred_il = rbms_pass*weights;
softmax_pred_il = softmax(dlarray(pred_il','CB'));
pred_il = extractdata(softmax_pred_il)';
[~, max_act] = max(pred_il,[],2); 
[r2,~] = find(inc_l_s'); 
acc_inc_l_s = (max_act == r2);
[r1,~] = find(inc_l_t');
acc_inc_l_t = (max_act == r1);
%% EVAL - Pseudo-Letter Incongruent
hid_out_1_d = 1./(1 + exp(-inc_pl_d*vishid_1 - repmat(hidbiases_1,size(inc_pl_d,1),1)));
rbms_pass = 1./(1 + exp(-hid_out_1_d*vishid_2 - repmat(hidbiases_2,size(hid_out_1_d,1),1)));
ONES = ones(size(rbms_pass, 1), 1);  
rbms_pass = [rbms_pass ONES];
pred_ipl = rbms_pass*weights;
softmax_pred_ipl = softmax(dlarray(pred_ipl','CB'));
pred_ipl = extractdata(softmax_pred_ipl)';
[~, max_act] = max(pred_ipl,[],2); 
[r2,~] = find(inc_pl_s'); 
acc_inc_pl_s = (max_act == r2);
[r1,~] = find(inc_pl_t');
acc_inc_pl_t = (max_act == r1);

%% Store the Evaluation Data
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

% acc matrix - decision based on CONTOUR shape
accs = cat(1,acc_cong_l_s,acc_cong_pl_s);
accs = cat(1,accs,acc_inc_l_s);
accs = cat(1,accs,acc_inc_pl_s);
% acc matrix - decision based on INNER shape
accs_i = cat(1,acc_cong_l_t,acc_cong_pl_t);
accs_i = cat(1,accs_i,acc_inc_l_t);
accs_i = cat(1,accs_i,acc_inc_pl_t);
%inner-nature:
prep_l =  letter_int2str(cong_l_t);
prep_l1 = psletter_int2str(cong_pl_t);
prep_l2 = letter_int2str(inc_l_t);
prep_l3 = psletter_int2str(inc_pl_t);
Letter = cat(1,prep_l,prep_l1);
Letter = cat(1,Letter,prep_l2);
Letter = cat(1,Letter,prep_l3);
%outer:
s_targets = cat(1,cong_l_s,cong_pl_s);
s_targets = cat(1,s_targets,inc_l_s);
s_targets = cat(1,s_targets,inc_pl_s);

Subjects = repmat(subj_str,size(accs,1),1);
% to store full predictions:
A = cat(1,pred_cl(:,1),pred_cpl(:,1));
A = cat(1,A,pred_il(:,1));
A = cat(1,A,pred_ipl(:,1));
H = cat(1,pred_cl(:,2),pred_cpl(:,2));
H = cat(1,H,pred_il(:,2));
H = cat(1,H,pred_ipl(:,2));
M = cat(1,pred_cl(:,3),pred_cpl(:,3));
M = cat(1,M,pred_il(:,3));
M = cat(1,M,pred_ipl(:,3));
T = cat(1,pred_cl(:,4),pred_cpl(:,4));
T = cat(1,T,pred_il(:,4));
T = cat(1,T,pred_ipl(:,4));
U = cat(1,pred_cl(:,5),pred_cpl(:,5));
U = cat(1,U,pred_il(:,5));
U = cat(1,U,pred_ipl(:,5));
X = cat(1,pred_cl(:,6),pred_cpl(:,6));
X = cat(1,X,pred_il(:,6));
X = cat(1,X,pred_ipl(:,6));
cross = cat(1,pred_cl(:,7),pred_cpl(:,7));
cross = cat(1,cross,pred_il(:,7));
cross = cat(1,cross,pred_ipl(:,7));
elipse = cat(1,pred_cl(:,8),pred_cpl(:,8));
elipse = cat(1,elipse,pred_il(:,8));
elipse = cat(1,elipse,pred_ipl(:,8));
hexa = cat(1,pred_cl(:,9),pred_cpl(:,9));
hexa = cat(1,hexa,pred_il(:,9));
hexa = cat(1,hexa,pred_ipl(:,9));
rect = cat(1,pred_cl(:,10),pred_cpl(:,10));
rect = cat(1,rect,pred_il(:,10));
rect = cat(1,rect,pred_ipl(:,10));
square = cat(1,pred_cl(:,11),pred_cpl(:,11));
square = cat(1,square,pred_il(:,11));
square = cat(1,square,pred_ipl(:,11));
trian = cat(1,pred_cl(:,12),pred_cpl(:,12));
trian = cat(1,trian,pred_il(:,12));
trian = cat(1,trian,pred_ipl(:,12));
%
Contour = shape_int2str(s_targets);
accs = int2str(accs);
accs_i = int2str(accs_i);
if ii==1
    matrix_1 = table(Subjects,Letter,Contour,accs_i,accs);
    matrix_1_pd = table(Subjects,Letter,Contour,A,H,M,T,U,X,cross,elipse,hexa,rect,square,trian);
else
    m = table(Subjects,Letter,Contour,accs_i,accs);
    m_p =  table(Subjects,Letter,Contour,A,H,M,T,U,X,cross,elipse,hexa,rect,square,trian);
    matrix_1 = cat(1,matrix_1,m);
    matrix_1_pd = cat(1,matrix_1_pd,m_p);
end

%% matrix 3 inner
% for the illiterate method, this matrix reprensented the inner target but
% in respect to their geometrical shape, so A was treated as a triangle
% here, an A is treated as a A
Cong_Acc = cat(1,acc_cong_l_t,acc_cong_pl_t);
Inner_Inc_Acc = cat(1,acc_inc_l_t,acc_inc_pl_t);
str_target_l = letter_int2str(cong_l_t);
str_target_pl = psletter_int2str(cong_pl_t);
Letter = cat(1,str_target_l,str_target_pl);

% 1a = rect, 1b=elip, 2a=tria 2b=elip, 3a=hex, 3b=cros,4a=hex,4b=squ,
% 5a = rec 5b= trian,6a=cross, 6b= squa
inc_1a=[];inc_1b=[];inc_2a=[];inc_2b=[];
inc_3a=[];inc_3b=[];inc_4a=[];inc_4b=[];
inc_5a=[];inc_5b=[];inc_6a=[];inc_6b=[];
for i=1:size(inc_l_t)
    if find(inc_l_t(i,:)) == 1
        if find(inc_l_s(i,:)) == 10
            inc_1a=[inc_1a;acc_inc_l_t(i)];
        elseif find(inc_l_s(i,:)) == 8
            inc_1b=[inc_1b;acc_inc_l_t(i)];
        end
    end
    if find(inc_l_t(i,:)) == 2
        if find(inc_l_s(i,:)) == 12
            inc_2a=[inc_2a;acc_inc_l_t(i)];
        elseif find(inc_l_s(i,:)) == 8
            inc_2b=[inc_2b;acc_inc_l_t(i)];
        end
    end
    if find(inc_l_t(i,:)) == 3
        if find(inc_l_s(i,:)) == 9
            inc_3a=[inc_3a;acc_inc_l_t(i)];
        elseif find(inc_l_s(i,:)) == 7
            inc_3b=[inc_3b;acc_inc_l_t(i)];
        end
    end
    if find(inc_l_t(i,:)) == 4
        if find(inc_l_s(i,:)) == 9
            inc_4a=[inc_4a;acc_inc_l_t(i)];
        elseif find(inc_l_s(i,:)) == 11
            inc_4b=[inc_4b;acc_inc_l_t(i)];
        end
    end
    if find(inc_l_t(i,:)) == 5
        if find(inc_l_s(i,:)) == 10
            inc_5a=[inc_5a;acc_inc_l_t(i)];
        elseif find(inc_l_s(i,:)) == 12
            inc_5b=[inc_5b;acc_inc_l_t(i)];
        end
    end
    if find(inc_l_t(i,:)) == 6
        if find(inc_l_s(i,:)) == 7
            inc_6a=[inc_6a;acc_inc_l_t(i)];
        elseif find(inc_l_s(i,:)) == 11
            inc_6b=[inc_6b;acc_inc_l_t(i)];
        end
    end
end
letter_inc_a = [inc_1a;inc_2a;inc_3a;inc_4a;inc_5a;inc_6a];
letter_inc_b = [inc_1b;inc_2b;inc_3b;inc_4b;inc_5b;inc_6b];
% for pseudo:
inc_1a=[];inc_1b=[];inc_2a=[];inc_2b=[];
inc_3a=[];inc_3b=[];inc_4a=[];inc_4b=[];
inc_5a=[];inc_5b=[];inc_6a=[];inc_6b=[];
for i=1:size(inc_pl_t)
    if find(inc_pl_t(i,:)) == 1
        if find(inc_pl_s(i,:)) == 10
            inc_1a=[inc_1a;acc_inc_pl_t(i)];
        elseif find(inc_pl_s(i,:)) == 8
            inc_1b=[inc_1b;acc_inc_pl_t(i)];
        end
    end
    if find(inc_pl_t(i,:)) == 2
        if find(inc_pl_s(i,:)) == 12
            inc_2a=[inc_2a;acc_inc_pl_t(i)];
        elseif find(inc_pl_s(i,:)) == 8
            inc_2b=[inc_2b;acc_inc_pl_t(i)];
        end
    end
    if find(inc_pl_t(i,:)) == 3
        if find(inc_pl_s(i,:)) == 3
            inc_3a=[inc_3a;acc_inc_pl_t(i)];
        elseif find(inc_pl_s(i,:)) == 1
            inc_3b=[inc_3b;acc_inc_pl_t(i)];
        end
    end
    if find(inc_pl_t(i,:)) == 4
        if find(inc_pl_s(i,:)) == 9
            inc_4a=[inc_4a;acc_inc_pl_t(i)];
        elseif find(inc_pl_s(i,:)) == 11
            inc_4b=[inc_4b;acc_inc_pl_t(i)];
        end
    end
    if find(inc_pl_t(i,:)) == 5
        if find(inc_pl_s(i,:)) == 10
            inc_5a=[inc_5a;acc_inc_pl_t(i)];
        elseif find(inc_pl_s(i,:)) == 12
            inc_5b=[inc_5b;acc_inc_pl_t(i)];
        end
    end
    if find(inc_pl_t(i,:)) == 6
        if find(inc_pl_s(i,:)) == 7
            inc_6a=[inc_6a;acc_inc_pl_t(i)];
        elseif find(inc_pl_s(i,:)) == 11
            inc_6b=[inc_6b;acc_inc_pl_t(i)];
        end
    end
end
psletter_inc_a = [inc_1a;inc_2a;inc_3a;inc_4a;inc_5a;inc_6a];
psletter_inc_b = [inc_1b;inc_2b;inc_3b;inc_4b;inc_5b;inc_6b];
% 1a = rect, 1b=elip, 2a=tria 2b=elip, 3a=hex, 3b=cros,4a=hex,4b=squ,
% 5a = rec 5b= trian,6a=cross, 6b= squa
a1 = repmat("rectangle",100,1);
a2 = repmat("triangle",100,1);
a3 = repmat("hexagon",100,1);
a4= repmat("hexagon",100,1);
a5= repmat("rectangle",100,1);
a6= repmat("cross",100,1);
b1 = repmat("elipse",100,1);
b2 = repmat("elipse",100,1);
b3 = repmat("cross",100,1);
b4= repmat("square",100,1);
b5= repmat("triangle",100,1);
b6= repmat("square",100,1);
a = [a1;a2;a3;a4;a5;a6;a1;a2;a3;a4;a5;a6];
b = [b1;b2;b3;b4;b5;b6;b1;b2;b3;b4;b5;b6];

Acc_inc_a = [letter_inc_a;psletter_inc_a];
Acc_inc_b= [letter_inc_b;psletter_inc_b];
Subjects = repmat(subj_str,size(Acc_inc_a,1),1);

if ii==1
    matrix_3 = table(Subjects,Letter,Cong_Acc,Acc_inc_a,Acc_inc_b,a,b);
else
    m3i = table(Subjects,Letter,Cong_Acc,Acc_inc_a,Acc_inc_b,a,b);
    matrix_3 = cat(1,matrix_3,m3i);
end

%% functions
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

function [str_shape] = shape_int2str(shape_id)
    str_shape = strings(size(shape_id,1),1);
    for i=1:size(shape_id,1)
        if find(shape_id(i,:)) == 7
            str_shape(i) = "cross";
        elseif find(shape_id(i,:)) == 8
            str_shape(i) = "elips";
        elseif find(shape_id(i,:)) == 9
            str_shape(i) = "hexagon";
        elseif find(shape_id(i,:)) == 10 
            str_shape(i) = "rectangle";
        elseif find(shape_id(i,:)) == 11
            str_shape(i) = "square";
        elseif find(shape_id(i,:)) == 12
            str_shape(i) = "triangle";
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


