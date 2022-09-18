% Computing performance -- identification of letters based on their
% letter identity 
assert(literate);
dd = strsplit(date,'-');c = clock;clean_date = strcat(dd(1),dd(2)); %without "-YYYY"
strh3 ="";
%load openCV_letterPOS3.mat data target_s target_l;


load openCV_letterPOS1.mat
load saved_models/literate_models/14Jul_19h59m_lit_LS_n1.mat
load t_model DN
vishid_1 = DN.L{1,1}.vishid;
hidbiases_1 = DN.L{1,1}.hidbiases;
visbiases_1 = DN.L{1,1}.visbiases;
clear DN
numhid3=0;
least_square=1;

d_double = zeros(size(data));
for i=1:size(data,1)
    d_double(i,:) = reshape(im2double(reshape(data(i,:),[40 40 1])), [1 1600]);
end
clear data; data = d_double;
target_s= double(target_s);
target_l = double(target_l);
z = zeros(size(target_s));
target_s = cat(2,z,target_s);

% pass letterdata throught RBMs:
hid_out_1_l = 1./(1 + exp(-data*vishid_1 - repmat(hidbiases_1,size(data,1),1)));
rbms_pass_l = 1./(1 + exp(-hid_out_1_l*vishid_2 - repmat(hidbiases_2,size(hid_out_1_l,1),1)));
if numhid3 ~= 0
    rbms_pass_l = 1./(1 + exp(-rbms_pass_l*vishid_3 - repmat(hidbiases_3,size(rbms_pass_l,1),1)));
end

if least_square
    str_ls = "LS";
    weights = W2;
    ONES = ones(size(rbms_pass_l, 1), 1);  
    rbms_pass_l = [rbms_pass_l ONES];
    pred1 = rbms_pass_l*weights;
    softmax_pred1 = softmax(dlarray(pred1','CB'));
    pred1 = extractdata(softmax_pred1)';
else
    str_ls = "MLP";
    pred1 = net(rbms_pass_l')';
end

[~, max_act_l] = max(pred1,[],2);
if literate
    [r1,~] = find(target_l'); 
else
    [r1,~] = find(target_s'); 
end
acc_l = (max_act_l == r1);
accuracy_l = mean(acc_l);
%loss?
l_loss = extractdata(crossentropy(softmax_pred1,target_s'));


fprintf(1,'\n Identification of Letter based on Letter Identity');
fprintf(1,'\n Accuracy = %d ',accuracy_l);


%%%% Extended ASSESMENT
%% Letters
% (assement of identification based on their "congruent"shape)
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
% carefull, here for some reason, t<->acc5 and u<->acc4, ... @todo fix it 
acc1 = (max_a == r1);acc2 = (max_h == r2);acc3 = (max_m == r3);
acc4 = (max_u == r5);acc5 = (max_t == r4);acc6 = (max_x == r6);
acc1 = mean(acc1);acc2 = mean(acc2);acc3 = mean(acc3);
acc4 = mean(acc4);acc5 = mean(acc5);acc6 = mean(acc6);
% create Table for Letters:
Targets = ["A" ; "H"; "M"; "U"; "T"; "X"];
Mode = [mode_A;mode_H;mode_M;mode_U;mode_T;mode_X];
Std = [std_A;std_H;std_M;std_U;std_T;std_X];
Acc = [acc1;acc2;acc3;acc4;acc5;acc6];
table_3letter = table(Targets,Mode,Std,Acc);
letter3_pdr = [prD_A;prD_H;prD_M;prD_U;prD_T;prD_X];

f = figure;
%:
subplot(2,3,1);
bar(prD_A);
xlabel('A');
%:
subplot(2,3,2);
bar(prD_H);
xlabel('H');
%:
subplot(2,3,3);
bar(prD_M);
xlabel('M');
%:
subplot(2,3,4);
bar(prD_T);
xlabel('T');
%:
subplot(2,3,5);
bar(prD_U);
xlabel('U');
%:
subplot(2,3,6);
bar(prD_X);
xlabel('X');
sgtitle("Day: " + clean_date + ", Models' Prediction/Prob Distr: ");
file_name = "Evals/plots/"+ "lit_" +int2str(ii) + "_"+ str_ls + "_"+clean_date +"_Lpos3_distr" + ".pdf";
exportgraphics(f,file_name);


