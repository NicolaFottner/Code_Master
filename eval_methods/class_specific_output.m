% class_specific_output; computed on the test set
dd = strsplit(date,'-');clean_date = strcat(dd(1),dd(2)); %without "-YY
weightsx = W2;
index = floor(size(g_batchtargets,1)*p); % p = percentage of test data
test_l =  g_batchtargets(1:index,:); % here 'l' for 'labels'

if numhid3 == 0
    eval_data = hid_out_2(1:index, :); % basically like the test set from before
    str_hid3 = "";
else
    eval_data = hid_out_3(1:index, :);
    str_hid3 = "three/";
end
ONES = ones(size(eval_data, 1), 1);
eval_data = [eval_data ONES];
pred = eval_data*weightsx;
softmax_pred = softmax(dlarray(pred','CB'));
pred = extractdata(softmax_pred)';
[~, max_act] = max(pred,[],2);

if geo_shape_class == 6
    %% In 6 class problem -- Illiterate Model
    pred_1 = [];pred_2 = [];pred_3 = [];pred_4 = [];pred_5 = [];pred_6 = [];
    max_1 = [];max_2 = [];max_3 = [];max_4 = [];max_5 = [];max_6 = [];
    for i=1:size(max_act,1)
        idx_l = find(test_l(i,:));
        if idx_l == 1 
            pred_1 = [pred_1;pred(i,:)];
            max_1 = [max_1;max_act(i,:)];
        elseif idx_l == 2
            pred_2 = [pred_2;pred(i,:)];
            max_2 = [max_2;max_act(i,:)];
        elseif idx_l == 3
            pred_3 = [pred_3;pred(i,:)];
            max_3 = [max_3;max_act(i,:)];
        elseif idx_l == 4
            pred_4 = [pred_4;pred(i,:)];
            max_4 = [max_4;max_act(i,:)];
        elseif idx_l == 5
            pred_5 = [pred_5;pred(i,:)];
            max_5 = [max_5;max_act(i,:)];
        elseif idx_l == 6
            pred_6 = [pred_6;pred(i,:)];
            max_6 = [max_6;max_act(i,:)];
        end
    end  
%     f = figure;
%     %:
%     subplot(2,3,1);
%     prD_pA =  mean(pred_1,1);
%     bar(prD_pA);
%     xlabel('Cross');
%     %:
%     subplot(2,3,2);
%     prD_pH =  mean(pred_2,1);
%     bar(prD_pH);
%     xlabel('Elipse');
%     %:
%     subplot(2,3,3);
%     prD_pM =  mean(pred_3,1);
%     bar(prD_pM);
%     xlabel('Hexagon');
%     %:
%     subplot(2,3,4);
%     prD_pU =  mean(pred_4,1);
%     bar(prD_pU);
%     xlabel('Rectangle');
%     %:
%     subplot(2,3,5);
%     prD_pT =  mean(pred_5,1);
%     bar(prD_pT);
%     xlabel('Square');
%     %:
%     subplot(2,3,6);
%     prD_pX =  mean(pred_6,1);
%     bar(prD_pX);
%     xlabel('Triangle');
%     sgtitle("Day: " + clean_date + ", Models' Prediction/Prob Distr: ");
%     file_name = "plots_results/classf_perf6/details/" + str_hid3 + clean_date + "_" + int2str(c(4)) + "h" + int2str(c(5))+"m_"+"detail_distr" + ".pdf";
%     exportgraphics(f,file_name);
    r1 = ones(size(test_l,1)/6,1);r2 = ones(size(test_l,1)/6,1)*2;r3 = ones(size(test_l,1)/6,1)*3;
    r4 = ones(size(test_l,1)/6,1)*4;r5 = ones(size(test_l,1)/6,1)*5;r6 = ones(size(test_l,1)/6,1)*6;
    acc1 = (max_1 == r1);acc2 = (max_2 == r2);acc3 = (max_3 == r3);
    acc4 = (max_4 == r4);acc5 = (max_5 == r5);acc6 = (max_6 == r6); 
    acc1 = mean(acc1);acc2 = mean(acc2);acc3 = mean(acc3);
    acc4 = mean(acc4);acc5 = mean(acc5);acc6 = mean(acc6);
%     filename = "plots_results/classf_perf6/details/" + str_hid3 + clean_date + "_" + int2str(c(4)) + "h" + int2str(c(5))+"m_" + "detail_acc";
%     save(filename,'acc1','acc2','acc3','acc4', 'acc5','acc6','numhid2','numhid3','final_epoch');
%     
    Output = ["1";"2";"3";"4";"5";"6"];
    Accuracy = [acc1;acc2;acc3;acc4;acc5;acc6];
    Classifier_Details = table(Output,Accuracy);

else % when geo_class  == 12

    %% In 12 class problem  --- Literate model

    %%% pred 1 -> 6, for letter identities
    %   1=A,2=H,3=M,4=U,5=T,6=X
    %%% pred 7 -> 12, for shape identities
    %   7=cross,8=elipse,9=hexa,10=rect,11=squr,12=triangle

    pred_1 = [];pred_2 = [];pred_3 = [];pred_4 = [];pred_5 = [];pred_6 = [];
    pred_7 = [];pred_8 = [];pred_9 = [];pred_10 = [];pred_11 = [];pred_12 = [];
    max_1 = [];max_2 = [];max_3 = [];max_4 = [];max_5 = [];max_6 = [];
    max_7 = [];max_8 = [];max_9 = [];max_10 = [];max_11 = [];max_12 = [];
    for i=1:size(max_act,1)
        idx_l = find(test_l(i,:));
        if idx_l == 1 
            pred_1 = [pred_1;pred(i,:)];
            max_1 = [max_1;max_act(i,:)];
        elseif idx_l == 2
            pred_2 = [pred_2;pred(i,:)];
            max_2 = [max_2;max_act(i,:)];
        elseif idx_l == 3
            pred_3 = [pred_3;pred(i,:)];
            max_3 = [max_3;max_act(i,:)];
        elseif idx_l == 4
            pred_4 = [pred_4;pred(i,:)];
            max_4 = [max_4;max_act(i,:)];
        elseif idx_l == 5
            pred_5 = [pred_5;pred(i,:)];
            max_5 = [max_5;max_act(i,:)];
        elseif idx_l == 6
            pred_6 = [pred_6;pred(i,:)];
            max_6 = [max_6;max_act(i,:)];
       elseif idx_l == 7
            pred_7 = [pred_7;pred(i,:)];
            max_7 = [max_7;max_act(i,:)];
        elseif idx_l == 8
            pred_8 = [pred_8;pred(i,:)];
            max_8 = [max_8;max_act(i,:)];
        elseif idx_l == 9
            pred_9 = [pred_9;pred(i,:)];
            max_9 = [max_9;max_act(i,:)];
        elseif idx_l == 10
            pred_10 = [pred_10;pred(i,:)];
            max_10 = [max_10;max_act(i,:)];
        elseif idx_l == 11
            pred_11 = [pred_11;pred(i,:)];
            max_11 = [max_11;max_act(i,:)];
        elseif idx_l == 12
            pred_12 = [pred_12;pred(i,:)];
            max_12 = [max_12;max_act(i,:)];
        end
    end  
    f = figure;
    %:
    subplot(4,3,1);
    prD_A =  mean(pred_1,1);
    bar(prD_A);
    xlabel('A');
    %:
    subplot(4,3,2);
    prD_H =  mean(pred_2,1);
    bar(prD_H);
    xlabel('H');
    %:
    subplot(4,3,3);
    prD_M =  mean(pred_3,1);
    bar(prD_M);
    xlabel('M');
    %:
    subplot(4,3,4);
    prD_U =  mean(pred_4,1);
    bar(prD_U);
    xlabel('T');
    %:
    subplot(4,3,5);
    prD_T =  mean(pred_5,1);
    bar(prD_T);
    xlabel('U');
    %:
    subplot(4,3,6);
    prD_X =  mean(pred_6,1);
    bar(prD_X);
    xlabel('X');
    %:%%%%%%%%%%%:%%%%%%%%%%%:%%%%%%%%%%%:%%%%%%%%%%%:%%%%%%%%%%
    subplot(4,3,7);
    prD_7 =  mean(pred_7,1);
    bar(prD_7);
    xlabel('Cross');
    %:
    subplot(4,3,8);
    prD_8 =  mean(pred_8,1);
    bar(prD_8);
    xlabel('Elipse');
    %:
    subplot(4,3,9);
    prD_9 =  mean(pred_9,1);
    bar(prD_9);
    xlabel('Hexagon');
    %:
    subplot(4,3,10);
    prD_10 =  mean(pred_10,1);
    bar(prD_10);
    xlabel('Rectangle');
    %:
    subplot(4,3,11);
    prD_11 =  mean(pred_11,1);
    bar(prD_11);
    xlabel('Square');
    %:
    subplot(4,3,12);
    prD_12 =  mean(pred_12,1);
    bar(prD_12);
    xlabel('Triangle');

    sgtitle("Day: " + clean_date + ", Models' Prediction/Prob Distr: ");
    file_name = "Evals/plots/"+ "lit" + clean_date + "_" + int2str(c(4)) + "h" + int2str(c(5))+"m_"+"cl_detail_distr" + ".pdf";
    exportgraphics(f,file_name);
    r1 = ones(size(test_l,1)/12,1);r2 = ones(size(test_l,1)/12,1)*2;r3 = ones(size(test_l,1)/12,1)*3;
    r4 = ones(size(test_l,1)/12,1)*4;r5 = ones(size(test_l,1)/12,1)*5;r6 = ones(size(test_l,1)/12,1)*6;
    r7 = ones(size(test_l,1)/12,1)*7;r8 = ones(size(test_l,1)/12,1)*8;r9 = ones(size(test_l,1)/12,1)*9;
    r10 = ones(size(test_l,1)/12,1)*10;r11 = ones(size(test_l,1)/12,1)*11;r12 = ones(size(test_l,1)/12,1)*12;
    acc1 = (max_1 == r1);acc2 = (max_2 == r2);acc3 = (max_3 == r3);
    acc4 = (max_4 == r4);acc5 = (max_5 == r5);acc6 = (max_6 == r6);
    acc7 = (max_7 == r7);acc8 = (max_8 == r8);acc9 = (max_9 == r9);
    acc10 = (max_10 == r10);acc11 = (max_11 == r11);acc12 = (max_12 == r12); 
    acc1 = mean(acc1);acc2 = mean(acc2);acc3 = mean(acc3);
    acc4 = mean(acc4);acc5 = mean(acc5);acc6 = mean(acc6);
    acc7 = mean(acc7);acc8 = mean(acc8);acc9 = mean(acc9);
    acc10 = mean(acc10);acc11 = mean(acc11);acc12 = mean(acc12);
%     filename = "plots_results/classf_perf/details/"+ str_hid3  + clean_date + "_" + int2str(c(4)) + "h" + int2str(c(5))+"m_" + "detail_acc";
%     save(filename,'acc1','acc2','acc3','acc4', 'acc5','acc6','acc7','acc8','acc9','acc10', 'acc11','acc12','numhid2','numhid3','final_epoch');

    Output = ["A";"H";"M";"T";"U";"X";"Cross";"Elipse";"Hexagon";"Rectangle";"Square";"Triangle"];
    Accuracy = [acc1;acc2;acc3;acc4;acc5;acc6;acc7;acc8;acc9;acc10;acc11;acc12];
    Classifier_Details = table(Output,Accuracy);

    fprintf(1,'\n Its detailed classification: \n');
    fprintf(1,'\n A =  %d\n',acc1);
    fprintf(1,'\n H =  %d\n',acc2);
    fprintf(1,'\n M =  %d\n',acc3);
    fprintf(1,'\n T =  %d\n',acc4);
    fprintf(1,'\n U =  %d\n',acc5);
    fprintf(1,'\n X =  %d\n\n',acc6);
    fprintf(1,'\n Cross =  %d\n',acc7);
    fprintf(1,'\n Elipse =  %d\n',acc8);
    fprintf(1,'\n Hexagon =  %d\n',acc9);
    fprintf(1,'\n Rectangle =  %d\n',acc10);
    fprintf(1,'\n Square =  %d\n',acc11);
    fprintf(1,'\n Triangle =  %d\n\n',acc12);
end
