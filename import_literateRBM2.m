%% concatenate shape and letter-in-string data:

%%% Percentage of shape data?, for equal part: perc = 1
perc = 1;
%%%

load openCV_letterPOS3.mat data target_l target_s
xdata = zeros(size(data));
for i=1:size(data,1)
    xdata(i,:) = reshape(im2double(reshape(data(i,:),[40 40 1])), [1 1600]);
end
target_l_s = double(target_s);
target_l = double(target_l);

load openCV_shapePOS3.mat data target_s
shapedata = zeros(size(data));
for i=1:size(data,1)
    shapedata(i,:) = reshape(im2double(reshape(data(i,:),[40 40 1])), [1 1600]);
end
data = shapedata;
target_s = double(target_s);

% concatenate
total_data = [data;xdata];

% batchsize for training set
batchsize = 12;
% now create the batches
totnum = size(total_data,1);
fprintf(1, 'Size of the G&L Training Dataset= %d \n', totnum);
rand('state',0); %so we know the permutation of the training data
randomorder = randperm(totnum);
numdims  =  size(data,2);
numbatches= totnum/g_batchsize;
batchdata = zeros(batchsize, numdims, numbatches);
batchtargets = zeros(batchsize, batchsize, numbatches);


%%% class 1 -> 6 for letters
class1_data = [];class2_data = [];class3_data = [];
class4_data = [];class5_data = [];class6_data = [];

for i=1:size(xdata,1)
    if find(target_l_s(i,:)) == 1
        class1_data = [class1_data;xdata(i,:)];
    end
    if find(target_l_s(i,:)) == 2
        class2_data = [class2_data;xdata(i,:)];
    end
    if find(target_l_s(i,:)) == 3
        class3_data = [class3_data;xdata(i,:)];
    end
    if find(target_l_s(i,:)) == 4
        class4_data = [class4_data;xdata(i,:)];
    end
    if find(target_l_s(i,:)) == 5
        class5_data = [class5_data;xdata(i,:)];
    end
    if find(target_l_s(i,:)) == 6
        class6_data = [class6_data;xdata(i,:)];
    end
end

%%% class 7 -> 12 for shapes
class7_data = [];class8_data = [];class9_data = [];
class10_data = [];class11_data = [];class12_data = [];

for i=1:size(data,1)
    if find(target_s(i,:)) == 1
        class7_data = [class7_data;data(i,:)];
    end
    if find(target_s(i,:)) == 2
        class8_data = [class8_data;data(i,:)];
    end
    if find(target_s(i,:)) == 3
        class9_data = [class9_data;data(i,:)];
    end
    if find(target_s(i,:)) == 4
        class10_data = [class10_data;data(i,:)];
    end
    if find(target_s(i,:)) == 5
        class11_data = [class11_data;data(i,:)];
    end
    if find(target_s(i,:)) == 6
        class12_data = [class12_data;data(i,:)];
    end
end

%%% now its a 12 class problem:
g_class_size = totnum/12;
class_b_M_init = zeros(batchsize,numdims);
targets_M_init = zeros(batchsize,12);
num = g_batchsize/12;
for b=1:numbatches
    % as hinton suggest, we want equal number of elemts per class in each b
    class_b_M_init(1:num,:) = class1_data(1 +(b-1)*num:b*num,:);
    targets_M_init(1:num,:) = repmat([1 0 0 0 0 0 0 0 0 0 0 0],num,1);
    class_b_M_init(1 + num:num*2,:) = class2_data(1 +(b-1)*num:b*num,:);
    targets_M_init(1 + num:num*2,:) = repmat([0 1 0 0 0 0 0 0 0 0 0 0],num,1);
    class_b_M_init(1 + num*2:num*3,:) = class3_data(1 +(b-1)*num:b*num,:);
    targets_M_init(1 + num*2:num*3,:) = repmat([0 0 1 0 0 0 0 0 0 0 0 0],num,1);
    class_b_M_init(1 + num*3:num*4,:) = class4_data(1 +(b-1)*num:b*num,:);
    targets_M_init(1 + num*3:num*4,:) = repmat([0 0 0 1 0 0 0 0 0 0 0 0],num,1);
    class_b_M_init(1 + num*4:num*5,:) = class5_data(1 +(b-1)*num:b*num,:);
    targets_M_init(1 + num*4:num*5,:) = repmat([0 0 0 0 1 0 0 0 0 0 0 0],num,1);
    class_b_M_init(1 + num*5:num*6,:) = class6_data(1 +(b-1)*num:b*num,:);
    targets_M_init(1 + num*5:num*6,:) = repmat([0 0 0 0 0 1 0 0 0 0 0 0],num,1);

    class_b_M_init(1 + num*6:num*7,:) = class7_data(1 +(b-1)*num:b*num,:);
    targets_M_init(1 + num*6:num*7,:) = repmat([0 0 0 0 0 0 1 0 0 0 0 0],num,1);
    class_b_M_init(1 + num*7:num*8,:) = class8_data(1 +(b-1)*num:b*num,:);
    targets_M_init(1 + num*7:num*8,:) = repmat([0 0 0 0 0 0 0 1 0 0 0 0],num,1);
    class_b_M_init(1 + num*8:num*9,:) = class9_data(1 +(b-1)*num:b*num,:);
    targets_M_init(1 + num*8:num*9,:) = repmat([0 0 0 0 0 0 0 0 1 0 0 0],num,1);
    class_b_M_init(1 + num*9:num*10,:) = class10_data(1 +(b-1)*num:b*num,:);
    targets_M_init(1 + num*9:num*10,:) = repmat([0 0 0 0 0 0 0 0 0 1 0 0],num,1);
    class_b_M_init(1 + num*10:num*11,:) = class11_data(1 +(b-1)*num:b*num,:);
    targets_M_init(1 + num*10:num*11,:) = repmat([0 0 0 0 0 0 0 0 0 0 1 0],num,1);
    class_b_M_init(1 + num*11:num*12,:) = class12_data(1 +(b-1)*num:b*num,:);
    targets_M_init(1 + num*11:num*12,:) = repmat([0 0 0 0 0 0 0 0 0 0 0 1],num,1);

    %shuffle -- careful; currently:default seed
    idx = randperm(g_batchsize);
    class_b_M = class_b_M_init(idx,:);
    targets_M = targets_M_init(idx,:);    
    batchdata(:,:,b) = class_b_M;
    batchtargets(:,:,b) = targets_M;
end
% the following works in precision because of the uniform batches

% for the letter case, it bears a random factor but is still representative
% if g_batchsize == 12 ||g_batchsize == 36 ||g_batchsize == 48

val_idx = floor(numbatches*0.15);

% elseif g_batchsize == 24 % works for the 6class problem
%     val_idx = floor(numbatches*0.12); %works for numcases=24
% end

g_val_data = batchdata(:,:,(numbatches-val_idx+1):numbatches);
g_val_target = batchtargets(:,:,(numbatches-val_idx+1):numbatches);
% unfold for "overfitting" method in "rbm2"
v_d = [];v_t= [];
for i=1:size(g_val_data,3)  % I prefer to use a loop over "reshape" for now
    v_d = [v_d; g_val_data(:,:,i)];
    v_t = [v_t; g_val_target(:,:,i)];
end
val_data = v_d;val_target = v_t;
batchdata = batchdata(:,:, 1:(numbatches-val_idx));
batchtargets = batchtargets(:,:, 1:(numbatches-val_idx));

save g&L_batchdata_m12 batchdata batchtargets val_data val_targets

