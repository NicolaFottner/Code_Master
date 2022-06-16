% 1. import data (geometrical shapes) from local directory
% 2. create "mini-batches"
% equivalent to the concatenation of the methods "converter.m" and "make_batches.m" from the original code

load openCV_final_Shapes.mat data;
%load openCV_Shapes_xtra_balanced.mat data;

shapedata = zeros(size(data));
for i=1:size(data,1)
    shapedata(i,:) = reshape(im2double(reshape(data(i,:),[40 40 1])), [1 1600]);
end
% imshow(reshape(shapedata(45,:),[40 40 1])');
% imshow(reshape(data(45,:),[40 40 1])');
clear data;
num_classes = geo_shape_class;
% now create the batches
g_totnum=size(shapedata,1);
fprintf(1, 'Size of the geo-shape training dataset= %5d \n', g_totnum);
rand('state',0); %so we know the permutation of the training data
randomorder=randperm(g_totnum);
numdims  =  size(shapedata,2);
numbatches=g_totnum/g_batchsize;
g_batchdata = zeros(g_batchsize, numdims, numbatches);
g_batchtargets = zeros(g_batchsize, num_classes, numbatches);

class1_data = shapedata(3004:4002,:);
class2_data = shapedata(4003:5001,:);
class3_data = shapedata(7:1005,:);
class4_data = shapedata(1006:2004,:);
class5_data = shapedata(5002:6000,:);
class6_data = shapedata(2005:3003,:);
class1_data = [class1_data ; shapedata(4,:)];
class2_data = [class2_data ; shapedata(5,:)];
class3_data = [class3_data ; shapedata(1,:)];
class4_data = [class4_data ; shapedata(2,:)];
class5_data = [class5_data ; shapedata(6,:)];
class6_data = [class6_data ; shapedata(3,:)];
g_class_size=g_totnum/6;
class_b_M_init = zeros(g_batchsize,numdims);
targets_M_init = zeros(g_batchsize,6);
    for b=1:numbatches
        num = g_batchsize/6;
        %as hinton suggest, we want equal number of elemts per class in each b
        class_b_M_init(1:num,:) = class1_data(1 +(b-1)*num:b*num,:);
        targets_M_init(1:num,:) = repmat([1 0 0 0 0 0],num,1);
        class_b_M_init(1 + num:num*2,:) = class2_data(1 +(b-1)*num:b*num,:);
        targets_M_init(1 + num:num*2,:) = repmat([0 1 0 0 0 0],num,1);
        class_b_M_init(1 + num*2:num*3,:) = class3_data(1 +(b-1)*num:b*num,:);
        targets_M_init(1 + num*2:num*3,:) = repmat([0 0 1 0 0 0],num,1);
        class_b_M_init(1 + num*3:num*4,:) = class4_data(1 +(b-1)*num:b*num,:);
        targets_M_init(1 + num*3:num*4,:) = repmat([0 0 0 1 0 0],num,1);
        class_b_M_init(1 + num*4:num*5,:) = class5_data(1 +(b-1)*num:b*num,:);
        targets_M_init(1 + num*4:num*5,:) = repmat([0 0 0 0 1 0],num,1);
        class_b_M_init(1 + num*5:num*6,:) = class6_data(1 +(b-1)*num:b*num,:);
        targets_M_init(1 + num*5:num*6,:) = repmat([0 0 0 0 0 1],num,1);
        %shuffle -- careful; currently:default seed
        idx = randperm(g_batchsize);
        class_b_M = class_b_M_init(idx,:);
        targets_M = targets_M_init(idx,:);    
        g_batchdata(:,:,b) = class_b_M;
        g_batchtargets(:,:,b) = targets_M;
    end
% the following works in precision because of the uniform batches
% for the letter case, it bears a random factor but is still representative
if g_batchsize == 6|| g_batchsize == 12 ||g_batchsize == 30||g_batchsize == 60
    val_idx = floor(numbatches*0.15);
elseif g_batchsize == 24 % works for the 6class problem
    val_idx = floor(numbatches*0.12); %works for numcases=24
end
g_val_data = g_batchdata(:,:,(numbatches-val_idx+1):numbatches);
g_val_target = g_batchtargets(:,:,(numbatches-val_idx+1):numbatches);
% unfold for "overfitting" method in "rbm2"
v_d = [];v_t= [];
for i=1:size(g_val_data,3)  % I prefer to use a loop over "reshape" for now
    v_d = [v_d; g_val_data(:,:,i)];
    v_t = [v_t; g_val_target(:,:,i)];
end
g_val_data = v_d;g_val_target = v_t;
g_batchdata = g_batchdata(:,:, 1:(numbatches-val_idx));
g_batchtargets = g_batchtargets(:,:, 1:(numbatches-val_idx));
clear shapedata targets;
