% for 12 class problem:

% 1. import data (geometrical shapes) from local directory
% 2. create "mini-batches"
% equivalent to the concatenation of the methods "converter.m" and "make_batches.m" from the original code

% importing img from .png files
%geo_shapes;

if importcv == 1 && ~use_t_two
    load shapedata_opencv12n.mat data target;
    shapedata = data(13:12012,:); % only if new dataset is used
    targets = target(13:12012,:);
    
    
    for i=1:size(shapedata,2)
        shapedata(i,:)= shapedata(i,:)./255;
    end


    %     imshow(reshape(shapedata(1,:), [40 40 1])');
    %     x = reshape(shapedata(1,:), [40 40 1])';
    %     imshow(x);
    %     x = reshape(x,[1600 1]);
    %     x = reshape (x,[40 40 1]);
    %     imshow(x);

    %shapedata = data;
    clear data;
    clear target;
% elseif ~use_t_two  %if numclasses=6
%     load shapedata_opencv.mat data target;
%     shapedata = data;
%     targets = target;
%     clear data;
%     clear target;
else 
    load testolin/LettGR-whit.mat Letters
    shapedata = Letters.inputdata;
    targets = Letters.dataindexes;
end

%imshow(reshape(shapedata(1,:), [40 40 1])');

new = true;
letters_class = 26;

if ~use_t_two
    num_classes = geo_shape_class;
else
    num_classes = letters_class;
end
% now create the batches
g_totnum=size(shapedata,1);
fprintf(1, 'Size of the geo-shape training dataset= %5d \n', g_totnum);
rand('state',0); %so we know the permutation of the training data
randomorder=randperm(g_totnum);
numdims  =  size(shapedata,2);
numbatches=g_totnum/g_batchsize;
g_batchdata = zeros(g_batchsize, numdims, numbatches);
if ~use_t_two
    g_batchtargets = zeros(g_batchsize, num_classes, numbatches);
else
    g_batchtargets = zeros(g_batchsize, 1, numbatches);
end 
    
if ~use_t_two
    
    if new
        class1_data = shapedata(7001:8000,:);
        class2_data = shapedata(10001:11000,:);
        class3_data = shapedata(1001:2000,:);
        class4_data = shapedata(2001:3000,:);
        class5_data = shapedata(11001:12000,:);
        class6_data = shapedata(3001:4000,:);
        class7_data = shapedata(8001:9000,:);
        class8_data = shapedata(6001:7000,:);
        class9_data = shapedata(4001:5000,:);
        class10_data = shapedata(5001:6000,:);
        class11_data = shapedata(9001:10000,:);
        class12_data = shapedata(1:1000,:);
    else
        class1_data = [shapedata(1:100,:);shapedata(1201:2100,:)];
        class2_data = [shapedata(101:200,:);shapedata(2101:3000,:)];
        class3_data = [shapedata(201:300,:);shapedata(3001:3900,:)];
        class4_data = [shapedata(301:400,:);shapedata(3901:4800,:)];
        class5_data = [shapedata(401:500,:);shapedata(4801:5700,:)];
        class6_data = [shapedata(501:600,:);shapedata(5701:6600,:)];
        class7_data = [shapedata(601:700,:);shapedata(6601:7500,:)];
        class8_data = [shapedata(701:800,:);shapedata(7501:8400,:)];
        class9_data = [shapedata(801:900,:);shapedata(8401:9300,:)];
        class10_data = [shapedata(901:1000,:);shapedata(9301:10200,:)];
        class11_data = [shapedata(1001:1100,:);shapedata(10201:11100,:)];
        class12_data = [shapedata(1101:1200,:);shapedata(11101:12000,:)];
        
    end
    g_class_size=g_totnum/12;
    class_b_M_init = zeros(g_batchsize,numdims);
    targets_M_init = zeros(g_batchsize,12);
        for b=1:numbatches
            num = g_batchsize/12;
            %as hinton suggest, we want equal number of elemts per class in each b
            class_b_M_init(1:num,:) = class1_data(1 +(b-1)*num:b*num,:);
            targets_M_init(1:num,:) = [1 0 0 0 0 0 0 0 0 0 0 0];
            class_b_M_init(1 + num:num*2,:) = class2_data(1 +(b-1)*num:b*num,:);
            targets_M_init(1 + num:num*2,:) = [0 1 0 0 0 0 0 0 0 0 0 0];
            class_b_M_init(1 + num*2:num*3,:) = class3_data(1 +(b-1)*num:b*num,:);
            targets_M_init(1 + num*2:num*3,:) = [0 0 1 0 0 0 0 0 0 0 0 0];
            class_b_M_init(1 + num*3:num*4,:) = class4_data(1 +(b-1)*num:b*num,:);
            targets_M_init(1 + num*3:num*4,:) = [0 0 0 1 0 0 0 0 0 0 0 0];
            class_b_M_init(1 + num*4:num*5,:) = class5_data(1 +(b-1)*num:b*num,:);
            targets_M_init(1 + num*4:num*5,:) = [0 0 0 0 1 0 0 0 0 0 0 0];
            class_b_M_init(1 + num*5:num*6,:) = class6_data(1 +(b-1)*num:b*num,:);
            targets_M_init(1 + num*5:num*6,:) = [0 0 0 0 0 1 0 0 0 0 0 0];
            class_b_M_init(1 + num*6:num*7,:) = class7_data(1 +(b-1)*num:b*num,:);
            targets_M_init(1 + num*6:num*7,:) = [0 0 0 0 0 0 1 0 0 0 0 0];
            class_b_M_init(1 + num*7:num*8,:) = class8_data(1 +(b-1)*num:b*num,:);
            targets_M_init(1 + num*7:num*8,:) = [0 0 0 0 0 0 0 1 0 0 0 0];
            class_b_M_init(1 + num*8:num*9,:) = class9_data(1 +(b-1)*num:b*num,:);
            targets_M_init(1 + num*8:num*9,:) = [0 0 0 0 0 0 0 0 1 0 0 0];
            class_b_M_init(1 + num*9:num*10,:) = class10_data(1 +(b-1)*num:b*num,:);
            targets_M_init(1 + num*9:num*10,:) = [0 0 0 0 0 0 0 0 0 1 0 0];
            class_b_M_init(1 + num*10:num*11,:) = class11_data(1 +(b-1)*num:b*num,:);
            targets_M_init(1 + num*10:num*11,:) = [0 0 0 0 0 0 0 0 0 0 1 0];
            class_b_M_init(1 + num*11:num*12,:) = class12_data(1 +(b-1)*num:b*num,:);
            targets_M_init(1 + num*11:num*12,:) = [0 0 0 0 0 0 0 0 0 0 0 1];
            %shuffle -- careful; currently:default seed
            idx = randperm(g_batchsize);
            class_b_M = class_b_M_init(idx,:);
            targets_M = targets_M_init(idx,:);    
            g_batchdata(:,:,b) = class_b_M;
            g_batchtargets(:,:,b) = targets_M;
        end

else % testolin case (shapedata = letters)
    rand('state',0); %so we know the permutation of the training data / ≈rng object
    g_numbatches = size(shapedata,1)/g_batchsize;
    g_randomorder=randperm(g_totnum);
    for b=1:g_numbatches
      g_batchdata(:,:,b) = shapedata(g_randomorder(1+(b-1)*g_batchsize:b*g_batchsize),:);
      g_batchtargets(:,:,b) = targets(g_randomorder(1+(b-1)*g_batchsize:b*g_batchsize),:);
    end
end

% the following works in precision because of the uniform batches
% for the letter case, it bears a random factor but is still representative

if g_batchsize == 6|| g_batchsize == 26 || g_batchsize == 12
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

% remove the val and test samples/batches from g_batch..
g_batchdata = g_batchdata(:,:, 1:(numbatches-val_idx));
%imshow(reshape(g_batchdata(9,:,1), [40 40 1])');
g_batchtargets = g_batchtargets(:,:, 1:(numbatches-val_idx));

% save for speed when testing methods

% g_val: for overfitting measuring
% g_test: for classifier
% imshow(reshape(g_batchdata(3,:,250),[40 40 1])');
% fprintf("this is target: %d",find(g_batchtargets(3,:,250)));

clear shapedata targets;
