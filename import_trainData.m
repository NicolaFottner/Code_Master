%% concatenate shape and letter-in-string data:

%%% Percentage of shape data?, for equal part: perc = 1
perc = 1;
%%%

load open_CV_trainData12_dbl.mat data target_l target_pos
xdata = data;

load openCV_shapePOS3.mat data target_s
perc = perc * size(xdata,1) / size(data,1);

shapedata = zeros(size(data));
for i=1:size(data,1)
    shapedata(i,:) = reshape(im2double(reshape(data(i,:),[40 40 1])), [1 1600]);
end
data = shapedata;
target_s = double(target_s);


% take equal amount of each shape:
class1_data = [];class2_data = [];class3_data = [];
class4_data = [];class5_data = [];class6_data = [];
for i=1:size(data,1)
    if find(target_s(i,:)) == 1
        class1_data = [class1_data;data(i,:)];
    end
    if find(target_s(i,:)) == 2
        class2_data = [class2_data;data(i,:)];
    end
    if find(target_s(i,:)) == 3
        class3_data = [class3_data;data(i,:)];
    end
    if find(target_s(i,:)) == 4
        class4_data = [class4_data;data(i,:)];
    end
    if find(target_s(i,:)) == 5
        class5_data = [class5_data;data(i,:)];
    end
    if find(target_s(i,:)) == 6
        class6_data = [class6_data;data(i,:)];
    end
end

minus = size(class1_data,1) - perc * size(class1_data,1);
class1_data(1:minus,:)=[];
class2_data(1:minus,:)=[];
class3_data(1:minus,:)=[];
class4_data(1:minus,:)=[];
class5_data(1:minus,:)=[];
class6_data(1:minus,:)=[];

targets1 = repmat([0 0 0 0 0 0 1 0 0 0 0 0],size(class1_data,1),1);
targets2 = repmat([0 0 0 0 0 0 0 1 0 0 0 0],size(class1_data,1),1);
targets3 = repmat([0 0 0 0 0 0 0 0 1 0 0 0],size(class1_data,1),1);
targets4 = repmat([0 0 0 0 0 0 0 0 0 1 0 0],size(class1_data,1),1);
targets5 = repmat([0 0 0 0 0 0 0 0 0 0 1 0],size(class1_data,1),1);
targets6 = repmat([0 0 0 0 0 0 0 0 0 0 0 1],size(class1_data,1),1);

n_data = [class1_data;class2_data;class3_data;class4_data;class5_data;class6_data];
n_target = [targets1;targets2;targets3;targets4;targets5;targets6];

data = [xdata;n_data];
targets = [target_l;n_target];
%shuffle
idx = randperm(size(data,1));
data = data(idx,:);
targets = targets(idx,:);

save data/new04Jl/50_50_trainData.mat data targets target_pos








