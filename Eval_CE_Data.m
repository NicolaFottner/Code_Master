% Computing average of CE-assesments stores in file /CE_22_April

addpath("plots_results/")

sourceDir = 'plots_results/CE_22April/'; % already in 64x64 format
fprintf(1,'Importing Data from runned simulations \n');
loadData = dir([sourceDir '*.mat']);

load([sourceDir loadData(1).name],"Accuracy_Measurements_letter","Accuracy_Measurements_shape","CE_letter");

acc_shape =[];
acc_letter = [];
ce_l = [];
ce_pl=[];
for i=1:length(loadData)
    load([sourceDir loadData(i).name],"Accuracy_Measurements_letter","Accuracy_Measurements_shape","CE_letter");
    acc_letter = [acc_letter;Accuracy_Measurements_letter.Combined_l(3)];
    acc_shape = [acc_shape;Accuracy_Measurements_shape.Combined(3)];
    ce_l = [ce_l;CE_letter.CE_l(1)];
    ce_pl = [ce_pl;CE_letter.CE_l(2)];
end

mean_acc_s  = mean(acc_shape);
mean_acc_l  = mean(acc_letter);
mean_cl_l  = mean(ce_l);
mean_cl_pl  = mean(ce_pl);

% plot "bars"

tiledlayout(2,1)
ax1 = nexttile;
title('Accuracy');
ylabel('in %');
y = [mean_acc_s;mean_acc_l];
bar(ax1,y)
set(ax1,'xticklabel', {'Shape','Letter'});
y = [mean_cl_l;mean_cl_pl];
ax2 = nexttile;
title('Accuracy');
ylabel('CE effect');
bar(ax2,y,'stacked')
set(ax2,'xticklabel', {'Letter','PseudoLetter'});




