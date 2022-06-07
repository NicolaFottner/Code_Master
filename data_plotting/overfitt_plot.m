
% Overfitting plotting:

%load needed hyperparameters data
load info g_numcases
%% rbm 2
load err_rbm_2 overfitting_g;

% avgEnergy(training) - avgEnergy(validation)
energy_diff = abs(overfitting_g(:,1) - overfitting_g(:,2));
over_x=1:size(overfitting_g,1);
subplot(2,1,2);
plot(over_x,overfitting_g(:,1));
hold on 
plot(over_x,overfitting_g(:,2));
hold on
plot(over_x,energy_diff);
hold off
title("RBM2" + " w/ Bsize: " + int2str(g_numcases) + "--" + clean_date)
xlabel('Epoch')
ylabel('AvgEnergy')

file_name = 'data_plotting/overfitting.pdf';
exportgraphics(f,file_name,'Resolution',500);

% for future: to append to preexisting pdf:
% use: exportgraphics(ax,'myplots.pdf','Append',true)


% other method could be:
% avgEnergy(training) - avgEnergy(validation)
%energy_diff = overfitting(:,1) - overfitting(:,2);

%% rbm 3


