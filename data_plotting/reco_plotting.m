% plot (avg batch) reconstruction error
dd = strsplit(date,'-');
clean_date = strcat(dd(1),dd(2)); %without "-YYYY"

addpath("Evals/25J_tania/")
load Evals/25J_tania/25Jun_13h36m_H2300_H30.mat reco_error Overfitting


numbatches = size(reco_error,2);
full_rec_err = reco_error;
g_numcases = 12;

% reco-error
f = figure;
x_n= (1:size(full_rec_err,1));
y_n=sum(full_rec_err,2)/numbatches;
subplot(1,1,1);
plot(x_n,y_n);
title("")
xlabel('Epoch'),ylabel('Average reconstruction error')
% % % 
% % % % overfitting measure:
% % % % avgEnergy(training) - avgEnergy(validation)
% % % energy_diff = abs(Overfitting(:,1) - Overfitting(:,2));
% % % over_x=1:size(Overfitting,1);
% % % subplot(3,1,2);
% % % plot(over_x,Overfitting(:,1));
% % % hold on 
% % % plot(over_x,Overfitting(:,2));
% % % hold off
% % % subplot(3,1,3)
% % % plot(over_x,energy_diff);
% % % hold off
% % % title("")
% % % xlabel('Epoch')
% % % ylabel('Average Free energy')


file_name = "data_plotting/18_Sep_Reco_error.pdf";
exportgraphics(f,file_name)






