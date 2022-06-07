% plot (avg batch) reconstruction error
dd = strsplit(date,'-');
clean_date = strcat(dd(1),dd(2)); %without "-YYYY"

load info maxepoch numbatches numcases g_numbatches g_numcases

load err_rbm_1_200.mat full_rec_err
f = figure;
x_n= (1:size(full_rec_err,1));
y_n=sum(full_rec_err,2)/numbatches;
subplot(1,2,1);
plot(x_n,y_n);
title("RBM 1 on: " + clean_date)
xlabel('Epoch'),ylabel('avg batch Reco-error')

load err_rbm_2.mat full_rec_err_g
x_g= (1:size(full_rec_err_g,1));
y_g=sum(full_rec_err_g,2)/g_numbatches;
subplot(1,2,2);
plot(x_g,y_g);
title("RBM 2" + " w/ B_size: " + int2str(g_numcases) +"--" + clean_date)
xlabel('Epoch'),ylabel('avg batch Reco-error')

file_name = "data_plotting/reco_error.pdf";
exportgraphics(f,file_name)












