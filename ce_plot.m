% plot _ CE effect
dd = strsplit(date,'-'); clean_date = strcat(dd(1),dd(2));c=clock; %store date without "-YYYY"
str_date = "" + clean_date + int2str(c(4)) + "h" + int2str(c(5)) + "m";

addpath("Evals/");
load 20Jun_17h27m_H2350_H30.mat CE_eval

pdfs = CE_eval.detail.pdr_cong_l;
f = figure;
% for letter A:
subplot(2,3,1);
bar(pdfs(1,:));
xlabel('Triangle / A');
% for letter H:
subplot(2,3,2);
bar(pdfs(2,:));
xlabel('Rectangle / H');
% for letter M:
subplot(2,3,3);
bar(pdfs(3,:));
xlabel('Square / M');
% for letter U:
subplot(2,3,4);
bar(pdfs(4,:));
xlabel('Elipse / U');
% for letter T:
subplot(2,3,5);
bar(pdfs(5,:));
xlabel('Cross / T');
% for letter X:
subplot(2,3,6);
bar(pdfs(6,:));
xlabel('Hexagon / X');
sgtitle("Models' Predictions/Prob Distr on CE-data: ");
file_namee = "Evals/plots/" + str_date + "_congCE_L_prD"+".png";
exportgraphics(f,file_namee);
clear figure;

pdfs = CE_eval.detail.pdr_cong_pl;
f = figure;
% for letter A:
subplot(2,3,1);
bar(pdfs(1,:));
xlabel('Triangle / pA');
% for letter H:
subplot(2,3,2);
bar(pdfs(2,:));
xlabel('Rectangle / pH');
% for letter M:
subplot(2,3,3);
bar(pdfs(3,:));
xlabel('Square / pM');
% for letter U:
subplot(2,3,4);
bar(pdfs(4,:));
xlabel('Elipse / pU');
% for letter T:
subplot(2,3,5);
bar(pdfs(5,:));
xlabel('Cross / pT');
% for letter X:
subplot(2,3,6);
bar(pdfs(6,:));
xlabel('Hexagon / pX');
sgtitle("Congr: Predictions/Prob Distr on CE-data: ");
file_namee = "Evals/plots/" + str_date + "_congCE_psL_prD"+".png";
exportgraphics(f,file_namee);
clear figure;

%% for "general" incongruent case
pdfs = CE_eval.detail.pdr_inc_l;
f = figure;
% for letter A:
subplot(2,3,1);bar(pdfs(1,:));xlabel('Triangle / A');
% for letter H:
subplot(2,3,2);bar(pdfs(2,:));xlabel('Rectangle / H');
% for letter M:
subplot(2,3,3);bar(pdfs(3,:));xlabel('Square / M');
% for letter U:
subplot(2,3,4);bar(pdfs(4,:));xlabel('Elipse / U');
% for letter T:
subplot(2,3,5);bar(pdfs(5,:));xlabel('Cross / T');
% for letter X:
subplot(2,3,6);bar(pdfs(6,:));xlabel('Hexagon / X');
sgtitle("Incong: Predictions/Prob Distr on CE-data: ");
file_namee = "Evals/plots/" + str_date + "_incCE_L_prD"+".png";
exportgraphics(f,file_namee);
clear figure;
%%% PseudoLetters:
pdfs = CE_eval.detail.pdr_inc_pl;
f = figure;
% for letter A:
subplot(2,3,1);bar(pdfs(1,:));xlabel('Triangle / pA');
% for letter H:
subplot(2,3,2);bar(pdfs(2,:));xlabel('Rectangle / pH');
% for letter M:
subplot(2,3,3);bar(pdfs(3,:));xlabel('Square / pM');
% for letter U:
subplot(2,3,4);bar(pdfs(4,:));xlabel('Elipse / pU');
% for letter T:
subplot(2,3,5);bar(pdfs(5,:));xlabel('Cross / pT');
% for letter X:
subplot(2,3,6);bar(pdfs(6,:));xlabel('Hexagon / pX');
sgtitle("Incong: Predictions/Prob Distr on CE-data: ");
file_namee = "Evals/plots/" + str_date + "_incCE_psL_prD"+".png";
exportgraphics(f,file_namee);
clear figure;


%% for detailed incongruent case 
%%% LETTER:
pdfs = CE_eval.detail.detailed_inc_pdr_l;
f = figure;
% for letter A:
subplot(3,2,1);bar(pdfs(1,:));xlabel('Triangle / H');
subplot(3,2,2);bar(pdfs(2,:));xlabel('Triangle / U');
% for letter H:
subplot(3,2,3);bar(pdfs(3,:));xlabel('Rectangle / A');
subplot(3,2,4);bar(pdfs(4,:));xlabel('Rectangle / U');
% for letter M:
subplot(3,2,5);bar(pdfs(5,:));xlabel('Square / T');
subplot(3,2,6);bar(pdfs(6,:));xlabel('Square / X');
sgtitle("Incongr: Detailed1 - Predictions/Prob Distr on CE-data: ");
file_namee = "Evals/plots/" + str_date + "_incDetailed1_L_prD"+".png";
exportgraphics(f,file_namee);
clear figure;
f = figure;
% for letter U:
subplot(3,2,1);bar(pdfs(7,:));xlabel('Elipse / A');
subplot(3,2,2);bar(pdfs(8,:));xlabel('Elipse / H');
% for letter T:
subplot(3,2,3);bar(pdfs(9,:));xlabel('Cross / M');
subplot(3,2,4);bar(pdfs(10,:));xlabel('Cross / X');
% for letter X:
subplot(3,2,5);bar(pdfs(11,:));xlabel('Hexagon / M');
subplot(3,2,6);bar(pdfs(12,:));xlabel('Hexagon / T');
sgtitle("Incongr: Detailed2 - Predictions/Prob Distr on CE-data: ");
file_namee = "Evals/plots/" + str_date + "_incDetailed2_L_prD"+".png";
exportgraphics(f,file_namee);
clear figure;

%%% PSEUDO - LETTER:
pdfs = CE_eval.detail.detailed_inc_pdr_pl;
f = figure;
% for letter A:
subplot(3,2,1);bar(pdfs(1,:));xlabel('Triangle / pH');
subplot(3,2,2);bar(pdfs(2,:));xlabel('Triangle / pU');
% for letter H:
subplot(3,2,3);bar(pdfs(3,:));xlabel('Rectangle / pA');
subplot(3,2,4);bar(pdfs(4,:));xlabel('Rectangle / pU');
% for letter M:
subplot(3,2,5);bar(pdfs(5,:));xlabel('Square / pT');
subplot(3,2,6);bar(pdfs(6,:));xlabel('Square / pX');
sgtitle("Incongr: Detailed1 - Predictions/Prob Distr on CE-data: ");
file_namee = "Evals/plots/" + str_date + "_incDetailed1_psL_prD"+".png";
exportgraphics(f,file_namee);
clear figure;
f = figure;
% for letter U:
subplot(3,2,1);bar(pdfs(7,:));xlabel('Elipse / pA');
subplot(3,2,2);bar(pdfs(8,:));xlabel('Elipse / pH');
% for letter T:
subplot(3,2,3);bar(pdfs(9,:));xlabel('Cross / pM');
subplot(3,2,4);bar(pdfs(10,:));xlabel('Cross / pX');
% for letter X:
subplot(3,2,5);bar(pdfs(11,:));xlabel('Hexagon / pM');
subplot(3,2,6);bar(pdfs(12,:));xlabel('Hexagon / pT');
sgtitle("Incongr: Detailed2 - Predictions/Prob Distr on CE-data: ");
file_namee = "Evals/plots/" + str_date + "_incDetailed2_psL_prD"+".png";
exportgraphics(f,file_namee);
clear figure;


%%% detailed

