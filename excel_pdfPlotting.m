% plot PDF from final/evaluation data
% nothing to do with excel actually ...

addpath("Evals/");
% for 1300,900,500,300
for i = 1:1
    sourceDir = 'Evals/'; % already in 64x64 format
    token = '16Jun_11h39m_H2350_H30';
    sourceDir = strcat(sourceDir,token);
    %loadData = dir([sourceDir '*.mat']);
    load(sourceDir,"Id_BasedOnGeoS");
    pdfs_l = Id_BasedOnGeoS.letter_pdr;
    pdfs_pl = Id_BasedOnGeoS.pletter_pdr;
    f = figure;
    % for letter A:
    subplot(2,3,1);
    bar(pdfs_l(1,:));
    xlabel('Letter A');
    % for letter H:
    subplot(2,3,2);
    bar(pdfs_l(2,:));
    xlabel('Letter H');
    % for letter M:
    subplot(2,3,3);
    bar(pdfs_l(3,:));
    xlabel('Letter M');
    % for letter U:
    subplot(2,3,4);
    bar(pdfs_l(4,:));
    xlabel('Letter U');
    % for letter T:
    subplot(2,3,5);
    bar(pdfs_l(5,:));
    xlabel('Letter T');
    % for letter X:
    subplot(2,3,6);
    bar(pdfs_l(6,:));
    xlabel('Letter X');
    sgtitle("Models' Prediction/Prob Distr: ");
    
    file_namee = "Evals/plots/" + token + "_L_prD"+".png";
    exportgraphics(f,file_namee);
    clear figure;
    f = figure;
    % for letter A:
    subplot(2,3,1);
    bar(pdfs_pl(1,:));
    xlabel('PseudoLetter A');
    % for letter H:
    subplot(2,3,2);
    bar(pdfs_pl(2,:));
    xlabel('PseudoLetter H');
    % for letter M:
    subplot(2,3,3);
    bar(pdfs_pl(3,:));
    xlabel('PseudoLetter M');
    % for letter U:
    subplot(2,3,4);
    bar(pdfs_pl(4,:));
    xlabel('PseudoLetter U');
    % for letter T:
    subplot(2,3,5);
    bar(pdfs_pl(5,:));
    xlabel('PseudoLetter T');
    % for letter X:
    subplot(2,3,6);
    bar(pdfs_pl(6,:));
    xlabel('PseudoLetter X');
    sgtitle("Models' Prediction/Prob Distr: ");
    file_namee = "Evals/plots/" + token + "_pseudoL_prD_"+ ".png";
    exportgraphics(f,file_namee);
end


