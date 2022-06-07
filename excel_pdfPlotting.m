% plot PDF from final/evaluation data

addpath("Evals/");

% for 1300,900,500,300
for i = 1:2
    sourceDir = 'Evals/'; % already in 64x64 format
    if i == 1
        token = 'sim6_1300n';
    elseif i == 2
        token = 'sim6_1300_100n';
    end
    sourceDir = strcat(sourceDir,token);
    %loadData = dir([sourceDir '*.mat']);
    load(sourceDir,"Id_BasedonS_PDFs");
    pdfs = Id_BasedonS_PDFs.Models_Output_PDF;
    f = figure;
    % for letter A:
    subplot(2,3,1);
    bar(pdfs(1,:));
    xlabel('Letter A');
    % for letter H:
    subplot(2,3,2);
    bar(pdfs(2,:));
    xlabel('Letter H');
    % for letter M:
    subplot(2,3,3);
    bar(pdfs(3,:));
    xlabel('Letter M');
    % for letter U:
    subplot(2,3,4);
    bar(pdfs(4,:));
    xlabel('Letter U');
    % for letter T:
    subplot(2,3,5);
    bar(pdfs(5,:));
    xlabel('Letter T');
    % for letter X:
    subplot(2,3,6);
    bar(pdfs(6,:));
    xlabel('Letter X');
    sgtitle("Models' Prediction/Prob Distr: ");
    
    file_namee = "Evals/plots/" + token + "_L_prD"+".png";
    exportgraphics(f,file_namee);
    clear figure;
    f = figure;
    % for letter A:
    subplot(2,3,1);
    bar(pdfs(7,:));
    xlabel('PseudoLetter A');
    % for letter H:
    subplot(2,3,2);
    bar(pdfs(8,:));
    xlabel('PseudoLetter H');
    % for letter M:
    subplot(2,3,3);
    bar(pdfs(9,:));
    xlabel('PseudoLetter M');
    % for letter U:
    subplot(2,3,4);
    bar(pdfs(10,:));
    xlabel('PseudoLetter U');
    % for letter T:
    subplot(2,3,5);
    bar(pdfs(11,:));
    xlabel('PseudoLetter T');
    % for letter X:
    subplot(2,3,6);
    bar(pdfs(12,:));
    xlabel('PseudoLetter X');
    sgtitle("Models' Prediction/Prob Distr: ");
    file_namee = "Evals/plots/" + token + "_pseudoL_prD_"+ ".png";
    exportgraphics(f,file_namee);
end


