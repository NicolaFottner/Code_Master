%% export to excel spreadsheets
addpath("lit_eval/")
sourceDir = 'lit_eval/'; % already in 64x64 format
loadData = dir([sourceDir '*.mat']);

% strings(x,y), with x: number of architectures (only numHid)
%                    y: number of cases with D-&-Mini_b within a architecture
mean_list = strings(5,2);
j=1;
for i=1:2:size(loadData,1)
    mean_list(j,1) = cellstr(strcat(sourceDir,loadData(i).name));
    mean_list(j,2) = cellstr(strcat(sourceDir,loadData(i+1).name));
    j=j+1;
end

%% test error
C  = {};
x =repmat("XXX",[6 1]);
% add legends to C
param = "Parameters";
strg = "Test Acc";
legend_strg = [param;strg;"Epoch2";param;strg;"Epoch2"];
C(:,end+1) = cellstr(legend_strg);
for i=1:size(mean_list,1)
    letter_case=[];
    load(convertCharsToStrings(mean_list(i,1)),"properties","Classifier");
    ss = "D" + int2str(properties.dropout)+ "_M" +int2str(properties.minibatchsize) + ...
        "_h"+int2str(properties.numhid2) + "_LS";
    letter_case = [letter_case;ss];
    letter_case = [letter_case;num2str(Classifier.te_acc(1))];
    letter_case = [letter_case;num2str(properties.epoch2)];
    load(convertCharsToStrings(mean_list(i,2)),"properties","Classifier");
    ss = "D" + int2str(properties.dropout)+ "_M" +int2str(properties.minibatchsize) + ...
        "_h"+int2str(properties.numhid2) + "_MLP";
    letter_case = [letter_case;ss];
    letter_case = [letter_case;num2str(Classifier.te_acc(1))];
    letter_case = [letter_case;num2str(properties.epoch2)];
    C(:,end+1) = cellstr(letter_case);
    C(:,end+1) = cellstr(x);
end
Test_Error = C;


%% Classifier Details
C  = {};
x =repmat("XXX",[28 1]);
x_m ="XXXXXXXXXX";
% add legends to C
param = "Parameters";
legend_strg = [param;"(p)/A" ; "(p)/H"; "(p)/M";"(p)/U"; "(p)/T";"(p)/X";"1/Cross";...
    "2/Elipse";"3/Hexagon";"4/Rectangle";"5/Square";"6/Triangle";...
        "----";"----";param;"(p)/A" ; "(p)/H"; "(p)/M";"(p)/U"; "(p)/T";"(p)/X";...
            "1/Cross";"2/Elipse";"3/Hexagon";"4/Rectangle";"5/Square";"6/Triangle"];
C(:,end+1) = cellstr(legend_strg);

for i=1:size(mean_list,1)
    letter_case=[];
    load(convertCharsToStrings(mean_list(i,1)),"Classifier_Details","properties");
    ss = "D" + int2str(properties.dropout)+ "_M" +int2str(properties.minibatchsize) + ...
        "_h"+int2str(properties.numhid2) + "_LS";
    letter_case = [letter_case;ss];
    letter_case = [letter_case;num2str(Classifier_Details.Accuracy)];
    letter_case = [letter_case;x_m];
    letter_case = [letter_case;x_m];
    load(convertCharsToStrings(mean_list(i,2)),"Classifier_Details","properties");
    ss = "D" + int2str(properties.dropout)+ "_M" +int2str(properties.minibatchsize) + ...
        "_h"+int2str(properties.numhid2) + "_MLP";
    letter_case = [letter_case;ss];
    letter_case = [letter_case;num2str(Classifier_Details.Accuracy)];
    C(:,end+1) = cellstr(letter_case);
    C(:,end+1) = cellstr(x);
end
Classifier_details = C;

%% Id Based on Shape
C  = {};
x =repmat("XXX",[8 1]);
% add legends to C
param = "Parameters";
legend_strg = [param;"Geoshape";"Letters";"Ps-Letters";...
    param;"Geoshape";"Letters";"Ps-Letters"];
C(:,end+1) = cellstr(legend_strg);

for i=1:size(mean_list,1)
    letter_case=[];
    load(convertCharsToStrings(mean_list(i,1)),"Id_BasedOnGeoS","properties","Classifier");
    ss = "D" + int2str(properties.dropout)+ "_M" +int2str(properties.minibatchsize) + ...
        "_h"+int2str(properties.numhid2) + "_LS";
    letter_case = [letter_case;ss];
    holder = Id_BasedOnGeoS;
    id_gAcc = [Classifier.te_acc(1);holder.accuracy_l;holder.accuracy_pl];
    letter_case = [letter_case;num2str(id_gAcc)];
    load(convertCharsToStrings(mean_list(i,2)),"Id_BasedOnGeoS","properties","Classifier");
    ss = "D" + int2str(properties.dropout)+ "_M" +int2str(properties.minibatchsize) + ...
        "_h"+int2str(properties.numhid2) + "_MLP";
    letter_case = [letter_case;ss];
    holder = Id_BasedOnGeoS;
    id_gAcc = [Classifier.te_acc(1);holder.accuracy_l;holder.accuracy_pl];
    letter_case = [letter_case;num2str(id_gAcc)];
    C(:,end+1) = cellstr(letter_case);
    C(:,end+1) = cellstr(x);
end
Id_BasedOnShape = C;

%% Id Based on Shape --- Details
C  = {};
x =repmat("XXX",[18 1]);
%add legends to C
param = "Parameters";
legend_strg = [param;"Target";"(p)/A" ; "(p)/H"; "(p)/M";"(p)/U"; "(p)/T";"(p)/X";...
    "-----";"-----";param;"Target";"(p)/A" ; "(p)/H"; "(p)/M";"(p)/U"; "(p)/T";"(p)/X"];
C(:,end+1) = cellstr(legend_strg);

for i=1:size(mean_list,1)
    letter_case=[]; % for letters
    pletter_case = []; % for p-letters
    load(convertCharsToStrings(mean_list(i,1)),"Id_BasedOnGeoS","properties");
    ss = "D" + int2str(properties.dropout)+ "_M" +int2str(properties.minibatchsize) + ...
        "_h"+int2str(properties.numhid2) + "_LS";
    letter_case = [letter_case;ss];
    % for letters
    letter_case = [letter_case;"letter"];
    letter_case = [letter_case;num2str(Id_BasedOnGeoS.table_letter.Acc)];
    letter_case = [letter_case;x_m];letter_case = [letter_case;x_m];
    %for p-letters
    pletter_case = [pletter_case;"XXXXXX"];
    pletter_case = [pletter_case;"p-letter"];
    pletter_case = [pletter_case;num2str(Id_BasedOnGeoS.table_pletter.Acc)];
    pletter_case = [pletter_case;x_m];pletter_case = [pletter_case;x_m];
    
    load(convertCharsToStrings(mean_list(i,2)),"Id_BasedOnGeoS","properties");
    ss = "D" + int2str(properties.dropout)+ "_M" +int2str(properties.minibatchsize) + ...
        "_h"+int2str(properties.numhid2) + "_MLP";
    letter_case = [letter_case;ss];
    % for letters
    letter_case = [letter_case;"letter"];
    letter_case = [letter_case;num2str(Id_BasedOnGeoS.table_letter.Acc)];
    %for p-letters
    pletter_case = [pletter_case;"XXXXXX"];
    pletter_case = [pletter_case;"p-letter"];
    pletter_case = [pletter_case;num2str(Id_BasedOnGeoS.table_pletter.Acc)];
    C(:,end+1) = cellstr(letter_case);
    C(:,end+1) = cellstr(pletter_case);
    C(:,end+1) = cellstr(x);
end
Id_BasedOnS_Details =C;

%% ID based on shape -- Mode

C  = {};
x =repmat("XXX",[18 1]);
%add legends to C
param = "Parameters";
legend_strg = [param;"Target";"(p)/A" ; "(p)/H"; "(p)/M";"(p)/U"; "(p)/T";"(p)/X";...
    "-----";"-----";param;"Target";"(p)/A" ; "(p)/H"; "(p)/M";"(p)/U"; "(p)/T";"(p)/X"];
C(:,end+1) = cellstr(legend_strg);

for i=1:size(mean_list,1)
    letter_case=[]; % for letters
    pletter_case = []; % for p-letters
    load(convertCharsToStrings(mean_list(i,1)),"Id_BasedOnGeoS","properties");
    ss = "D" + int2str(properties.dropout)+ "_M" +int2str(properties.minibatchsize) + ...
        "_h"+int2str(properties.numhid2) + "_LS";
    letter_case = [letter_case;ss];
    % for letters
    letter_case = [letter_case;"letter"];
    letter_case = [letter_case;num2str(Id_BasedOnGeoS.table_letter.Mode)];
    letter_case = [letter_case;x_m];letter_case = [letter_case;x_m];
    %for p-letters
    pletter_case = [pletter_case;"XXXXXX"];
    pletter_case = [pletter_case;"p-letter"];
    pletter_case = [pletter_case;num2str(Id_BasedOnGeoS.table_pletter.Mode)];
    pletter_case = [pletter_case;x_m];pletter_case = [pletter_case;x_m];
    
    load(convertCharsToStrings(mean_list(i,2)),"Id_BasedOnGeoS","properties");
    ss = "D" + int2str(properties.dropout)+ "_M" +int2str(properties.minibatchsize) + ...
        "_h"+int2str(properties.numhid2) + "_MLP";
    letter_case = [letter_case;ss];
    % for letters
    letter_case = [letter_case;"letter"];
    letter_case = [letter_case;num2str(Id_BasedOnGeoS.table_letter.Mode)];
    %for p-letters
    pletter_case = [pletter_case;"XXXXXX"];
    pletter_case = [pletter_case;"p-letter"];
    pletter_case = [pletter_case;num2str(Id_BasedOnGeoS.table_pletter.Mode)];
    C(:,end+1) = cellstr(letter_case);
    C(:,end+1) = cellstr(pletter_case);
    C(:,end+1) = cellstr(x);
end
Id_BasedOnS_Mode =C;






%% writeOut
filename = "excel_files/xlit_13Jl.xlsx";
writecell(Test_Error,filename,'Sheet','Test_Error');
writecell(Classifier_details,filename,'Sheet','Classifier_details');
writecell(Id_BasedOnShape,filename,'Sheet','Id_BasedOnShape');
writecell(Id_BasedOnS_Mode,filename,'Sheet','Id_BasedonS_Details');



%% export PDFs for CE assesment
for i=1:size(mean_list,1)
    letter_case=[];
    load(convertCharsToStrings(mean_list(i,1)),"properties","CE_eval");
    ss = "D" + int2str(properties.dropout)+ "_M" +int2str(properties.minibatchsize) + ...
        "_h"+int2str(properties.numhid2) + "_LS";
    letter_case = [letter_case;ss];
    letter_case = [letter_case;num2str(CE_eval.CEs_Letter];
    letter_case = [letter_case;num2str(properties.epoch2)];
    load(convertCharsToStrings(mean_list(i,2)),"properties","Classifier");
    ss = "D" + int2str(properties.dropout)+ "_M" +int2str(properties.minibatchsize) + ...
        "_h"+int2str(properties.numhid2) + "_MLP";
    letter_case = [letter_case;ss];
    letter_case = [letter_case;num2str(Classifier.te_acc(1))];
    letter_case = [letter_case;num2str(properties.epoch2)];
    C(:,end+1) = cellstr(letter_case);
    C(:,end+1) = cellstr(x);
end



% % % % 
% % % % 
% % % % 
% % % % %% GET THE RELEVANT DATA
% % % % 
% % % % % CASE: Id Based on Shape --- Details
% % % % % (x,y) with x: num_architectures * num_cases per architecture
% % % % %            y: number of targets (here: 6 shapes)
% % % % letter_case = zeros(10,12);
% % % % pletter_case = zeros(10,12);
% % % % 
% % % % for i=1:size(mean_list,1)
% % % % 
% % % %     load(convertCharsToStrings(mean_list(i,1)),"Id_BasedOnGeoS");
% % % %     letter_case((i-1)*12+1,:) = Id_BasedOnGeoS.table_letter.Acc;
% % % %     pletter_case((i-1)*12+1,:) = Id_BasedOnGeoS.table_pletter.Acc;
% % % %     load(convertCharsToStrings(mean_list(i,2)),"Id_BasedOnGeoS");
% % % %     letter_case((i-1)*12+2,:) = Id_BasedOnGeoS.table_letter.Acc;
% % % %     pletter_case((i-1)*12+2,:) =Id_BasedOnGeoS.table_pletter.Acc;
% % % % 
% % % % end
% % % % 
% % % % aal_file_names =[];
% % % % aap_file_names =[];
% % % % aab_file_names =[];
% % % % aac_file_names = [];
% % % % aaal_file_names =[];
% % % % aaap_file_names =[];
% % % % for i=1:size(letter_case,1)
% % % %     cond1 = letter_case(i,:) > 0.5;
% % % %     cond2 = pletter_case(i,:) > 0.5;
% % % %     idx1= floor((i-1)/12)+1;
% % % %     idx2 = mod(i,12) * (mod(i,12)~=0) + (mod(i,12)+12) * (mod(i,12)==0);
% % % %     if sum(cond1) >= 3&& sum(cond2) >= 3
% % % %         aab_file_names = [aab_file_names;mean_list(idx1,idx2)];
% % % %     elseif sum(cond1) >=2 && sum(cond2) >= 2
% % % %     aac_file_names = [aac_file_names;mean_list(idx1,idx2)];
% % % %     elseif sum(cond1) >= 3
% % % %         aal_file_names = [aal_file_names;mean_list(idx1,idx2)];
% % % %     elseif sum(cond2) >= 3
% % % %         aap_file_names = [aap_file_names;mean_list(idx1,idx2)];
% % % %     elseif sum(cond1) >= 2
% % % %         aaal_file_names = [aaal_file_names;mean_list(idx1,idx2)];
% % % %     elseif sum(cond2) >= 2
% % % %         aaap_file_names = [aaap_file_names;mean_list(idx1,idx2)];
% % % %     end
% % % % end
% % % % for i=1:size(aab_file_names)
% % % %     fprintf("\nBest '>0.5' in 3: ALL:   " + aab_file_names(i));
% % % % end
% % % % for i=1:size(aac_file_names)
% % % %     fprintf("\nBest '>0.5' in 2: ALL:   " + aac_file_names(i));
% % % % end
% % % % for i=1:size(aal_file_names)
% % % %     fprintf("\nBest '>0.5' in 3: LETTER:   " + aal_file_names(i));
% % % % end
% % % % for i=1:size(aap_file_names)
% % % %     fprintf("\nBest '>0.5' in 3: PS-L   " + aap_file_names(i));
% % % % end
% % % % for i=1:size(aaal_file_names)
% % % %     fprintf("\nBest '>0.5' in 2: LETTER:   " + aaal_file_names(i));
% % % % end
% % % % for i=1:size(aaap_file_names)
% % % %     fprintf("\nBest '>0.5' in 2: PS-L   " + aaap_file_names(i));
% % % % end
