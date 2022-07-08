%% export to excel spreadsheets
addpath("Evals/mean/8Jl/")
sourceDir = 'Evals/mean/8Jl/'; % already in 64x64 format
loadData = dir([sourceDir '*.mat']);

% strings(x,y), with x: number of architectures (only numHid)
%                    y: number of cases with D-&-Mini_b within a architecture
mean_list = strings(5,6);

for i=1:size(loadData,1)
    load([loadData(i).name],"mean_properties");
    if i == 1
        h2 = mean_properties.numhid2;
        h3= mean_properties.numhid3;
        mean_list(1,1) = cellstr(strcat(sourceDir,loadData(1).name));
        j = 1;
        z  = 2; 
    else
        if h2 == mean_properties.numhid2 && h3 == mean_properties.numhid3
            mean_list(j,z) = cellstr(strcat(sourceDir,loadData(i).name));
            z = z+1;
        else 
            h2 = mean_properties.numhid2;
            h3=mean_properties.numhid3;
            j = j+1;
            z =1;
            mean_list(j,z) = cellstr(strcat(sourceDir,loadData(i).name));
            z=z+1;
        end
    end
end

%
mean_list(1,:) = [];

%% test error
C  = {};
x =repmat("XXX",[12 1]);
% add legends to C
param = "Parameters";
strg = "Test Error";
legend_strg = [param;strg;"Epoch2";"Epoch3";param;strg;"Epoch2";"Epoch3";...
    param;strg;"Epoch2";"Epoch3"];
C(:,end+1) = cellstr(legend_strg);

for i=1:size(mean_list,1)
    letter_case=[];
    load(convertCharsToStrings(mean_list(i,1)),"mean_properties");
    ss = "D" + int2str(mean_properties.dropout)+ "_M" +int2str(mean_properties.minibatch) + ...
        "_h"+int2str(mean_properties.numhid2)+"-"+int2str(mean_properties.numhid3);
    letter_case = [letter_case;ss];
    letter_case = [letter_case;num2str(mean_properties.test_error)];
    letter_case = [letter_case;num2str(mean_properties.epoch2)];
    letter_case = [letter_case;num2str(mean_properties.epoch3)];
    load(convertCharsToStrings(mean_list(i,2)),"mean_properties");
    ss = "D" + int2str(mean_properties.dropout)+ "_M" +int2str(mean_properties.minibatch) + ...
        "_h"+int2str(mean_properties.numhid2)+"-"+int2str(mean_properties.numhid3);
    letter_case = [letter_case;ss];
    letter_case = [letter_case;num2str(mean_properties.test_error)];
    letter_case = [letter_case;num2str(mean_properties.epoch2)];
    letter_case = [letter_case;num2str(mean_properties.epoch3)];
    load(convertCharsToStrings(mean_list(i,3)),"mean_properties");
    ss = "D" + int2str(mean_properties.dropout)+ "_M" +int2str(mean_properties.minibatch) + ...
        "_h"+int2str(mean_properties.numhid2)+"-"+int2str(mean_properties.numhid3);
    letter_case = [letter_case;ss];
    letter_case = [letter_case;num2str(mean_properties.test_error)];
    letter_case = [letter_case;num2str(mean_properties.epoch2)];
    letter_case = [letter_case;num2str(mean_properties.epoch3)];
    C(:,end+1) = cellstr(letter_case);
    %
    letter_case=[];
    load(convertCharsToStrings(mean_list(i,4)),"mean_properties");
    ss = "D" + int2str(mean_properties.dropout)+ "_M" +int2str(mean_properties.minibatch) + ...
        "_h"+int2str(mean_properties.numhid2)+"-"+int2str(mean_properties.numhid3);
    letter_case = [letter_case;ss];
    letter_case = [letter_case;num2str(mean_properties.test_error)];
    letter_case = [letter_case;num2str(mean_properties.epoch2)];
    letter_case = [letter_case;num2str(mean_properties.epoch3)];
    
    load(convertCharsToStrings(mean_list(i,5)),"mean_properties");
    ss = "D" + int2str(mean_properties.dropout)+ "_M" +int2str(mean_properties.minibatch) + ...
        "_h"+int2str(mean_properties.numhid2)+"-"+int2str(mean_properties.numhid3);
    letter_case = [letter_case;ss];
    letter_case = [letter_case;num2str(mean_properties.test_error)];
    letter_case = [letter_case;num2str(mean_properties.epoch2)];
    letter_case = [letter_case;num2str(mean_properties.epoch3)];
    
    load(convertCharsToStrings(mean_list(i,6)),"mean_properties");
    ss = "D" + int2str(mean_properties.dropout)+ "_M" +int2str(mean_properties.minibatch) + ...
        "_h"+int2str(mean_properties.numhid2)+"-"+int2str(mean_properties.numhid3);
    letter_case = [letter_case;ss];
    letter_case = [letter_case;num2str(mean_properties.test_error)];
    letter_case = [letter_case;num2str(mean_properties.epoch2)];
    letter_case = [letter_case;num2str(mean_properties.epoch3)];
    C(:,end+1) = cellstr(letter_case);
    C(:,end+1) = cellstr(x);
end
Test_Error = C;


%% Classifier Details
C  = {};
x =repmat("XXX",[25 1]);
x_m ="XXXXXXXXXX";
% add legends to C
param = "Parameters";
legend_strg = [param;"1/Cross";"2/Elipse";"3/Hexagon";"4/Rectangle";"5/Square";"6/Triangle";...
    "----";"----";param;"1/Cross";"2/Elipse";"3/Hexagon";"4/Rectangle";"5/Square";"6/Triangle";...
    "----";"----";param;"1/Cross";"2/Elipse";"3/Hexagon";"4/Rectangle";"5/Square";"6/Triangle"];
C(:,end+1) = cellstr(legend_strg);

for i=1:size(mean_list,1)
    letter_case=[];
    load(convertCharsToStrings(mean_list(i,1)),"Classifier_Details","mean_properties");
    ss = "D" + int2str(mean_properties.dropout)+ "_M" +int2str(mean_properties.minibatch) + ...
        "_h"+int2str(mean_properties.numhid2)+"-"+int2str(mean_properties.numhid3);
    letter_case = [letter_case;ss];
    letter_case = [letter_case;num2str(Classifier_Details.Accuracy)];
    letter_case = [letter_case;x_m];
    letter_case = [letter_case;x_m];
    load(convertCharsToStrings(mean_list(i,2)),"Classifier_Details","mean_properties");
    ss = "D" + int2str(mean_properties.dropout)+ "_M" +int2str(mean_properties.minibatch) + ...
        "_h"+int2str(mean_properties.numhid2)+"-"+int2str(mean_properties.numhid3);
    letter_case = [letter_case;ss];
    letter_case = [letter_case;num2str(Classifier_Details.Accuracy)];
    letter_case = [letter_case;x_m];
    letter_case = [letter_case;x_m];
    load(convertCharsToStrings(mean_list(i,3)),"Classifier_Details","mean_properties");
    ss = "D" + int2str(mean_properties.dropout)+ "_M" +int2str(mean_properties.minibatch) + ...
        "_h"+int2str(mean_properties.numhid2)+"-"+int2str(mean_properties.numhid3);
    letter_case = [letter_case;ss];
    letter_case = [letter_case;num2str(Classifier_Details.Accuracy)];
    C(:,end+1) = cellstr(letter_case);
    
    letter_case=[];
    load(convertCharsToStrings(mean_list(i,4)),"Classifier_Details","mean_properties");
    ss = "D" + int2str(mean_properties.dropout)+ "_M" +int2str(mean_properties.minibatch) + ...
        "_h"+int2str(mean_properties.numhid2)+"-"+int2str(mean_properties.numhid3);
    letter_case = [letter_case;ss];
    letter_case = [letter_case;num2str(Classifier_Details.Accuracy)];
    letter_case = [letter_case;x_m];
    letter_case = [letter_case;x_m];
    load(convertCharsToStrings(mean_list(i,5)),"Classifier_Details","mean_properties");
    ss = "D" + int2str(mean_properties.dropout)+ "_M" +int2str(mean_properties.minibatch) + ...
        "_h"+int2str(mean_properties.numhid2)+"-"+int2str(mean_properties.numhid3);
    letter_case = [letter_case;ss];
    letter_case = [letter_case;num2str(Classifier_Details.Accuracy)];
    letter_case = [letter_case;x_m];
    letter_case = [letter_case;x_m];
    load(convertCharsToStrings(mean_list(i,6)),"Classifier_Details","mean_properties");
    ss = "D" + int2str(mean_properties.dropout)+ "_M" +int2str(mean_properties.minibatch) + ...
        "_h"+int2str(mean_properties.numhid2)+"-"+int2str(mean_properties.numhid3);
    letter_case = [letter_case;ss];
    letter_case = [letter_case;num2str(Classifier_Details.Accuracy)];
    C(:,end+1) = cellstr(letter_case);
    C(:,end+1) = cellstr(x);
end
Classifier_details = C;

%% Id Based on Shape
C  = {};
x =repmat("XXX",[12 1]);
% add legends to C
param = "Parameters";
legend_strg = [param;"Geoshape";"Letters";"Ps-Letters";...
    param;"Geoshape";"Letters";"Ps-Letters";param;"Geoshape";"Letters";"Ps-Letters"];
C(:,end+1) = cellstr(legend_strg);

for i=1:size(mean_list,1)
    letter_case=[];
    load(convertCharsToStrings(mean_list(i,1)),"Id_BasedonS","mean_properties");
    ss = "D" + int2str(mean_properties.dropout)+ "_M" +int2str(mean_properties.minibatch) + ...
        "_h"+int2str(mean_properties.numhid2)+"-"+int2str(mean_properties.numhid3);
    letter_case = [letter_case;ss];
    id_gAcc = Id_BasedonS.Accuracy;
    id_gAcc(1) =  1 - mean_properties.test_error;
    letter_case = [letter_case;num2str(id_gAcc)];
    load(convertCharsToStrings(mean_list(i,2)),"Id_BasedonS","mean_properties");
    ss = "D" + int2str(mean_properties.dropout)+ "_M" +int2str(mean_properties.minibatch) + ...
        "_h"+int2str(mean_properties.numhid2)+"-"+int2str(mean_properties.numhid3);
    letter_case = [letter_case;ss];
    id_gAcc = Id_BasedonS.Accuracy;
    id_gAcc(1) =  1 - mean_properties.test_error;
    letter_case = [letter_case;num2str(id_gAcc)];
    load(convertCharsToStrings(mean_list(i,3)),"Id_BasedonS","mean_properties");
    ss = "D" + int2str(mean_properties.dropout)+ "_M" +int2str(mean_properties.minibatch) + ...
        "_h"+int2str(mean_properties.numhid2)+"-"+int2str(mean_properties.numhid3);
    letter_case = [letter_case;ss];
    id_gAcc = Id_BasedonS.Accuracy;
    id_gAcc(1) =  1 - mean_properties.test_error;
    letter_case = [letter_case;num2str(id_gAcc)];
    C(:,end+1) = cellstr(letter_case);
    %%%
    letter_case=[];
    load(convertCharsToStrings(mean_list(i,4)),"Id_BasedonS","mean_properties");
    ss = "D" + int2str(mean_properties.dropout)+ "_M" +int2str(mean_properties.minibatch) + ...
        "_h"+int2str(mean_properties.numhid2)+"-"+int2str(mean_properties.numhid3);
    letter_case = [letter_case;ss];
    id_gAcc = Id_BasedonS.Accuracy;
    id_gAcc(1) =  1 - mean_properties.test_error;
    letter_case = [letter_case;num2str(id_gAcc)];
    
    load(convertCharsToStrings(mean_list(i,5)),"Id_BasedonS","mean_properties");
    ss = "D" + int2str(mean_properties.dropout)+ "_M" +int2str(mean_properties.minibatch) + ...
        "_h"+int2str(mean_properties.numhid2)+"-"+int2str(mean_properties.numhid3);
    letter_case = [letter_case;ss];
    id_gAcc = Id_BasedonS.Accuracy;
    id_gAcc(1) =  1 - mean_properties.test_error;
    letter_case = [letter_case;num2str(id_gAcc)];
    load(convertCharsToStrings(mean_list(i,6)),"Id_BasedonS","mean_properties");
    ss = "D" + int2str(mean_properties.dropout)+ "_M" +int2str(mean_properties.minibatch) + ...
        "_h"+int2str(mean_properties.numhid2)+"-"+int2str(mean_properties.numhid3);
    letter_case = [letter_case;ss];
    id_gAcc = Id_BasedonS.Accuracy;
    id_gAcc(1) =  1 - mean_properties.test_error;
    letter_case = [letter_case;num2str(id_gAcc)];
    C(:,end+1) = cellstr(letter_case);
    C(:,end+1) = cellstr(x);
end
Id_BasedOnShape = C;

%% Id Based on Shape --- Details
C  = {};
x =repmat("XXX",[30 1]);
%add legends to C
param = "Parameters";
legend_strg = [param;"Target";"(p)/A" ; "(p)/H"; "(p)/M";"(p)/U"; "(p)/T";"(p)/X";...
    "-----";"-----";param;"Target";"(p)/A" ; "(p)/H"; "(p)/M";"(p)/U"; "(p)/T";"(p)/X";...
    "-----";"-----";param;"Target";"(p)/A" ; "(p)/H"; "(p)/M";"(p)/U"; "(p)/T";"(p)/X";...
    "-----";"-----"];
C(:,end+1) = cellstr(legend_strg);

for i=1:size(mean_list,1)
    letter_case=[]; % for letters
    pletter_case = []; % for p-letters
    load(convertCharsToStrings(mean_list(i,1)),"Id_BasedonS_details","mean_properties");
    ss = "D" + int2str(mean_properties.dropout)+ "_M" +int2str(mean_properties.minibatch) + ...
        "_h"+int2str(mean_properties.numhid2)+"-"+int2str(mean_properties.numhid3);
    letter_case = [letter_case;ss];
    % for letters
    letter_case = [letter_case;"letter"];
    letter_case = [letter_case;num2str(Id_BasedonS_details.letter_Acc)];
    letter_case = [letter_case;x_m];letter_case = [letter_case;x_m];
    %for p-letters
    pletter_case = [pletter_case;"XXXXXX"];
    pletter_case = [pletter_case;"p-letter"];
    pletter_case = [pletter_case;num2str(Id_BasedonS_details.pletter_Acc)];
    pletter_case = [pletter_case;x_m];pletter_case = [pletter_case;x_m];
    
    load(convertCharsToStrings(mean_list(i,2)),"Id_BasedonS_details","mean_properties");
    ss = "D" + int2str(mean_properties.dropout)+ "_M" +int2str(mean_properties.minibatch) + ...
        "_h"+int2str(mean_properties.numhid2)+"-"+int2str(mean_properties.numhid3);
    letter_case = [letter_case;ss];
    % for letters
    letter_case = [letter_case;"letter"];
    letter_case = [letter_case;num2str(Id_BasedonS_details.letter_Acc)];
    letter_case = [letter_case;x_m];letter_case = [letter_case;x_m];
    %for p-letters
    pletter_case = [pletter_case;"XXXXXX"];
    pletter_case = [pletter_case;"p-letter"];
    pletter_case = [pletter_case;num2str(Id_BasedonS_details.pletter_Acc)];
    pletter_case = [pletter_case;x_m];pletter_case = [pletter_case;x_m];

    load(convertCharsToStrings(mean_list(i,3)),"Id_BasedonS_details","mean_properties");
    ss = "D" + int2str(mean_properties.dropout)+ "_M" +int2str(mean_properties.minibatch) + ...
        "_h"+int2str(mean_properties.numhid2)+"-"+int2str(mean_properties.numhid3);
    letter_case = [letter_case;ss];
    % for letters
    letter_case = [letter_case;"letter"];
    letter_case = [letter_case;num2str(Id_BasedonS_details.letter_Acc)];
    letter_case = [letter_case;x_m];letter_case = [letter_case;x_m];
    %for p-letters
    pletter_case = [pletter_case;"XXXXXX"];
    pletter_case = [pletter_case;"p-letter"];
    pletter_case = [pletter_case;num2str(Id_BasedonS_details.pletter_Acc)];
    pletter_case = [pletter_case;x_m];pletter_case = [pletter_case;x_m];
    C(:,end+1) = cellstr(letter_case);
    C(:,end+1) = cellstr(pletter_case);
    C(:,end+1) = cellstr(x);

    % changing  D-case

    letter_case=[];
    pletter_case = [];
    load(convertCharsToStrings(mean_list(i,4)),"Id_BasedonS_details","mean_properties");
    ss = "D" + int2str(mean_properties.dropout)+ "_M" +int2str(mean_properties.minibatch) + ...
        "_h"+int2str(mean_properties.numhid2)+"-"+int2str(mean_properties.numhid3);
    letter_case = [letter_case;ss];
    % for letters
    letter_case = [letter_case;"letter"];
    letter_case = [letter_case;num2str(Id_BasedonS_details.letter_Acc)];
    letter_case = [letter_case;x_m];letter_case = [letter_case;x_m];
    %for p-letters
    pletter_case = [pletter_case;"-----"];
    pletter_case = [pletter_case;"p-letter"];
    pletter_case = [pletter_case;num2str(Id_BasedonS_details.pletter_Acc)];
    pletter_case = [pletter_case;x_m];pletter_case = [pletter_case;x_m];

    load(convertCharsToStrings(mean_list(i,5)),"Id_BasedonS_details","mean_properties");
    ss = "D" + int2str(mean_properties.dropout)+ "_M" +int2str(mean_properties.minibatch) + ...
        "_h"+int2str(mean_properties.numhid2)+"-"+int2str(mean_properties.numhid3);
    letter_case = [letter_case;ss];
    % for letters
    letter_case = [letter_case;"letter"];
    letter_case = [letter_case;num2str(Id_BasedonS_details.letter_Acc)];
    letter_case = [letter_case;x_m];letter_case = [letter_case;x_m];
    %for p-letters
    pletter_case = [pletter_case;"-----"];
    pletter_case = [pletter_case;"p-letter"];
    pletter_case = [pletter_case;num2str(Id_BasedonS_details.pletter_Acc)];
    pletter_case = [pletter_case;x_m];pletter_case = [pletter_case;x_m];

    load(convertCharsToStrings(mean_list(i,6)),"Id_BasedonS_details","mean_properties");
    ss = "D" + int2str(mean_properties.dropout)+ "_M" +int2str(mean_properties.minibatch) + ...
        "_h"+int2str(mean_properties.numhid2)+"-"+int2str(mean_properties.numhid3);
    letter_case = [letter_case;ss];
    % for letters
    letter_case = [letter_case;"letter"];
    letter_case = [letter_case;num2str(Id_BasedonS_details.letter_Acc)];
    letter_case = [letter_case;x_m];letter_case = [letter_case;x_m];
    %for p-letters
    pletter_case = [pletter_case;"-----"];
    pletter_case = [pletter_case;"p-letter"];
    pletter_case = [pletter_case;num2str(Id_BasedonS_details.pletter_Acc)];
    pletter_case = [pletter_case;x_m];pletter_case = [pletter_case;x_m];
    C(:,end+1) = cellstr(letter_case);
    C(:,end+1) = cellstr(pletter_case);
    C(:,end+1) = cellstr(x);
end
Id_BasedonS_Details =C;






%% writeOut
filename = "excel_files/8Jl_sim5.xlsx";
writecell(Test_Error,filename,'Sheet','Test_Error');
writecell(Classifier_details,filename,'Sheet','Classifier_details');
writecell(Id_BasedOnShape,filename,'Sheet','Id_BasedOnShape');
writecell(Id_BasedonS_Details,filename,'Sheet','Id_BasedonS_Details');




%% GET THE RELEVANT DATA

% CASE: Id Based on Shape --- Details
% (x,y) with x: num_architectures * num_cases per architecture
%            y: number of targets (here: 6 shapes)
letter_case = zeros(30,6);
pletter_case = zeros(30,6);

for i=1:size(mean_list,1)

    load(convertCharsToStrings(mean_list(i,1)),"Id_BasedonS_details");
    letter_case((i-1)*6+1,:) = Id_BasedonS_details.letter_Acc;
    pletter_case((i-1)*6+1,:) = Id_BasedonS_details.pletter_Acc;
    load(convertCharsToStrings(mean_list(i,2)),"Id_BasedonS_details");
    letter_case((i-1)*6+2,:) = Id_BasedonS_details.letter_Acc;
    pletter_case((i-1)*6+2,:) = Id_BasedonS_details.pletter_Acc;
    load(convertCharsToStrings(mean_list(i,3)),"Id_BasedonS_details");
    letter_case((i-1)*6+3,:) = Id_BasedonS_details.letter_Acc;
    pletter_case((i-1)*6+3,:) = Id_BasedonS_details.pletter_Acc;
    load(convertCharsToStrings(mean_list(i,4)),"Id_BasedonS_details");
    letter_case((i-1)*6+4,:) = Id_BasedonS_details.letter_Acc;
    pletter_case((i-1)*6+4,:) = Id_BasedonS_details.pletter_Acc;
    load(convertCharsToStrings(mean_list(i,5)),"Id_BasedonS_details");
    letter_case((i-1)*6+5,:) = Id_BasedonS_details.letter_Acc;
    pletter_case((i-1)*6+5,:) = Id_BasedonS_details.pletter_Acc;
    load(convertCharsToStrings(mean_list(i,6)),"Id_BasedonS_details");
    letter_case((i-1)*6+6,:) = Id_BasedonS_details.letter_Acc;
    pletter_case((i-1)*6+6,:) = Id_BasedonS_details.pletter_Acc;

end

aal_file_names =[];
aap_file_names =[];
aab_file_names =[];
aac_file_names = [];
aaal_file_names =[];
aaap_file_names =[];
for i=1:size(letter_case,1)
    cond1 = letter_case(i,:) > 0.5;
    cond2 = pletter_case(i,:) > 0.5;
    idx1= floor((i-1)/6)+1;
    idx2 = mod(i,6) * (mod(i,6)~=0) + (mod(i,6)+6) * (mod(i,6)==0);
    if sum(cond1) >= 3&& sum(cond2) >= 3
        aab_file_names = [aab_file_names;mean_list(idx1,idx2)];
    elseif sum(cond1) >=2 && sum(cond2) >= 2
    aac_file_names = [aac_file_names;mean_list(idx1,idx2)];
    elseif sum(cond1) >= 3
        aal_file_names = [aal_file_names;mean_list(idx1,idx2)];
    elseif sum(cond2) >= 3
        aap_file_names = [aap_file_names;mean_list(idx1,idx2)];
    elseif sum(cond1) >= 2
        aaal_file_names = [aaal_file_names;mean_list(idx1,idx2)];
    elseif sum(cond2) >= 2
        aaap_file_names = [aaap_file_names;mean_list(idx1,idx2)];
    end
end
for i=1:size(aab_file_names)
    fprintf("\nBest '>0.5' in 3: ALL:   " + aab_file_names(i));
end
for i=1:size(aac_file_names)
    fprintf("\nBest '>0.5' in 2: ALL:   " + aac_file_names(i));
end
for i=1:size(aal_file_names)
    fprintf("\nBest '>0.5' in 3: LETTER:   " + aal_file_names(i));
end
for i=1:size(aap_file_names)
    fprintf("\nBest '>0.5' in 3: PS-L   " + aap_file_names(i));
end
for i=1:size(aaal_file_names)
    fprintf("\nBest '>0.5' in 2: LETTER:   " + aaal_file_names(i));
end
for i=1:size(aaap_file_names)
    fprintf("\nBest '>0.5' in 2: PS-L   " + aaap_file_names(i));
end
