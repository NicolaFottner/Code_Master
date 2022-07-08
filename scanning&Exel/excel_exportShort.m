%% export to excel spreadsheets
addpath("Evals/mean/")
sourceDir = 'Evals/mean/'; % already in 64x64 format
loadData = dir([sourceDir '*.mat']);

% strings(x,y), with x: number of architectures
%                    y: number of cases within a architecture
mean_list = strings(2,1);

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

%% test error
C  = {};
x =repmat("XXX",[4 1]);
% add legends to C
param = "Parameters";
strg = "Test Error";
legend_strg = [param;strg;"Epoch2";"Epoch3"];
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
    C(:,end+1) = cellstr(letter_case);
    C(:,end+1) = cellstr(x);
end
Test_Error = C;


%% Classifier Details
C  = {};
x =repmat("XXX",[7 1]);
% add legends to C
param = "Parameters";
legend_strg = [param;"1/Cross";"2/Elipse";"3/Hexagon";"4/Rectangle";"5/Square";"6/Triangle"];
C(:,end+1) = cellstr(legend_strg);

for i=1:size(mean_list,1)
    letter_case=[];
    load(convertCharsToStrings(mean_list(i,1)),"Classifier_Details","mean_properties");
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
x =repmat("XXX",[4 1]);
% add legends to C
param = "Parameters";
legend_strg = [param;"Geoshape";"Letters";"Ps-Letters"];
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
    C(:,end+1) = cellstr(letter_case);
    C(:,end+1) = cellstr(x);
end
Id_BasedOnShape = C;

%% Id Based on Shape --- Details
C  = {};
x =repmat("XXX",[8 1]);
%add legends to C
param = "Parameters";
legend_strg = [param;"Target";"(p)/A" ; "(p)/H"; "(p)/M";"(p)/U"; "(p)/T";"(p)/X"];
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
    %for p-letters
    pletter_case = [pletter_case;"XXXXXX"];
    pletter_case = [pletter_case;"p-letter"];
    pletter_case = [pletter_case;num2str(Id_BasedonS_details.pletter_Acc)];
    C(:,end+1) = cellstr(letter_case);
    C(:,end+1) = cellstr(pletter_case);
    C(:,end+1) = cellstr(x);
end
Id_BasedonS_Details =C;

%% writeOut
filename = "excel_files/16J_sim5.xlsx";
writecell(Test_Error,filename,'Sheet','Test_Error');
writecell(Classifier_details,filename,'Sheet','Classifier_details');
writecell(Id_BasedOnShape,filename,'Sheet','Id_BasedOnShape');
writecell(Id_BasedonS_Details,filename,'Sheet','Id_BasedonS_Details');


%% GET THE RELEVANT DATA

% CASE: Id Based on Shape --- Details
letter_case = zeros(2,6);
pletter_case = zeros(2,6);

for i=1:size(mean_list,1)

    load(convertCharsToStrings(mean_list(i,1)),"Id_BasedonS_details");
    letter_case((i-1)*6+1,:) = Id_BasedonS_details.letter_Acc;
    pletter_case((i-1)*6+1,:) = Id_BasedonS_details.pletter_Acc;
    load(convertCharsToStrings(mean_list(i,2)),"Id_BasedonS_details");
    letter_case((i-1)*6+2,:) = Id_BasedonS_details.letter_Acc;
    pletter_case((i-1)*6+2,:) = Id_BasedonS_details.pletter_Acc;
%     load(convertCharsToStrings(mean_list(i,3)),"Id_BasedonS_details");
%     letter_case((i-1)*6+3,:) = Id_BasedonS_details.letter_Acc;
%     pletter_case((i-1)*6+3,:) = Id_BasedonS_details.pletter_Acc;
%     load(convertCharsToStrings(mean_list(i,4)),"Id_BasedonS_details");
%     letter_case((i-1)*6+4,:) = Id_BasedonS_details.letter_Acc;
%     pletter_case((i-1)*6+4,:) = Id_BasedonS_details.pletter_Acc;

end

l_file_names =[];
p_file_names =[];
b_file_names =[];
for i=1:size(letter_case,1)
    cond1 = letter_case(i,:) > 0.5;
    cond2 = pletter_case(i,:) > 0.5;
    idx1= floor((i-1)/6)+1;
    idx2 = mod(i,6) * (mod(i,6)~=0) + (mod(i,6)+6) * (mod(i,6)==0);
    if sum(cond1) >= 3 && sum(cond2) >= 3
        b_file_names = [b_file_names;mean_list(idx1,idx2)];
    elseif sum(cond1) >= 3
        l_file_names = [l_file_names;mean_list(idx1,idx2)];
    elseif sum(cond2) >= 3
        p_file_names = [p_file_names;mean_list(idx1,idx2)];
    end
end
for i=1:size(b_file_names)
    fprintf("\nBest Model: " + b_file_names(i));
end


