
% scan files, where unequal numbers of sims are available
addpath("Evals/")
sourceDir = 'Evals/'; % already in 64x64 format
fprintf(1,'Importing Data from runned simulations \n');
loadData = dir([sourceDir '*.mat']);

% do first entry:
load([loadData(1).name],"properties");
d = properties.dropout;
m = properties.minibatchsize;
h2 = properties.numhid2;
h3=properties.numhid3;
sim_list={};
sim_list(1,1) = cellstr(strcat(sourceDir,loadData(1).name));
j = 1;
z  = 2; 
% do rest:
for i=2:size(loadData,1)
    load([loadData(i).name],"properties");
    if properties.dropout == d &&  m == properties.minibatchsize &&  h2 == properties.numhid2 && h3 == properties.numhid3
        sim_list(j,z) = cellstr(strcat(sourceDir,loadData(i).name));
        z = z+1;
    else 
        d = properties.dropout;
        m = properties.minibatchsize;
        h2 = properties.numhid2;
        h3=properties.numhid3;
        j = j+1;
        z =1;
        sim_list(j,z) = cellstr(strcat(sourceDir,loadData(i).name));
        z=z+1;
    end
end
for i=1:size(sim_list,1)
    scan_evals(sim_list(i,:)');
end

