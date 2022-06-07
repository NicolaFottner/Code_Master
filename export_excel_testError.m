addpath("Evals/mean/")
C  = {};
sourceDir =  'Evals/mean/';
for i=1:7
    sourceDir = [sourceDir int2str(i) '/'];
    loadData = dir([sourceDir '*.mat']);
    te_err=[];
    load(convertCharsToStrings(sourceDir) + convertCharsToStrings(loadData(1).name),"mean_properties");
    ss = "D" + int2str(mean_properties.dropout)+ "_M" +int2str(mean_properties.minibatch) + ...
        "_h"+int2str(mean_properties.numhid2)+"-"+int2str(mean_properties.numhid3);
    te_err = [te_err;ss];
    te_err = [te_err;num2str(mean_properties.test_error)];
    
    load(convertCharsToStrings(sourceDir) + convertCharsToStrings(loadData(2).name),"mean_properties");
    ss = "D" + int2str(mean_properties.dropout)+ "_M" +int2str(mean_properties.minibatch) + ...
        "_h"+int2str(mean_properties.numhid2)+"-"+int2str(mean_properties.numhid3);
    te_err = [te_err;ss];
    te_err = [te_err;num2str(mean_properties.test_error)];
    C(:,end+1) = cellstr(te_err);
    
    te_err=[];
    load(convertCharsToStrings(sourceDir) + convertCharsToStrings(loadData(3).name),"mean_properties");
    ss = "D" + int2str(mean_properties.dropout)+ "_M" +int2str(mean_properties.minibatch) + ...
        "_h"+int2str(mean_properties.numhid2)+"-"+int2str(mean_properties.numhid3);
    te_err = [te_err;ss];
    te_err = [te_err;num2str(mean_properties.test_error)];
    
    load(convertCharsToStrings(sourceDir) + convertCharsToStrings(loadData(4).name),"mean_properties");
    ss = "D" + int2str(mean_properties.dropout)+ "_M" +int2str(mean_properties.minibatch) + ...
        "_h"+int2str(mean_properties.numhid2)+"-"+int2str(mean_properties.numhid3);
    te_err = [te_err;ss];
    te_err = [te_err;num2str(mean_properties.test_error)];
    C(:,end+1) = cellstr(te_err);
    sourceDir ='Evals/mean/';
end



















X = ["X";"X";"X";"X";];
TestError = table(X,te_err,te_err);