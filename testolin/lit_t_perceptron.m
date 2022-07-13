% ************************************************************************
% Simple perceptron implemented with the pseudo-inverse method
% NB: sometimes the backslash and the pinv give different results!
%
% Alberto Testolin
% Computational Cognitive Neuroscience Lab
% University of Padova
% ************************************************************************

function [weights, tr_acc,te_acc,tr_loss,te_loss] = lit_t_perceptron(a,tr_data,tr_targets,te_data,te_targets)

% add biases
ONES = ones(size(tr_data, 1), 1);  
tr_data = [tr_data ONES];
if a == 0 
    weights = tr_data\tr_targets;
else 
    % to apply the weight decay only on weights!, make bias term in eye = 0
    eye_m = eye(size(tr_data,2));
    eye_m(size(tr_data,2),size(tr_data,2)) = 0; 
    % perform learning
    A = tr_data' * tr_data + a * eye_m;
    B = tr_data';
    weights = A\B  * tr_targets;
end

% carefull, inversion numerically unstable, see murphy p369
% @todo: check if:
% inv(tr_patterns' * tr_patterns + a * eye) * tr_patterns' * tr_labels
% ===
% tr_patterns\tr_labels
% OR 
% (tr_labels'*pinv(tr_patterns'))';

%% Get accuracies
pred = tr_data*weights;
softmax_pred = softmax(dlarray(pred','CB'));
tr_loss = extractdata(crossentropy(softmax_pred,tr_targets'));
pred = extractdata(softmax_pred)';
[~, max_act] = max(pred,[],2);
[r,~] = find(tr_targets');
acc = (max_act == r);
tr_acc = mean(acc);

% test accuracy
ONES = ones(size(te_data, 1), 1);
te_data= [te_data ONES];
pred = te_data * weights;
softmax_pred = softmax(dlarray(pred','CB'));
te_loss = extractdata(crossentropy(softmax_pred,te_targets'));
pred = extractdata(softmax_pred)';
[~, max_act] = max(pred,[],2);
[r,~] = find(te_targets');
acc = (max_act == r);
te_acc = mean(acc);

end
