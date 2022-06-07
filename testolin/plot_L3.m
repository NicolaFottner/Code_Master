% ************************************************************************
% Plot the receptive fields (i.e., connection weights) of the third layer
%
% Alberto Testolin
% Computational Cognitive Neuroscience Lab
% University of Padova
% ************************************************************************

function [] = plot_L3(DN, n_hidden,ff)
numhid = n_hidden;
dd = strsplit(date,'-');c = clock;clean_date = strcat(dd(1),dd(2)); %without "-YYYY"
f= figure;
[v,h] = size(DN.L{1}.vishid);  % number of visible and hidden units
imgsize = sqrt(v);
if n_hidden > h
    n_hidden = h;
end
n_x = floor(sqrt(n_hidden)); n_y = n_x;
n_hidden = n_x * n_y;

for i_n = 1:n_hidden
    % Select (strong) inputs to L3(i_n)
    W3 = DN.L{3}.vishid(:,i_n);
    W3 = W3 .* (abs(W3) > 0.0);
    % Select (strong) inputs to L2
    W2 = DN.L{2}.vishid;
    W2 = W2 .* (abs(W2) > 0.0);
    % Select (strong) inputs to L1
    W1 = DN.L{1}.vishid;
    W1 = W1 .* (abs(W1) > 0.0);
    
    % Weight filters with a linear combination
    ww = W1*W2;
    ww = ww*W3;
    ww = ww .* (abs(ww) > 0.0);    % threshold
    %ww=ww/max(abs(ww));
    
    pl = subplot(n_y,n_x,i_n);
    position = get(pl, 'pos');
    position(3) = position(3) + 0.004;
    position(4) = position(4) + 0.004;
    set(pl, 'pos', position);
    imagesc(reshape(ww,imgsize,imgsize)); %(1:end-2)
    colormap('gray'); axis square; axis off;
end
file_namee = "Evals/fig/" + "E_" + int2str(ff) + "H3_" + int2str(numhid) + "_" + clean_date + "_" + int2str(c(4)) + "h" + int2str(c(5))+"m_"+ ".pdf";
exportgraphics(f,file_namee);