% configurations

% as of 01.06:

layer2 = [900,500,350,300,150];
layer3=[];
configurations_list = [];
%c_list_size = size(layer2,1) * size(layer3,1) * 2 * 2;
for i=1:size(layer2,2)
    if layer2(i) == 250 ||layer2(i) == 150
        for j=1:size(layer3,2)
                for z = 1:2
                    for u = 1:4
                        elem.layer2=layer2(i);
                        elem.layer3=layer3(j);
                        elem.dropout= (z == 1);
                        elem.minibatchsize= 6 * (u == 1) + 12 * (u == 2)+ 24 * (u == 3);
                        configurations_list = [configurations_list;elem];
                    end
                end
        end
        % the case of layer 2 as in the if statement above but without layer 3
        for z = 1:2
            for u = 1:3
                elem.layer2=layer2(i);
                elem.layer3=0;
                elem.dropout= (z == 1);
                elem.minibatchsize= 6 * (u == 1) + 12 * (u == 2)+ 24 * (u == 3);
                configurations_list = [configurations_list;elem];
            end
        end
    else
        for z = 1:2
            for u = 1:3
                elem.layer2=layer2(i);
                elem.layer3=0;
                elem.dropout= (z == 1);
                elem.minibatchsize= 6 * (u == 1) + 12 * (u == 2)+ 24 * (u == 3);
                configurations_list = [configurations_list;elem];
            end
        end
    end
end

% % if the 3 layer architectures should be at end excel file / separate
% for i=1:2
%     if i == 1
%         l2 = 150;
%     else 
%         l2 = 250;
%     end
%     for j=1:size(layer3,2)
%         for z = 1:2
%             for u = 1:2
%                 elem.layer2=l2;
%                 elem.layer3=layer3(j);
%                 elem.dropout= (z == 1);
%                 elem.minibatchsize= 6 * (u == 1) + 12 * (u == 2);
%                 configurations_list = [configurations_list;elem];
%             end
%         end
%     end
% end


