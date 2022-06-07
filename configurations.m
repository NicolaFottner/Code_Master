% configurations

% as of 01.06:

layer2 = [900,300,10];
layer3 = [50,10];
configurations_list = [];
%c_list_size = size(layer2,1) * size(layer3,1) * 2 * 2;
for i=1:size(layer2,2)
    if layer2(i) == 900 ||layer2(i) == 300
        for j=1:size(layer3,2)
                for z = 1:2
                    for u = 1:2
                        elem.layer2=layer2(i);
                        elem.layer3=layer3(j);
                        elem.dropout= (z == 1);
                        elem.minibatchsize= 6 * (u == 1) + 12 * (u == 2);
                        configurations_list = [configurations_list;elem];
                    end
                end
        end
        % the case of layer 2 as in the if statement above but without layer 3
        for z = 1:2
            for u = 1:2
                elem.layer2=layer2(i);
                elem.layer3=0;
                elem.dropout= (z == 1);
                elem.minibatchsize= 6 * (u == 1) + 12 * (u == 2);
                configurations_list = [configurations_list;elem];
            end
        end
    else
        for z = 1:2
            for u = 1:2
                elem.layer2=layer2(i);
                elem.layer3=0;
                elem.dropout= (z == 1);
                elem.minibatchsize= 6 * (u == 1) + 12 * (u == 2);
                configurations_list = [configurations_list;elem];
            end
        end
    end
end

