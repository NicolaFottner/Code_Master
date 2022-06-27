
function [inner_t] = inner_targets_f(letter_id)

inner_t = zeros(letter_id);
for i=1:size(letter_id,1)
    if letter_id(i) == 1
        inner_t(i) = 6;
    elseif letter_id(i) == 2
        inner_t(i) = 4;
    elseif letter_id(i) == 3
        inner_t(i) = 5;
    elseif letter_id(i) == 4 
        inner_t(i) = 1;
    elseif letter_id(i) == 5
        inner_t(i) = 2;
    elseif letter_id(i) == 6
        inner_t(i) = 3;
    end
end


