function [Pos,Neg]= create_binary_class_data(separated_data, class_indx)
% Uses the output of seprate_class, the structure defined as
% c(num_class) = struct('label',[],'data',[],'count',[],'ori_index',[]);
    Pos = separated_data(class_indx).data;
    Neg = [];
    for i = 1:length(separated_data)
        if i ~= class_indx
            Neg = [Neg;separated_data(i).data];
        end
    end
end