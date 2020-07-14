function[c] = seprate_class(dataset)
    [~,n] = size(dataset);
    lab = unique(dataset(:,n));
    num_class = size(lab);
    c(num_class) = struct('label',[],'data',[],'count',[],'ori_index',[]);
    for i=1:num_class
        a = (dataset(:,n)==lab(i));
        c(i).data = dataset(a,1:n-1);
        c(i).label = lab(i);
        c(i).count = sum(a);
        [~,I] = sort(a,'descend');
        c(i).ori_index = I(1:c(i).count);
    end
end
