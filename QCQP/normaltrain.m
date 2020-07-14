function [ntrain,mn,mx] = normaltrain(train,range)
mx = max(train);
mn = min(train);
train = train-(ones(size(train,1),1)*mn);
diff = mx-mn;
ntrain = (train./(ones(size(train,1),1)*diff))*(range(2)-range(1)) + range(1);
end

