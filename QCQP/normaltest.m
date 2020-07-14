function [ntest] = normaltest(test,mn,mx,range)
test = test-(ones(size(test,1),1)*mn);
diff = mx-mn;
ntest = (test./(ones(size(test,1),1)*diff))*(range(2)-range(1)) + range(1);
end