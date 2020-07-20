function [Yn,c,X] = create_poly_toy(num_points, deg, SNR)
% regression problem polynomial of degree deg in nature with 1 independent var
% [Yn,~,X] = create_poly_toy(1000,3,5)

	x = randn([num_points,1]);
	[~,I] = sort(x);
	c = rand(deg+1,1);
    X=[];
    x1 = ones(size(x));
    for i=0:deg
        X = [x1 X];
        x1 = x1.*x;
    end
	y = X*c;
    oricoeff = c;
	Yn = y+randn([num_points,1])*SNR;
    dataset = [x Yn y];
	plot(x(I),y(I),'b-',x(I),Yn(I),'r.');
    save('polyRegression.mat','dataset','oricoeff');
end