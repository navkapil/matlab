function NonLinLbldNlbld_plot_decision_bndry(foldabledata,otherpara,optpara,Mdl,test_func)
% function NonLinLbldNlbld_plot_decision_bndry(u,lbldX,lbldY,ulbld,knl,M,mu,l)
% function will plot with respect to first two feature attributes

step =200;
figure(1)
hold on
lbldX = [];
for i = 1:length(foldabledata)
    lbldX = [lbldX;foldabledata(i).data];
end
lbldY = lbldX(:,end);
ulbld = otherpara.U;
% knl = otherpara.knl;
parameter = optpara.para;
% M = Mdl.M;
% u = Mdl.u;

plus = lbldY==1;
zero = lbldY==0;
neg = lbldY==-1;


minx1 = min([lbldX(:,1);ulbld(:,1)]); maxx1 = max([lbldX(:,1);ulbld(:,1)]);
minx2 = min([lbldX(:,2);ulbld(:,2)]); maxx2 = max([lbldX(:,2);ulbld(:,2)]);
x1 = (minx1:(maxx1-minx1)/(step-1):maxx1)';
x2 = (minx2:(maxx2-minx2)/(step-1):maxx2)';
[X1,X2] = meshgrid(x1,x2);
Xvec = [reshape(X1,[],1) reshape(X2,[],1)];
test_set = [Xvec ones(step*step,1)];

%% model prediction
     [f_test] = test_func(foldabledata,test_set,parameter,Mdl,otherpara);
%calculation of decision function
% K_test = knl(Xvec,trainX,knl_para);
% --f_test = [knl(Xvec,M,mu) ones(size(Xvec,1),1)]*u;%decision function
Y = reshape(f_test.PC,[step,step]);
delta = optpara.para(4);
%v = [-2,-1,-(1+delta)/2,-delta,0,delta,(1+delta)/2,1,2];
% contourf(X1,X2,Y,v,'ShowText','on');
contourf(X1,X2,Y,'ShowText','on');
% plot(ulbld(:,1),ulbld(:,2),'k.') % switch off when LSKSVCR is run and c5c must be zero
plot(lbldX(plus,1),lbldX(plus,2),'mo');
plot(lbldX(zero,1),lbldX(zero,2),'wo');
plot(lbldX(neg,1),lbldX(neg,2),'ro');


end