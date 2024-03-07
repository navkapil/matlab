close all
clear all

filename = "threecluster_triangle";
load(strcat(filename, "_structure.mat"));

c1 = 16; 
c2 = 2;
ep = 0.2;
mu = 0.5;
kernel = 'rbf'; %'linear'; %'rbf'

lblidx = 1;  %1,2,3
plbl = [1,2,3;2,3,1;3,1,2];  
labels = [c(plbl(lblidx,1)).orilabel, c(plbl(lblidx,2)).orilabel,...
            c(plbl(lblidx,3)).orilabel];

test = [c(plbl(lblidx,1)).testdata -ones(c(plbl(lblidx,1)).tstcnt,1);...
        c(plbl(lblidx,2)).testdata zeros(c(plbl(lblidx,2)).tstcnt,1);...
        c(plbl(lblidx,3)).testdata ones(c(plbl(lblidx,3)).tstcnt,1)];
m = size(test,1);
test = test(randperm(m),:);

[model] = KSVCR(c(plbl(lblidx,1)).traindata, ...
    c(plbl(lblidx,2)).traindata,c(plbl(lblidx,3)).traindata, ...
    c1,c2,ep,mu,kernel);
[acc,PC,score] = test_KSVCR(model,test(:,1:end-1),test(:,end));
[confusionMatrix, precision,recall,fscore] = calPerPara(test(:,end), PC);

save(strcat(filename,num2str(lblidx),num2str(c1),"_results.mat"),"c","c1","c2","ep","labels","PC","test","confusionMatrix","fscore","recall","precision","score","acc");

%% only for 2 domensional data
figure(1);
hold on;
set(groot,'defaultLineLineWidth',1.5);
xlim([-1.5 1.5]);
ylim([-1.5 1.5]);
set(gca,'XTick',(-1.5:.5:1.5));
set(gca,'YTick',(-1.5:.5:1.5));
grid on
w = plot_decision_bndry(model);
centb = mean(c(plbl(lblidx,1)).traindata);
centr = mean(c(plbl(lblidx,2)).traindata);
centg = mean(c(plbl(lblidx,3)).traindata);

plot(c(plbl(lblidx,1)).traindata(:,1),c(plbl(lblidx,1)).traindata(:,2),'bo');
plot(c(plbl(lblidx,2)).traindata(:,1),c(plbl(lblidx,2)).traindata(:,2),'ro');
plot(c(plbl(lblidx,3)).traindata(:,1),c(plbl(lblidx,3)).traindata(:,2),'go');
plot([0 w(1)],[0 w(2)],'k->');
set(groot,'defaultLineLineWidth',3);
plot([centb(1);centr(1);centg(1)],[centb(2);centr(2);centg(2)],'khexagram')
savefig(filename);
saveas(gcf,filename,'pdf');