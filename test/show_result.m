%show_result
clc;
clf;
clear;
result = load('result.txt');
x = result(:,1);
y = result(:,2);
reallabel = result(:,3);
labelbyAl = result(:,4);

rl1 = unique(reallabel);
rl2 = unique(labelbyAl);

N1 = size(rl1);
N2 = size(rl2);

colorarr = ['b','g','r','c','m','y','k'];
subplot(1,2,1);
for i=1:N1
    dataix = x(reallabel==rl1(i));
    dataiy = y(reallabel==rl1(i));
    scatter(dataix,dataiy,colorarr(i));
    hold on;
end
title('data with the source real label')

subplot(1,2,2);
for i=1:N2
    dataix = x(labelbyAl==rl2(i));
    dataiy = y(labelbyAl==rl2(i));
    scatter(dataix,dataiy,colorarr(i));
    hold on;
end
title('data labeled by clustering algorithm')
saveas(gcf,'result.jpg');  

