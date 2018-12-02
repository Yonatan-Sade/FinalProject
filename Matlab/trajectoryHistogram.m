clc;
close all;
clearvars;

withoutAlgoData = dlmread(' C:\Users\Public\Yonatan Sade\trajectory logs\21082018\trajectory - no galvo - 12 frames - 0.05mA - 31 pixel gaussian.txt');
x = withoutAlgoData(:, 1);
y = withoutAlgoData(:, 2);
figure(1), scatter(x, y);
title('without galvo');

histog = zeros(max(y) - min(y) + 1, max(x) - min(x) + 1);

for i = 1 : numel(x)
    deltaX = x(i) - min(x) + 1;
    deltaY = y(i) - min(y) + 1;
    histog(deltaY, deltaX) = histog(deltaY, deltaX) + 1;
end

[X, Y] = meshgrid(min(x) : max(x), min(y) : max(y));
surf(X, Y, histog);




withoutAlgoData = dlmread(' C:\Users\Public\Yonatan Sade\trajectory logs\21082018\trajectory - with galvo - 12 frames - 0.05mA - 31 pixel gaussian.txt');
x = withoutAlgoData(:, 1);
y = withoutAlgoData(:, 2);
figure(2), scatter(x, y);
title('without galvo');

histog = zeros(max(y) - min(y) + 1, max(x) - min(x) + 1);

for i = 1 : numel(x)
    deltaX = x(i) - min(x) + 1;
    deltaY = y(i) - min(y) + 1;
    histog(deltaY, deltaX) = histog(deltaY, deltaX) + 1;
end

[X, Y] = meshgrid(min(x) : max(x), min(y) : max(y));
surf(X, Y, histog);