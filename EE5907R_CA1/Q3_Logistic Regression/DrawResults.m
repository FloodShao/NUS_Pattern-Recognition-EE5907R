%Draw the results of Logistic Regression
load('Logfile.mat');

figure(1);
hold on;

plot(Logfile(:,1), Logfile(:,2), 'b', 'LineWidth', 2);
plot(Logfile(:,1), Logfile(:,3), 'b');
plot(Logfile(:,1), Logfile(:,5), 'r', 'LineWidth', 2);
plot(Logfile(:,1), Logfile(:,6), 'r');
plot(Logfile(:,1), Logfile(:,8), 'g', 'LineWidth', 2);
plot(Logfile(:,1), Logfile(:,9), 'g');
title('testErrorRate VS normalizeParameter');
xlabel('lambda');
ylabel('testErrorRate');
legend('onlyBinTest', 'onlyBinTrain', 'onlyLogTest', 'onlyLogTrain', 'onlyZnormalizeTest', 'onlyZnormalizeTrain');
hold off;

figure(2);
hold on;
plot(Logfile(:,1), Logfile(:,4), 'b');
plot(Logfile(:,1), Logfile(:,7), 'r');
plot(Logfile(:,1), Logfile(:,10), 'g');
title('iterationNum VS normalizeParameter');
xlabel('lambda');
ylabel('iterationNum');
legend('onlyBin', 'onlyLog', 'onlyZnormalize');
hold off;