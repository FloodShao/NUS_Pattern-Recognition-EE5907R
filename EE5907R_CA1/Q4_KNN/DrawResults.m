%%
%KNN results draw
figure(1);
hold on;
plot(LogFile(:,1),LogFile(:,2),'r','LineWidth',2); %bin test
plot(LogFile(:,1),LogFile(:,3),'r'); %bin train

plot(LogFile(:,1),LogFile(:,4),'b','LineWidth',2); %log test
plot(LogFile(:,1),LogFile(:,5),'b'); %log train

plot(LogFile(:,1),LogFile(:,6),'g','LineWidth',2); %norm test
plot(LogFile(:,1),LogFile(:,7),'g'); %norm train


xlabel('NO. of K');
ylabel('Error Rate');
title('Error VS K');
legend('binTest', 'binTrain', 'logTest', 'logTrain', 'normTest', 'normTrain');
grid on;


hold off;