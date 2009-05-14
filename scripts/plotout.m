load('meas.out')
load('simmeas.out')
load('state.out')
load('simstate.out')
load('cov.out')
figure(1)

labels = {'Tau ', 'Tau f', 'epsilon', 'Tau 0', 'alpha', 'E_0', 'V_0', 'V_t',...
            'Q_t', 'S_t', 'F_t'};

for i = 1:11
    subplot(6,2,i), plot(states(:,1), states(:,i+1), 'b','linewidth',2)
    hold on
    plot(statessim(:,1), statessim(:,i+1), 'r','linewidth',2)
    ylabel(labels{i})
    mymax = max(max(states(:,i+1)), max(statessim(:,i+1)));
    mymin = min(min(states(:,i+1)), min(statessim(:,i+1)));
    ylim([mymin-(mymax-mymin)/10 mymax+(mymax-mymin)/10])
end

subplot(6,2,12), plot(meassim(:,1), meassim(:,2)/100, 'g', 'linewidth', 2)
hold on
plot(bold(:,1), bold(:,3), 'b','linewidth',2)
plot(meassim(:,1), meassim(:,3), 'r','linewidth',1)
ylabel('Bold')
%legend('TAU_S', 'TAU_F', 'EPSILON', 'TAU_0', 'ALPHA', 'E_0', 'V_0', 'Volume', 'DeoxyHG', 'S_t', 'Flow')
%legend('Actual', 'Calculated')
print -depsc means.eps

covar = zeros(length(covariances(1,1,:)),length(covariances(1,:,1)));
for i = 1:11
    covar(:, i) = covariances(i, i, :);
end

figure(2)

for i = 1:11
    subplot(6,2,i), plot(states(:,1), covar(:,i), 'b','linewidth',2)
    hold on
    ylabel(labels{i})
end

print -depsc variance.eps
%
%%legend('TAU_S', 'TAU_F', 'EPSILON', 'TAU_0', 'ALPHA', 'E_0', 'V_0', 'Volume', 'DeoxyHG', 'S_t', 'Flow')
%%legend('Actual', 'Calculated')
