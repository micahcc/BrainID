load('meas.out')
load('simmeas.out')
load('state.out')
load('simstate.out')
load('cov.out')
figure(1)

subplot(6,2,1), plot(states(:,1), states(:,2), 'b','linewidth',2)
hold on
plot(statessim(:,1), statessim(:,2), 'r','linewidth',2)
ylabel('Tau s')

subplot(6,2,2), plot(states(:,1), states(:,3), 'b','linewidth',2)
hold on
plot(statessim(:,1), statessim(:,3), 'r','linewidth',2)
ylabel('Tau f')

subplot(6,2,3), plot(states(:,1), states(:,4), 'b','linewidth',2)
hold on
plot(statessim(:,1), statessim(:,4), 'r','linewidth',2)
ylabel('epsilon')

subplot(6,2,4), plot(states(:,1), states(:,5), 'b','linewidth',2)
hold on
plot(statessim(:,1), statessim(:,5), 'r','linewidth',2)
ylabel('Tau 0')

subplot(6,2,5), plot(states(:,1), states(:,6), 'b','linewidth',2)
hold on
plot(statessim(:,1), statessim(:,6), 'r','linewidth',2)
ylabel('alpha')

subplot(6,2,6), plot(states(:,1), states(:,7), 'b','linewidth',2)
hold on
plot(statessim(:,1), statessim(:,7), 'r','linewidth',2)
ylabel('E_0')

subplot(6,2,7), plot(states(:,1), states(:,8), 'b','linewidth',2)
hold on
plot(statessim(:,1), statessim(:,8), 'r','linewidth',2)
ylabel('V_0')

subplot(6,2,8), plot(states(:,1), states(:,9), 'b','linewidth',2)
hold on
plot(statessim(:,1), statessim(:,9), 'r','linewidth',2)
ylabel('Vt')

subplot(6,2,9), plot(states(:,1), states(:,10), 'b','linewidth',2)
hold on
plot(statessim(:,1), statessim(:,10), 'r','linewidth',2)
ylabel('Qt')

subplot(6,2,10), plot(states(:,1), states(:,11), 'b','linewidth',2)
hold on
plot(statessim(:,1), statessim(:,11), 'r','linewidth',2)
ylabel('St')

subplot(6,2,11), plot(states(:,1), states(:,12), 'b','linewidth',2)
hold on
plot(statessim(:,1), statessim(:,12), 'r','linewidth',2)
ylabel('Ft')

subplot(6,2,12), plot(bold(:,1), bold(:,3), 'b','linewidth',2)
hold on
plot(meassim(:,1), meassim(:,3), 'r','linewidth',2)
plot(meassim(:,1), meassim(:,2)/100, 'g', 'linewidth', 2)
ylabel('Bold')
%legend('TAU_S', 'TAU_F', 'EPSILON', 'TAU_0', 'ALPHA', 'E_0', 'V_0', 'Volume', 'DeoxyHG', 'S_t', 'Flow')
%legend('Actual', 'Calculated')
print -depsc means.eps

covar = zeros(length(covariances(1,1,:)),length(covariances(1,:,1)));
for i = 1:11
    covar(:, i) = covariances(i, i, :);
end

figure(2)

subplot(6,2,1), plot(states(:,1), covar(:,1), 'b','linewidth',2)
hold on
ylabel('Tau s')

subplot(6,2,2), plot(states(:,1), covar(:,2), 'b','linewidth',2)
hold on
ylabel('Tau f')

subplot(6,2,3), plot(states(:,1), covar(:,3), 'b','linewidth',2)
hold on
ylabel('epsilon')

subplot(6,2,4), plot(states(:,1), covar(:,4), 'b','linewidth',2)
hold on
ylabel('Tau 0')

subplot(6,2,5), plot(states(:,1), covar(:,5), 'b','linewidth',2)
hold on
ylabel('alpha')

subplot(6,2,6), plot(states(:,1), covar(:,6), 'b','linewidth',2)
hold on
ylabel('E_0')

subplot(6,2,7), plot(states(:,1), covar(:,7), 'b','linewidth',2)
hold on
ylabel('V_0')

subplot(6,2,8), plot(states(:,1), covar(:,8), 'b','linewidth',2)
hold on
ylabel('Vt')

subplot(6,2,9), plot(states(:,1), covar(:,9), 'b','linewidth',2)
hold on
ylabel('Qt')

subplot(6,2,10), plot(states(:,1), covar(:,10), 'b','linewidth',2)
hold on
ylabel('St')

subplot(6,2,11), plot(states(:,1), covar(:,11), 'b','linewidth',2)
hold on
ylabel('Ft')
print -depsc variance.eps
%
%%legend('TAU_S', 'TAU_F', 'EPSILON', 'TAU_0', 'ALPHA', 'E_0', 'V_0', 'Volume', 'DeoxyHG', 'S_t', 'Flow')
%%legend('Actual', 'Calculated')
