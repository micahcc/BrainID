load('meas.out')
load('simmeas.out')
load('state.out')
load('simstate.out')
load('cov.out')
figure(1)

subplot(6,2,1), plot(states(:,1), states(:,2), 'g')
hold on
plot(statessim(:,1), statessim(:,2), 'r')
ylabel('Tau s')

subplot(6,2,2), plot(states(:,1), states(:,3), 'g')
hold on
plot(statessim(:,1), statessim(:,3), 'r')
ylabel('Tau f')

subplot(6,2,3), plot(states(:,1), states(:,4), 'g')
hold on
plot(statessim(:,1), statessim(:,4), 'r')
ylabel('epsilon')

subplot(6,2,4), plot(states(:,1), states(:,5), 'g')
hold on
plot(statessim(:,1), statessim(:,5), 'r')
ylabel('Tau 0')

subplot(6,2,5), plot(states(:,1), states(:,6), 'g')
hold on
plot(statessim(:,1), statessim(:,6), 'r')
ylabel('alpha')

subplot(6,2,6), plot(states(:,1), states(:,7), 'g')
hold on
plot(statessim(:,1), statessim(:,7), 'r')
ylabel('E_0')

subplot(6,2,7), plot(states(:,1), states(:,8), 'g')
hold on
plot(statessim(:,1), statessim(:,8), 'r')
ylabel('V_0')

subplot(6,2,8), plot(states(:,1), states(:,9), 'g')
hold on
plot(statessim(:,1), statessim(:,9), 'r')
ylabel('Vt')

subplot(6,2,9), plot(states(:,1), states(:,10), 'g')
hold on
plot(statessim(:,1), statessim(:,10), 'r')
ylabel('Qt')

subplot(6,2,10), plot(states(:,1), states(:,11), 'g')
hold on
plot(statessim(:,1), statessim(:,11), 'r')
ylabel('St')

subplot(6,2,11), plot(states(:,1), states(:,12), 'g')
hold on
plot(statessim(:,1), statessim(:,12), 'r')
ylabel('Ft')

subplot(6,2,12), plot(bold(:,1), bold(:,3), 'g')
hold on
plot(meassim(:,1), meassim(:,3), 'r')
plot(meassim(:,1), meassim(:,2)/100, '.b')
ylabel('Bold')
%legend('TAU_S', 'TAU_F', 'EPSILON', 'TAU_0', 'ALPHA', 'E_0', 'V_0', 'Volume', 'DeoxyHG', 'S_t', 'Flow')
%legend('Actual', 'Calculated')

covar = zeros(length(covariances(1,1,:)),length(covariances(1,:,1)));
for i = 1:11
    covar(:, i) = covariances(i, i, :);
end

figure(2)

subplot(6,2,1), plot(states(:,1), covar(:,1), 'g')
hold on
ylabel('Tau s')

subplot(6,2,2), plot(states(:,1), covar(:,2), 'g')
hold on
ylabel('Tau f')

subplot(6,2,3), plot(states(:,1), covar(:,3), 'g')
hold on
ylabel('epsilon')

subplot(6,2,4), plot(states(:,1), covar(:,4), 'g')
hold on
ylabel('Tau 0')

subplot(6,2,5), plot(states(:,1), covar(:,5), 'g')
hold on
ylabel('alpha')

subplot(6,2,6), plot(states(:,1), covar(:,6), 'g')
hold on
ylabel('E_0')

subplot(6,2,7), plot(states(:,1), covar(:,7), 'g')
hold on
ylabel('V_0')

subplot(6,2,8), plot(states(:,1), covar(:,8), 'g')
hold on
ylabel('Vt')

subplot(6,2,9), plot(states(:,1), covar(:,9), 'g')
hold on
ylabel('Qt')

subplot(6,2,10), plot(states(:,1), covar(:,10), 'g')
hold on
ylabel('St')

subplot(6,2,11), plot(states(:,1), covar(:,11), 'g')
hold on
ylabel('Ft')
%
%%legend('TAU_S', 'TAU_F', 'EPSILON', 'TAU_0', 'ALPHA', 'E_0', 'V_0', 'Volume', 'DeoxyHG', 'S_t', 'Flow')
%%legend('Actual', 'Calculated')
