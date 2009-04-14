load('meas.out')
load('state.out')
load('simstate.out')
figure(1)

subplot(6,2,1), plot(states(:,1), states(:,2), 'g')
hold on
plot(statessim(:,1), statessim(:,3), 'r')
ylabel('Tau s')

subplot(6,2,2), plot(states(:,1), states(:,3), 'g')
hold on
plot(statessim(:,1), statessim(:,4), 'r')
ylabel('Tau f')

subplot(6,2,3), plot(states(:,1), states(:,4), 'g')
hold on
plot(statessim(:,1), statessim(:,5), 'r')
ylabel('epsilon')

subplot(6,2,4), plot(states(:,1), states(:,5), 'g')
hold on
plot(statessim(:,1), statessim(:,6), 'r')
ylabel('Tau 0')

subplot(6,2,5), plot(states(:,1), states(:,6), 'g')
hold on
plot(statessim(:,1), statessim(:,7), 'r')
ylabel('alpha')

subplot(6,2,6), plot(states(:,1), states(:,7), 'g')
hold on
plot(statessim(:,1), statessim(:,8), 'r')
ylabel('E_0')

subplot(6,2,7), plot(states(:,1), states(:,8), 'g')
hold on
plot(statessim(:,1), statessim(:,9), 'r')
ylabel('V_0')

subplot(6,2,8), plot(states(:,1), states(:,9), 'g')
hold on
plot(statessim(:,1), statessim(:,10), 'r')
ylabel('Vt')

subplot(6,2,9), plot(states(:,1), states(:,10), 'g')
hold on
plot(statessim(:,1), statessim(:,11), 'r')
ylabel('Qt')

subplot(6,2,10), plot(states(:,1), states(:,11), 'g')
hold on
plot(statessim(:,1), statessim(:,12), 'r')
ylabel('St')

subplot(6,2,11), plot(states(:,1), states(:,12), 'g')
hold on
plot(statessim(:,1), statessim(:,13), 'r')
ylabel('Ft')

subplot(6,2,12), plot(bold(:,1), bold(:,2), 'g')
hold on
plot(bold(:,1), bold(:,3), 'r')
ylabel('Bold')
%legend('TAU_S', 'TAU_F', 'EPSILON', 'TAU_0', 'ALPHA', 'E_0', 'V_0', 'Volume', 'DeoxyHG', 'S_t', 'Flow')
%legend('Actual', 'Calculated')
