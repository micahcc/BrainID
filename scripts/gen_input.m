%Matlab m file to build up input stimulus file
t = 0:1:1800;
u = [0];
u = zeros(1, length(t));
for i = 200:length(t)-200
    %flip u(i) from the previous
    %u(i) = floor(.999*sin((t(i)-900).^2/200)+1);
    %u(i) = floor(.999*sin(((t(i)-900)*(2*pi)/200).^2)+1);
    u(i) = floor(.999*sin(t(i)/(4*pi))+1);
    %Set the level
    u(i) = 10*u(i);
    %u(i) = u(i)*unifrnd(0,10);
end

out = [t; u]';
save input.stim
