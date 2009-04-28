%Matlab m file to build up input stimulus file
t = 0:1:1800;
u = [0];
for i = 2:length(t)
    %flip u(i) from the previous
    %u(i) = floor(.999*sin((t(i)-900).^2/200)+1);
    u(i) = floor(.999*sin(((t(i)-900)*(2*pi)/200).^2)+1);
    %Set the level
    u(i) = 10*u(i);
    %u(i) = u(i)*unifrnd(0,10);
end

out = [t; u]';
save input.stim
