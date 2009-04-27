%Matlab m file to build up input stimulus file
t = 0:5:1800;
u = [0];
for i = 2:length(t)
    %flip u(i) from the previous
    u(i) = xor(u(i-1), mod(t(i),200) == 0);
    %Set the level
    u(i) = u(i)*unifrnd(0,10);
end

out = [t; u]';
save input.stim
