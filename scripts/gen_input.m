t = 0:5:1800;
u = [0];
for i = 2:length(t)
    u(i) = xor(u(i-1), mod(t(i),30) == 0);
end
u = 5*u;
out = [t; u]';
save input.stim
