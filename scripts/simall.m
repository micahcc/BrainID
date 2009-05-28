%matlab script file to take each set of const parameters
%and use them to run a timeseries.
%boldgen:
%Allowed options:
%-h [ --help ]          produce help message
%-o [ --out ] arg       image file to write to
%-t [ --outtime ] arg   How often to sample
%-s [ --simtime ] arg   Step size for sim, smaller is more accurate
%-e [ --endtime ] arg   What time to end at
%-n [ --numseries ] arg Number of brain regions to simulate
%-m [ --matlab ] arg    prefix for matlab files
%-i [ --inputstim ] arg file to read in stimuli from
%-v [ --noisevar ] arg  Variance of Gaussian Noise to apply to bold signal

load simstate.out
load state.out
load simmeas.out
origmeas = meassim;

close all
hold off
for i = 1 : length(statessim)
    mystring = sprintf('%s -i stim.in -t 2 -s .01 -e 1800 -m resim%04i -p "%f %f %f %f %f %f %f %f %f %f %f"\n', '../boldgen', i, states(i,2:8), statessim(1, 9:12))
    system(mystring);
end
mse = zeros(length(meassim),1);
for i = 1 : length(statessim)
    load(sprintf('resim%04imeas.out', i))
    mse(i) = sum((origmeas(:,3)-meassim(:,3)).^2/length(statessim));
    hold off
    plot(meassim(:,1), meassim(:,3), 'b', 'linewidth', 2)
    hold on
    plot(origmeas(:,1), origmeas(:,3), 'r', 'linewidth', 2)
    title(sprintf('i=%04i', i))
    print('-djpeg90', sprintf('im%04imeas.jpeg', i))
end
%
save('mse.out', 'mse')
