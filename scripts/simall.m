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

%for i = 1 : length(statessim)
%%    fprintf('Running: %s -t 2 -s .01 -e 1800 -m sim%03i -p "%f %f %f %f %f %f %f %f %f %f %f"\n', '../boldgen ', i, states(i,2:8), statessim(1, 9:12));
%    mystring = sprintf('%s -t 2 -s .01 -e 1800 -m resim%03i -p "%f %f %f %f %f %f %f %f %f %f %f"\n', '../boldgen', i, states(i,2:8), statessim(1, 9:12))
%    system(mystring);
%end
mse = zeros(length(meassim),1);
for i = 1 : length(statessim)
    mystring = sprintf('resim%03imeas.out', i);
    load(mystring);
    mse(i) = sum((meassim(:,3) - origmeas(:,3)).^2)/length(meassim);
end

save('mse.out', 'mse')
