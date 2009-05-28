#!/bin/bash
STIM="stim.in"
SAMPLERATE=2
SIMRATE=.001
ENDTIME=1800

function getstart()
{
LINE=`grep -m 1 -v '^#' simstate.out`
tmp=${LINE%%#*}
if [ -n "$tmp" ] 
    then
    simstart=${tmp#*.*.*.*.*.*.*.* }
fi
}

getstart
echo $simstart

#iterate through all the states, putting the estimated state at that time in as the 
#static parameters. Match those parameters with the correct initial points.
N=0
while read line; do
    tmp=${line%%#*}
    if [ -n "$tmp" ] 
    then
        N=$((N+1))
        params=${tmp#* }
        params=${params% *.*.*.*.*}
        echo $params $simstart > "tmpfile$N"
        tmp=$(printf '../boldgen -i %s -t %f -s %f -e %f -o resim%04d.nii.gz -f tmpfile%d\n' $STIM $SAMPLERATE $SIMRATE $ENDTIME $N $N)
        echo $tmp
        $tmp&
        let MODULUS=N%4
        if [ $MODULUS -eq 0 ]
        then
            wait
        fi
   fi
done < state.out

#calculate the MSE between each timeseries and the real timeseries
rm tmpfile*
for i in `seq 1 $N`; do
    tmp=$(printf 'fslmaths simseries.nii.gz -sub resim%04i.nii.gz -sqr -Tmean mse%04i.nii.gz' $i $i)
    echo $tmp
    $tmp&
    let MODULUS=i%4
    if [ $MODULUS -eq 0 ]; then
        wait
    fi
done

#generate a timeseries plot with the MSE
fslmerge -t mse.nii.gz mse*nii.gz
