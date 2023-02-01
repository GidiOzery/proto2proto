#!/bin/sh

# RANDOMN=$$
# NOTRANDOM=org
# runName=eval_setting
# newRun=false
# serviceType=evaluate
# ablationType=Resnet50_18_cars
# ablationType=VGG19_VGG11_birds


# mkdir Experiments/$ablationType/$runName/$NOTRANDOM
# expOut=Experiments/$ablationType/$runName/$NOTRANDOM/$NOTRANDOM.out
# errorOut=Experiments/$ablationType/$runName/$NOTRANDOM/error+$RANDOMN.out

# cp Experiments/$ablationType/$runName/args.yaml Experiments/$ablationType/$runName/$NOTRANDOM/args.yaml

# CUDA_VISIBLE_DEVICES=1,2,3,4 python -u main.py $runName $newRun $serviceType $NOTRANDOM $ablationType > $expOut 2>$errorOut