#!/bin/bash

# useful arguments with default values
# wl_input = 8 (doesn't really exist)
# wl_activate = 8
# wl_error = 8
# wl_weight = 8
# inference = 0
# onoffratio = 10
# cellBit = 1
# subArray = 128
# ADCprecision = 5

wl_weight_array=(8 16 32)
subArray_array=(32 64 128 356 512 1024)
cellBit_array=(1 2 3 4 5 6)
ADCprecision_array=(1 3 5 7 9 11)
onoffratio_array=(2 6 10 15 20 50)

model_name="VGG8"
name="wl_weight"
is_linear=1
#time=`date +"%Y-%m-%d %T"`
#echo ${time}

# virtual environment
#conda activate DNN

# try to get more GPU space
#export CUDA_VISIBLE_DEVICES=1,2,3

# make a directory inside hrr_test named with date
date=`date +"%Y-%m-%d"`
cd ../hrr_test
mkdir ${date}
cd ../Inference_pytorch

for((i=0;i<${#wl_weight_array[*]};i++)) ; do
    time=`date +"%Y-%m-%d %T"`
    echo "====start==== || ${name}=${wl_weight_array[i]} || is_linear=${is_linear} || ${time}"
    python inference.py --model ${model_name} --${name} ${wl_weight_array[i]} --is_linear ${is_linear} > ../hrr_test/${date}/${model_name}_${name}=${wl_weight_array[i]}_is_linear=${is_linear}_${time}.txt 2>&1
    time=`date +"%Y-%m-%d %T"`
    echo "=====end===== || ${name}=${wl_weight_array[i]} || is_linear=${is_linear} || ${time}"
done
