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

#for((i=0;i<${#wl_weight_array[*]};i++)) ; do
#    time=`date +"%Y-%m-%d %T"`
#    echo "====start==== || ${name}=${wl_weight_array[i]} || is_linear=${is_linear} || ${time}"
#    python inference.py --model ${model_name} --${name} ${wl_weight_array[i]} --is_linear ${is_linear} > ../hrr_test/${date}/${model_name}_${name}=${wl_weight_array[i]}_is_linear=${is_linear}_${time}.txt 2>&1
#    time=`date +"%Y-%m-%d %T"`
#    echo "=====end===== || ${name}=${wl_weight_array[i]} || is_linear=${is_linear} || ${time}"
#done

# BNN test
#is_linear=1
wl_weight=8
wl_activate=8
cellBit=4
ADCprecision=5
inference=1
model_bit=8
model="best-140.pth"
quant_version="old"

time=`date +"%Y-%m-%d %T"`
echo "====start==== || quant_version=${quant_version} || ${model_name} || model=${model} || model_bit=${model_bit} || inference=${inference} || is_linear=${is_linear} || wl_weight=${wl_weight} || wl_activate=${wl_activate} || cellBit=${cellBit} || ADCprecision=${ADCprecision} || ${time}"
#test
python inference.py --model ${model_name} --inference ${inference} --is_linear ${is_linear} --wl_weight ${wl_weight} --wl_activate ${wl_activate} --cellBit ${cellBit} --ADCprecision ${ADCprecision} > ../hrr_test/${date}/inference_quant_version=${quant_version}_${model_name}_model=${model}_model_bit=${model_bit}_inference=${inference}_is_linear=${is_linear}_wl_weight=${wl_weight}_wl_activate=${wl_activate}_cellBit=${cellBit}_ADCprecision=${ADCprecision}_${time}.txt 2>&1
#train
#python train.py --model ${model_name} --inference ${inference} --is_linear ${is_linear} --wl_weight ${wl_weight} --wl_activate ${wl_activate} --cellBit ${cellBit} --ADCprecision ${ADCprecision} > ../hrr_test/${date}/train_${model_name}_is_linear=${is_linear}_wl_weight=${wl_weight}_wl_activate=${wl_activate}_cellBit=${cellBit}_ADCprecision=${ADCprecision}_${time}.txt 2>&1
time=`date +"%Y-%m-%d %T"`
echo "=====end===== || quant_version=${quant_version} || ${model_name} || model=${model} || model_bit=${model_bit} || inference=${inference} || is_linear=${is_linear} || wl_weight=${wl_weight} || wl_activate=${wl_activate} || cellBit=${cellBit} || ADCprecision=${ADCprecision} || ${time}"

