#!/bin/bash
#!/usr/bin/env bash

date=$(date +"%Y-%m-%d")

#scartchPath=/scratch-x3
scartchPath="/home/rh539/DNN_NeuroSim_V1.3/Inference_pytorch/BNN"
BNNPath="/home/rh539/DNN_NeuroSim_V1.3/Inference_pytorch/BNN"

mkdir ${scartchPath}/test
echo ${scartchPath}/test/${date}

cd ${scartchPath}/test
mkdir ${date}
cd ${BNNPath}

#model="vgg_cifar10_binary"
#model="alexnet_binary"
#model="resnet_binary"
#model="resnet"
#model="resnet_binary_orig"
#model="resnet_binary_tf"
model="densenet_binary_tf"

#dataset="cifar10"
dataset="imagenet"

hw=0

#ADCprecision_array=(6 7 8 9)
ADCprec=7
wl_input=8

# vgg cifar10
#model_path="/home/rh539/DNN_NeuroSim_V1.3/Inference_pytorch/BNN/results/2022-06-21_18-05-06/model_best.pth.tar"

# alexnet cifar10
#model_path="/home/rh539/DNN_NeuroSim_V1.3/Inference_pytorch/BNN/results/2022-07-09_16-34-31/model_best.pth.tar"
# alexnet new version
#model_path="/home/rh539/DNN_NeuroSim_V1.3/Inference_pytorch/BNN/results/2022-09-29_22-57-45/model_best.pth.tar"
# alexnet tensorflow version
#model_path="/home/rh539/DNN_NeuroSim_V1.3/Inference_pytorch/BNN/results/2022-10-05_01-13-20/model_best.pth.tar"
# alexnet imagenet 100 epoch
#model_path="/home/rh539/DNN_NeuroSim_V1.3/Inference_pytorch/BNN/results/2022-10-25_20-56-17/model_best.pth.tar"
# alexnet cifar10 non-binary
#model_path="/home/rh539/DNN_NeuroSim_V1.3/Inference_pytorch/BNN/results/2022-06-17_17-36-26/model_best.pth.tar"

# resnet binary 1 epoch hw=0
#model_path="/home/rh539/DNN_NeuroSim_V1.3/Inference_pytorch/BNN/results/2022-12-08_23-03-59/model_best.pth.tar"


time=$(date +"%Y-%m-%d %T")
echo "====start==== || ${model} || dataset=${dataset} || hw=${hw} || ADCprec=${ADCprec} || wl_input=${wl_input} || ${time}"
# train
python main_binary.py --dataset ${dataset} --model ${model} --hw ${hw} -b 40 --epochs 120 > ./test/${date}/trainlog_hrr_${model}_dataset=${dataset}_${time}.txt 2>&1

# inference
#for ((i = 0; i < ${#ADCprecision_array[*]}; i++)); do
#  python main_binary.py --hw ${hw} --ADCprec ${ADCprecision_array[i]} --model ${model} --dataset ${dataset} -e ${model_path} > ./test/${date}/inferencelogHRRmodel=${model}_dataset=${dataset}_hw=${hw}_ADCprec=${ADCprecision_array[i]}_${time}.txt 2>&1
#done

# run in parallel
#python main_binary.py --hw ${hw} --ADCprec ${ADCprec} --wl_input ${wl_input} --model ${model} --dataset ${dataset} -e ${model_path} > ./test/${date}/inferencelogHRRmodel=${model}_dataset=${dataset}_hw=${hw}_ADCprec=${ADCprec}_wl_input=${wl_input}_${time}.txt 2>&1

#rm -rf ../layer_record_${model}
#cp -r ./layer_record_${model} ..

time=$(date +"%Y-%m-%d %T")
echo "=====end===== || ${model} || dataset=${dataset} || hw=${hw} || ADCprec=${ADCprec} || wl_input=${wl_input} || ${time}"
