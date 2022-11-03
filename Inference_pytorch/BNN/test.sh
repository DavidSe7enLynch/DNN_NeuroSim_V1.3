#!/bin/bash

date=`date +"%Y-%m-%d"`

#scartchPath=/scratch-x3
scartchPath="/home/rh539/DNN_NeuroSim_V1.3/Inference_pytorch/BNN"
BNNPath="/home/rh539/DNN_NeuroSim_V1.3/Inference_pytorch/BNN"

mkdir ${scartchPath}/test
echo ${scartchPath}/test/${date}

cd ${scartchPath}/test
mkdir ${date}
cd ${BNNPath}

#model="vgg_cifar10_binary"
model="alexnet_binary"

#dataset="cifar10"
dataset="imagenet"

hw=1

# vgg cifar10
#model_path="/home/rh539/DNN_NeuroSim_V1.3/Inference_pytorch/BNN/results/2022-06-21_18-05-06/model_best.pth.tar"
# alexnet cifar10
#model_path="/home/rh539/DNN_NeuroSim_V1.3/Inference_pytorch/BNN/results/2022-07-09_16-34-31/model_best.pth.tar"
  # new version
#model_path="/home/rh539/DNN_NeuroSim_V1.3/Inference_pytorch/BNN/results/2022-09-29_22-57-45/model_best.pth.tar"
  # tensorflow version
#model_path="/home/rh539/DNN_NeuroSim_V1.3/Inference_pytorch/BNN/results/2022-10-05_01-13-20/model_best.pth.tar"
  # imagenet 100 epoch
model_path="/home/rh539/DNN_NeuroSim_V1.3/Inference_pytorch/BNN/results/2022-10-25_20-56-17/model_best.pth.tar"
# alexnet cifar10 non-binary
#model_path="/home/rh539/DNN_NeuroSim_V1.3/Inference_pytorch/BNN/results/2022-06-17_17-36-26/model_best.pth.tar"

time=`date +"%Y-%m-%d %T"`
echo "====start==== || ${model} || hw=${hw} || ${time}"
# train
#python main_binary.py --dataset ${dataset} --model ${model} --hw ${hw} --epochs 100 > ./test/${date}/trainlog_hrr_${model}_dataset=${dataset}_${time}.txt 2>&1
# inference
python main_binary.py --hw ${hw} --model ${model} --dataset ${dataset} -e ${model_path} > ./test/${date}/inferencelog_hrr_model=${model}_dataset=${dataset}_hw=${hw}_${time}.txt 2>&1
#cp -r ./layer_record_alexnet_binary ..

time=`date +"%Y-%m-%d %T"`
echo "=====end===== || ${model} || hw=${hw} || ${time}"
