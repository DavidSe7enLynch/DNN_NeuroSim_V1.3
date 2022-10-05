#!/bin/bash

date=`date +"%Y-%m-%d"`

cd test
mkdir ${date}
cd ..

#model="vgg_cifar10_binary"
model="alexnet_binary"

hw=0

# vgg cifar10
model_path="/home/rh539/DNN_NeuroSim_V1.3/Inference_pytorch/BNN/results/2022-06-21_18-05-06/model_best.pth.tar"
# alexnet cifar10
#model_path="/home/rh539/DNN_NeuroSim_V1.3/Inference_pytorch/BNN/results/2022-07-09_16-34-31/model_best.pth.tar"
  # new version
#model_path="/home/rh539/DNN_NeuroSim_V1.3/Inference_pytorch/BNN/results/2022-09-29_22-57-45/model_best.pth.tar"
  # tensorflow version
#model_path="/home/rh539/DNN_NeuroSim_V1.3/Inference_pytorch/BNN/results/2022-09-30_14-52-13/model_best.pth.tar"
# alexnet cifar10 non-binary
#model_path="/home/rh539/DNN_NeuroSim_V1.3/Inference_pytorch/BNN/results/2022-06-17_17-36-26/model_best.pth.tar"

time=`date +"%Y-%m-%d %T"`
echo "====start==== || ${model} || hw=${hw} || ${time}"
# train
python main_binary.py --dataset cifar10 --model ${model} --hw ${hw} --epochs 200 > ./test/${date}/trainlog_hrr_${model}_${time}.txt 2>&1
# inference
#python main_binary.py --hw ${hw} --model ${model} -e ${model_path} > ./test/${date}/inferencelog_hrr_model=${model}_hw=${hw}_${time}.txt 2>&1
#cp -r ./layer_record_alexnet_binary ..

time=`date +"%Y-%m-%d %T"`
echo "=====end===== || ${model} || hw=${hw} || ${time}"
