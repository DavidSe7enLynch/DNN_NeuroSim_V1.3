#!/bin/bash

date=`date +"%Y-%m-%d"`

cd test
mkdir ${date}
cd ..

time=`date +"%Y-%m-%d %T"`
echo "====start==== || ${time}"
#python main_binary.py --dataset cifar10 --model vgg_cifar10_binary --epochs 200 > ./test/${date}/trainlog_hrr_${time}.txt 2>&1
python main_binary.py -e /home/rh539/DNN_NeuroSim_V1.3/Inference_pytorch/BNN/results/2022-06-21_18-05-06/model_best.pth.tar > ./test/${date}/inferencelog_hrr_${time}.txt 2>&1
time=`date +"%Y-%m-%d %T"`
echo "=====end===== || ${time}"
