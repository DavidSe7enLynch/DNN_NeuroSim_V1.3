#!/bin/bash

date=`date +"%Y-%m-%d"`

cd BNN/test
mkdir ${date}
cd ../..

# remake Neurosim
cd NeuroSIM
make clean
make
cd ..

# test
time=`date +"%Y-%m-%d %T"`

#model='vgg_cifar10_binary'
model='alexnet_binary'
operationmode=6
cellBit=1
technode="22nm RRAM (Intel)"

sh ./layer_record_${model}/trace_command.sh > ./BNN/test/${date}/hwlog_hrr_${model}_technode=${technode}_operationmode=${operationmode}_cellBit=${cellBit}_${time}.txt 2>&1
