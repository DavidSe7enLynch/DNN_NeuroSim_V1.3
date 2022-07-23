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

sh ./layer_record_vgg_cifar10_binary/trace_command.sh > ./BNN/test/${date}/hwlog_hrr_${time}.txt 2>&1
