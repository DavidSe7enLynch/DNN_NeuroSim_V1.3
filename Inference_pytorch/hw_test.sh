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

#model='vgg_cifar10_binary'
#model='alexnet_binary'
#model='resnet_binary_tf'
model='densenet_binary_tf'

operationmode=6
cellBit=1
#technode="22nm SRAM 8T"
technode="22nm RRAM (Intel)"
#technode="22nm FeFET (GF)"

pipeline="true"
#pipeline="false"

#novelmapping="true"
novelmapping="false"

ADCprecision=3

wl_weight=1

#mv BNN/layer_record_${model} .

time=`date +"%Y-%m-%d %T"`
echo "====start==== || ${model} || ADCprec=${ADCprecision} || wl_weight=${wl_weight} || ${time}"

sh ./layer_record_${model}/trace_command.sh > ./BNN/test/${date}/hwlog_hrr_${model}_technode=${technode}_operationmode=${operationmode}_cellBit=${cellBit}_ADCprecision=${ADCprecision}_wl_weight=${wl_weight}_pipeline=${pipeline}_novelmapping=${novelmapping}_${time}.txt 2>&1

time=`date +"%Y-%m-%d %T"`
echo "=====end===== || ${model} || ADCprec=${ADCprecision} || wl_weight=${wl_weight} || ${time}"