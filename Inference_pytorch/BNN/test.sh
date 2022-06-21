#!/bin/bash

date=`date +"%Y-%m-%d"`

cd test
mkdir ${date}
cd ..

time=`date +"%Y-%m-%d %T"`
echo "====start==== || ${time}"
python main_binary.py --dataset cifar10 --epochs 200 > ./test/${date}/log_hrr_${time}.txt 2>&1
time=`date +"%Y-%m-%d %T"`
echo "=====end===== || ${time}"
