#!/bin/bash

name="test1"
window=15
usage=0.95

gcloud container clusters get-credentials cluster-2 --region=northamerica-northeast1-a
rm ./logs/$name/*
python3 autoscaler.py -m muOpt -n $name -t $window -ut $usage