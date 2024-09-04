#!/bin/bash

name="test1"
window=15
usage=0.2
method="muOpt-H" # muOpt or VPA
webapp="Acmeair" # Acmeair or 3tier

gcloud container clusters get-credentials cluster-2 --region=northamerica-northeast1-a
rm ~/muOptK8s/ctrl/logs/$name/*
python3 ~/muOptK8s/ctrl/autoscaler.py -m $method -n $name -t $window -ut $usage -wa $webapp
