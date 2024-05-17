#!/bin/bash

name="test1"
window=15
usage=0.5

gcloud container clusters get-credentials cluster-1 --region=northamerica-northeast1-a
rm ./logs/$name/*
python3 muopt.py -n $name -t $window -ut $usage