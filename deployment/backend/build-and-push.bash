#!/bin/bash


if [ -e Dockerfile ]; then
    # remove old repititions if they exist
    if [ -e app/models ]; then
        rm -r app/models
    fi
    if [ -e app/configs ]; then
        rm -r app/configs
    fi
    if [ -e app/src ]; then
        rm -r app/src
    fi

    # copy repetitions
    cp -r ../../models app/models
    cp -r ../../configs app/configs
    cp -r ../../src app/src

    # log into docker
    aws ecr get-login-password --region ap-south-1 | sudo docker login --username AWS --password-stdin 065257926712.dkr.ecr.ap-south-1.amazonaws.com

    # build and push
    sudo docker build -t 065257926712.dkr.ecr.ap-south-1.amazonaws.com/xai:latest .
    sudo docker push 065257926712.dkr.ecr.ap-south-1.amazonaws.com/xai:latest
else
    echo "Please change the working directory to the directory containing the Dockerfile"
    exit 1
fi