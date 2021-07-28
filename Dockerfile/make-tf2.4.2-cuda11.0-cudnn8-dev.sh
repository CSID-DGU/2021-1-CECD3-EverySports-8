#!/bin/bash

echo Create the docker container of tf2.4-dev \(2.4.2\)

sudo docker run -itd --gpus all --ipc=host --name tf2.4-dev -p 6000:8888 -v $DEFAULT_DIRECTORY/Data:/workspace/Data -v $DEFAULT_DIRECTORY/Developments:/workspace/Developments tf2.4-dev

sudo docker exec -it tf2.4-dev bash
