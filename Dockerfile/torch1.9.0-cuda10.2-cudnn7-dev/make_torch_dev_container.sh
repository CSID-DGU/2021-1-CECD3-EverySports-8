#!/bin/bash

echo Create the docker container of torch-dev

sudo docker run -itd --gpus all --ipc=host --name torch-dev -p 5000:8888 -v /home/uijeong/Data:/workspace/Data -v /home/uijeong/Developments:/workspace/Developments torch1.9-jupyter

sudo docker exec -it torch-dev bash
