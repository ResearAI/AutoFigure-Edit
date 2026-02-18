# build env
docker build -f docker/Dockerfile -t autofigure:latest .
# create
docker run --name autofigure --gpus all --shm-size 32g -p 30001:30000 --ipc=host -v MODEL_PATH:/root/models -v CODE_PATH:/app/ -it autofigure:latest /bin/bash