#!/bin/bash
CONTAINER_NAME="my_dmc_8g"
IMAGE_ID="ed631bf9e830"       # your images id

# 检查容器是否已存在（包括运行中和已停止的）
if docker ps -a --format '{{.Names}}' | grep -q "^${CONTAINER_NAME}$"; then
    echo "The container '${CONTAINER_NAME}' already exists."
    # 检查容器是否正在运行
    if docker ps --format '{{.Names}}' | grep -q "^${CONTAINER_NAME}$"; then
        echo "The container is running. Proceed directly..."
        docker exec -it ${CONTAINER_NAME} bash
    else
        echo "The container has been stopped and is now starting up..."
        docker start ${CONTAINER_NAME}
        docker exec -it ${CONTAINER_NAME} bash
    fi
else
    echo "The container does not exist. It is being created...."
    sudo docker run -it \
        --restart unless-stopped \
        -p 8080:22 \
        --shm-size=8g \
        --name ${CONTAINER_NAME} \
        --gpus all \
        -v $(pwd):/workspace \
        ${IMAGE_ID} bash
fi