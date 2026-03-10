# 🦷牙冠自动生成项目

本项目为基于3D点云的牙冠自动生成功能。
原项目[DMC](https://github.com/Golriz-code/DMC)使用PLY格式的点云数据，并缺少模块。
现已修改为支持**TXT格式**和**PLY格式**的点云数据集，可通过导入镜像文件在docker容器中运行该项目。

## 优化进度，持续更新中...
2026.03.10(分支：crown_box)：
```txt
1.使用真实牙冠min_gt和max_gt对牙冠的泊松表面重建数据psr.npy进行归一化。
```
### 训练效果
| 配置    | Value |
|---------|-------|
| max epoch | 100 |
| batch size | 8 |
| points | 2048 |
| Learning rate | 0.0005 |
| train/test | 773/86 |



| Metric     | Value |
|------------|-------|
| chamfer L1 | 0.0832 |
| chamfer L2 | 0.0173 |
| metric MSE | 0.0038 |


## 1. 在docker中运行项目

### 1.1 获得文件

```txt
- whc_pytorch271_cuda118_dmc:v4.0.tar # docker镜像文件
- DMC_TXT.tar                    # 项目文件
- data_ply.tar                   # 数据集
```

### 1.2 解压项目

解压项目文件 DMC_TXT.tar

```bash
sudo tar -xvf DMC_TXT.tar
```

解压后文件结构：

```shell
root@RJZ-WF20240312:/home/DMC_TXT# tree -L 1
.
├── SAP
├── cfgs
├── data
├── datasets
├── extensions
├── main.py
├── models
├── pointnet2_ops_lib
├── pytorch3d
├── readme.txt
├── rm_psr_npz.sh
├── run_docker_cuda118.sh
├── run_main.sh
├── tools
└── utils
```

### 1.2 导入镜像

将`whc_pytorch271_cuda118_dmc.tar`导入到本地镜像库

```
sudo docker load -i whc_pytorch271_cuda118_dmc.tar
```

导入完成查看导入的镜像信息

```shell
sudo docker images
```

输出：

```.txt
IMAGE                           ID               DISK USAGE   CONTENT SIZE   EXTRA
whc_pytorch271_cuda118_dmc:v4.0 e604aeaae42d       31.6GB         10.3GB       U
```

### 1.2 创建容器

1、进入DMC_TXT目录下

```shell
cd DMC_TXT
```

2、修改`run_docker_cuda118.sh`文件中的变量`CONTAINER_NAME`和`IMAGE_ID`。

- CONTAINER_NAME 是容器名字，自定义
- IMAGE_ID 名字可以使用 `sudo docker images`查看到

3、运行脚本，直接进入容器中。

```shell
./run_docker_cuda118.sh
```

run_docker_cuda118.sh内容如下：

```shell
#!/bin/bash
CONTAINER_NAME="my_cuda118_dmc_all_8g_v4.0"
IMAGE_ID="e604aeaae42d"       # your images id

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
        -p 8089:22 \
        --shm-size=8g \
        --name ${CONTAINER_NAME} \
        --gpus all \
        -v $(pwd):/workspace \
        ${IMAGE_ID} bash
fi
```

参数详解

| 参数                       | 全称            | 含义                                                         |
| :------------------------- | :-------------- | :----------------------------------------------------------- |
| `sudo`                     | -               | 以超级用户权限运行（需要 root 权限操作 Docker）              |
| `docker run`               | -               | 创建并启动一个新容器                                         |
| `-i`                       | `--interactive` | 保持 STDIN 打开，允许交互输入                                |
| `-t`                       | `--tty`         | 分配一个伪终端（TTY），支持命令行交互                        |
| `--restart unless-stopped` | -               | 重启策略：容器总是自动重启，除非手动停止                     |
| `-p 8080:22`               | `--publish`     | 端口映射：主机 8080 端口 → 容器 22 端口（SSH）               |
| `--shm-size=8g`            | -               | 共享内存大小：设置 `/dev/shm` 为 8GB（默认 64MB，深度学习常用） |
| `--name my_dmc_8g`         | -               | 容器名称：给容器起个名字叫 `my_dmc_8g`                       |
| `--gpus all`               | -               | GPU 支持：允许容器使用所有可用的 NVIDIA GPU                  |
| `-v $(pwd):/workspace`     | `--volume`      | 目录挂载：将当前目录挂载到容器的 `/workspace`                |
| `ed631bf9e830`             | -               | 镜像 ID：使用的 Docker 镜像（可以是镜像名或 ID）             |
| `bash`                     | -               | 启动命令：容器启动后执行 bash  shell                         |

## 2. 数据集准备

> 在容器中进行，容器中有执行数据集处理的环境。

### 2.1 数据集格式

数据集的格式如下（.TXT一样）：

```
data/dental/crown
├── train.txt            # 训练集样本列表（存放目录名）
├── test.txt             # 测试集样本列表（存放目录名）
├── 1162858478_Lower/ #下颚
│   ├── Preparation.ply       # 主牙齿点云
│   ├── Antagonist.ply        # 对颌牙齿点云
│   └── Crown.ply             # 牙冠点云
├── 1162858518_Upper/ #上颚
│   ├── Preparation.ply 
│   ├── Antagonist.ply 
│   └── Crown.ply
└── ...
```

- 每个样本存储在一个单独的文件夹中（例如：1162858478_Lower），代码中会根据目录的'Lower'和'Upper'关键字来判断，基牙在什么位置，如果目录中查找不对关键字，默认下颚。

- ply文件格式
  ```
  ply
  format binary_little_endian 1.0
  comment VCGLIB generated
  element vertex 205311
  property double x
  property double y
  property double z
  property double nx
  property double ny
  property double nz
  element face 410618
  property list uchar int vertex_indices
  end_header
  -29.38438416 18.46440506 -9.21412468 0.50048351 -0.86551374 -0.02005653
  -29.34631348 18.48053932 -8.96335983 0.50025159 -0.86563897 -0.02043301
  ```
  
- TXT文件格式
  每行包含6个值（空格或制表符分隔）：
  ```
  x y z nx ny nz
  ```
  - `x, y, z`: 点的3D坐标
  - `nx, ny, nz`: 点的法向量
  示例：

  ```bash
  -29.38438416 18.46440506 -9.21412468 0.50048351 -0.86551374 -0.02005653
  -29.34631348 18.48053932 -8.96335983 0.50025159 -0.86563897 -0.02043301
  ```

### 2.2 配置数据集

## 3. 运行项目

> 在容器中进行。

### 3.1 执行训练

```shell
python main.py --file_key_words Preparation Antagonist Crown
```

### 3.2 执行测试

1. 有真实冠数据（建议）

```
python main.py --test --use_crown --ckpts "experiments/PoinTr/Tooth_models/default/ckpt-best.pth" --file_key_words Preparation Antagonist Crown
```

--use_crown ：表示在测试的时候，回去加载数据集中的*_Crown.ply/txt文件，来回去真实的冠数据，获取到真实冠数据用于确定冠的中心，来计算最大值最小值，用于归一化和反归一化。

2. 没有真实冠数据

```
python main.py --test --ckpts "experiments/PoinTr/Tooth_models/default/ckpt-best.pth" --file_key_words Preparation Antagonist Crown
```

如果没有真实冠数据，则模型不会打印指标，预测出的冠数据保存在./af目录下

### 3.3 使用脚本运行

注释掉`1. 生成psr文件`和`2. 训练前检查`，运行run_main.sh脚本中的`3. 执行训练`和`4. 执行测试`。

run_main.sh内容如下：

```python
#!/bin/bash
# 1. 执行训练
python main.py --file_key_words Preparation Antagonist Crown

# 2. 执行测试
python main.py --test --use_crown --ckpts "experiments/PoinTr/Tooth_models/default/ckpt-best.pth" --file_key_words Preparation Antagonist Crown
```

## 4. 相关代码说明

目录说明：

| 文件                              | 说明                              |
| --------------------------------- | --------------------------------- |
| `datasets/crowndataset.py`        | 数据集加载器（已修改支持TXT/PLY） |
| `main.py`                         | 训练/测试入口                     |
| models/PoinTr.py                  | 定义模型结构                      |
| `cfgs/dataset_configs/Tooth.yaml` | 数据集配置                        |
| `cfgs/Tooth_models/PoinTr.yaml`   | 模型配置                          |


## 5. 常见问题

### Q: PSR文件是什么？必须要有吗？

PSR（Poisson Surface Reconstruction），泊松曲面重建，psr.npy是训练必需的（代码可根据.ply或者.txt点云数据自己生成）。

项目根据真实冠的点和法向量生成psr.npy（形状[128,128,128]）,在训练阶段，与预测出来的psr数据进行对比，计算loss值。

### Q: 如何调整训练/测试集划分？

编辑以下文件：

- `data/dental/crown/train.txt` - 训练集样本ID
- `data/dental/crown/test.txt` - 测试集样本ID

### Q: 可以混合使用PLY和TXT文件吗？

可以！代码会自动识别文件格式。
