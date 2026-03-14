# 🦷 牙冠自动生成项目

[![Ubuntu](https://img.shields.io/badge/Ubuntu-22.04-354F60?logo=ubuntu&logoColor=white)](https://ubuntu.com/)
[![Python](https://img.shields.io/badge/Python-3.8-3776AB?logo=python&logoColor=white)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.7.1-DD0000?logo=pytorch&logoColor=white)](https://pytorch.org/)
[![CUDA](https://img.shields.io/badge/CUDA-11.8-76B900?logo=nvidia&logoColor=white)](https://developer.nvidia.com/cuda)
[![Docker](https://img.shields.io/badge/Docker-20.10-2496ED?logo=docker&logoColor=white)](https://www.docker.com/)
[![Open3D](https://img.shields.io/badge/Open3D-0.19-0077B2?logo=github&logoColor=white)](http://www.open3d.org/)
[![License](https://img.shields.io/badge/License-MIT-00AA00?logo=opensourceinitiative&logoColor=white)](LICENSE)

本项目基于 [DMC](https://github.com/Golriz-code/DMC) 进行改进，支持在 Docker 容器中运行，实现基于 3D 点云的牙冠自动生成。

## ✨ 主要特性

- ✅ 支持 TXT 和 PLY 格式的点云数据（包含法向量）
- ✅ 自动从点云数据生成 PSR（泊松曲面重建）文件（psr.npy）
- ✅ 3 种自适应裁剪中心方法：基于真实冠、基于立方体、基于球体

***

## 📋 目录

- [快速开始](#-快速开始)
- [环境要求](#-环境要求)
- [安装和配置](#-安装和配置)
- [数据集准备](#-数据集准备)
- [使用方法](#-使用方法)
- [配置说明](#-配置说明)
- [输出说明](#-输出说明)
- [常见问题](#-常见问题)

***

## 🚀 快速开始

```bash
# 1. 导入本地镜像（必须，避免配置环境）
sudo docker load -i whc_pytorch271_cuda118_dmc:v4.0.tar
sudo docker images  # 查看导入的镜像

# 2. 启动 Docker 容器
./run_docker_cuda118.sh

# 3. 在容器内执行训练
python main.py

# 4. 在容器内执行测试
python main.py --test --use_crown --ckpts "experiments/PoinTr/Tooth_models/default/ckpt-best.pth"
```

***

## 📦 环境要求

| 安装包        | 版本    | 说明           |
| ------------- | ------- | -------------- |
| **Ubuntu**    | 22.04.5 | <br />         |
| **Docker**    | 29.2.0  | <br />         |
| **Python**    | 3.8     | 本地镜像已预装 |
| **PyTorch**   | 2.7.1   | 本地镜像已预装 |
| **CUDA**      | 11.8    | 本地镜像已预装 |
| **open3d**    | 0.19.0  | 本地镜像已预装 |
| **pytorch3d** | 0.7.9   | 本地镜像已预装 |
| **cmake**     | 4.0.2   | 本地镜像已预装 |
| **chamfer**   | 2.0.0   | 本地镜像已预装 |

> 💡 **作者配置**：RTX 1660 + 16GB 显存 + 32GB 内存

***

***

## 🔧 安装和配置

### 1. 获取文件

需要以下文件：

- `whc_pytorch271_cuda118_dmc:v4.0.tar` # Docker 镜像文件
- `DMC_BOX.tar`                         # 项目文件

### 2. 解压项目

```bash
sudo tar -xvf DMC_BOX.tar
```

解压后的目录结构：

```
DMC_BOX/
├── SAP/
├── cfgs/
├── data/
├── datasets/
├── extensions/
├── main.py
├── models/
├── pointnet2_ops_lib/
├── pytorch3d/
├── run_docker_cuda118.sh   # Docker 容器启动脚本
├── run_main.sh             # 训练/测试脚本
├── tools/
├── utils/
└── README.md
```

### 3. 导入 Docker 镜像

```bash
sudo docker load -i whc_pytorch271_cuda118_dmc.tar
sudo docker images  # 查看导入的镜像
```

> 💡 **为什么使用本地镜像？**\
> 本地镜像已预配置好所有依赖（Python、PyTorch、CUDA、Open3D 等），避免了复杂的环境配置过程，确保环境一致性。

### 4. 创建和启动容器

修改 `run_docker_cuda118.sh` 中的变量：

```bash
CONTAINER_NAME="my_dmc_8g"      # 容器名称（自定义）
IMAGE_ID="ed631bf9e830"         # 镜像 ID（从 docker images 获取）
```

启动容器：

```bash
cd DMC_BOX
./run_docker_cuda118.sh
```

#### 端口映射说明

| 参数                     | 说明                             |
| ---------------------- | ------------------------------ |
| `-p 8080:22`           | 主机 8080 端口 → 容器 22 端口（远程SSH使用） |
| `--shm-size=8g`        | 共享内存 8GB                       |
| `-v $(pwd):/workspace` | 挂载当前目录到容器 `/workspace`         |

***

## 📊 数据集准备

### 数据集格式

```
data/dental/crown/
├── train.txt            # 训练集样本列表（目录名）
├── test.txt             # 测试集样本列表（目录名）
├── 1162858478_Lower/    # 下颚样本
│   ├── Preparation.ply  # 基牙点云
│   ├── Antagonist.ply   # 对颌点云
│   └── Crown.ply        # 牙冠点云（真实值）
├── 1162858518_Upper/    # 上颚样本
│   ├── Preparation.ply
│   ├── Antagonist.ply
│   └── Crown.ply
└── ...
```

#### 文件格式说明

**PLY 格式**：

```
ply
format binary_little_endian 1.0
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
```

**TXT 格式**（每行 6 个值，空格或制表符分隔）：

```
x y z nx ny nz
-29.38438416 18.46440506 -9.21412468 0.50048351 -0.86551374 -0.02005653
```

### 配置数据集路径

修改 `cfgs/dataset_configs/Tooth.yaml`：

```yaml
NAME: crown
DATA_PATH: data/dental/crown      # 存放 train.txt/test.txt
N_POINTS: 10240
PC_PATH: /workspace/data/dental/crown  # Docker 内的数据路径
```

> 💡 **注意**：`PC_PATH` 是 Docker 容器内的路径。如果主机数据在 `E:\data\crown`，需要修改 `run_docker_cuda118.sh` 的挂载参数：
>
> ```bash
> -v E:/data/crown:/workspace/data/dental/crown
> ```

***

## 🎯 使用方法

### 训练模型

```bash
# 基础训练
python main.py

# 自定义文件名关键字
python main.py --file_key_words Preparation Antagonist Crown
```

### 测试模型

#### 方式 1：有真实冠数据（推荐）

```bash
python main.py --test --use_crown \
  --ckpts "experiments/PoinTr/Tooth_models/default/ckpt-best.pth" \
  --file_key_words Preparation Antagonist Crown
```

#### 方式 2：无真实冠数据（需指定冠中心）

```bash
python main.py --test \
  --shell_center [-25, 5, -4] \
  --ckpts "experiments/PoinTr/Tooth_models/default/ckpt-best.pth" \
  --file_key_words Preparation Antagonist Crown
```

> ⚠️ **注意**：如果没有真实冠数据，模型不会计算评估指标。

***

## ⚙️ 配置说明

### 关键参数说明

| 参数                 | 类型          | 默认值                             | 说明             |
| ------------------ | ----------- | ------------------------------- | -------------- |
| `--config`         | str         | `cfgs/Tooth_models/PoinTr.yaml` | 模型配置文件         |
| `--file_key_words` | str (nargs) | `Preparation Antagonist Crown`  | 文件名关键字         |
| `--shell_center`   | int (list)  | `[-25, 5, -4]`                  | 冠中心坐标（无真实冠时使用） |
| `--use_crown`      | flag        | `False`                         | 是否使用真实冠数据      |
| `--ckpts`          | str         | `None`                          | 测试时的模型路径       |
| `--test`           | flag        | `False`                         | 测试模式           |
| `--resume`         | flag        | `False`                         | 恢复训练           |

## 📁 输出说明

### 训练输出

训练会在 `./experiments/` 下生成：

```
experiments/
└── PoinTr/
    └── Tooth_models/
        └── default/
            ├── 20260314_120000.log      # 日志文件
            ├── ckpt-best.pth            # 最佳模型权重
            └── TFBoard/
                ├── train/               # 训练 TensorBoard
                └── test/                # 测试 TensorBoard
```

### 测试输出

测试结果保存在 `./af/` 和`./Results-pointr/`目录下：

```
af/                            #重建的网格模型
Results-pointr/                #预测的牙冠点云
```

***

## 📚 功能代码说明

### 目录结构

| 文件/目录                         | 说明                                |
| --------------------------------- | ----------------------------------- |
| `datasets/crowndataset.py`        | 数据集加载器                        |
| `main.py`                         | 训练/测试入口                       |
| `models/PoinTr.py`                | PoinTr 模型定义                     |
| `cfgs/dataset_configs/Tooth.yaml` | 数据集配置                          |
| `cfgs/Tooth_models/PoinTr.yaml`   | 模型训练配置                        |
| `SAP/`                            | Poisson Surface Reconstruction 实现 |
| `tools/`                          | 训练/测试工具函数                   |
| `utils/`                          | 工具函数                            |

***

## ❓ 常见问题

### Q1: PSR 文件是什么？必须要有吗？

**PSR**（Poisson Surface Reconstruction，泊松曲面重建）是训练必需的。代码会根据真实冠的点云自动生成 PSR 文件（默认形状 `[128,128,128]`）。

- **作用**：作为训练时的监督信号，与预测结果对比计算 loss
- **生成时机**：数据加载时自动创建
- **存储位置**：与点云数据同目录

### Q2: 如何调整训练/测试集划分？

编辑以下文件：

- `data/dental/crown/train.txt` - 训练集样本 ID
- `data/dental/crown/test.txt` - 测试集样本 ID

每行一个目录名，例如：

```
1162858478_Lower
1162858518_Upper
1162858523_Lower
```

### Q3: 可以混合使用 PLY 和 TXT 文件吗？

**可以！** 代码会自动识别文件格式并加载。

***

## 🙏 致谢

- 感谢 [DMC](https://github.com/Golriz-code/DMC) 和[shape_as_points](https://github.com/autonomousvision/shape_as_points)项目提供的基础框架
- 感谢 PyTorch3D 和 PointNet++提供的工具

***

## 📬 联系方式

如有问题，请通过以下方式联系：

- 提交 GitHub Issue
- 发送邮件至：\[<153887677@qq.com>]

***

**⭐⭐⭐**
