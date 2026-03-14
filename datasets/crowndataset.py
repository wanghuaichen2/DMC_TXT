import os
import torch
import numpy as np
import torch.utils.data as data
import random
from .build import DATASETS
import open3d as o3d
import open3d
from os import listdir
import logging
import copy
from models.PoinTr import fps
from utils import parser, dist_utils, misc,config
from pointnet2_ops import pointnet2_utils
from SAP.src.dpsr import DPSR
import sys
from SAP.src.utils import *

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
@DATASETS.register_module()
class crown(data.Dataset):
    def __init__(self, config):
        self.data_root = config.DATA_PATH
        self.pc_path = config.PC_PATH
        self.subset = config.subset
        self.npoints = config.N_POINTS
        self.data_list_file = os.path.join(self.data_root, f'{self.subset}.txt')

        print(f'[DATASET] Open file {self.data_list_file}')
        with open(self.data_list_file, 'r') as f:
            lines = f.readlines()

        self.file_list = []
        for line in lines:
            line = line.strip()
            tax_id = line
            if 'Lower' in tax_id:
                taxonomy_id = '0'
            elif 'Upper' in tax_id:
                taxonomy_id = '1'
            else:
                taxonomy_id = '0'

            self.file_list.append({
                'taxonomy_id': taxonomy_id,
                'model_id': tax_id,
                'file_path': line
            })
        print(f'[DATASET] {len(self.file_list)} instances were loaded')

    def pc_norm(self, pc):
        centroid = np.mean(pc, axis=0)
        std_pc = np.std(pc, axis=0)
        pc = (pc - centroid) / std_pc
        return pc, centroid, std_pc

    def normalize_points_mean_std(self, main_opposing_partial, crown):

        # 计算均值和标准差（指定 dim=0，保持维度用于广播）
        context_mean = torch.mean(main_opposing_partial, dim=0, keepdim=True)
        context_std = torch.std(main_opposing_partial, dim=0, keepdim=True,unbiased=False)

        # 标准化
        new_context_points = (main_opposing_partial - context_mean) / context_std

        # shell 是张量
        new_crown_points = (crown - context_mean) / context_std

        return new_context_points, new_crown_points, context_mean, context_std

    def load_point_cloud(self, file_path):
        """Load point cloud from either .ply or .txt file"""
        if file_path.endswith('.txt'):
            # Load txt file: each line contains x, y, z, nx, ny, nz
            data = np.loadtxt(file_path)
            pcd = o3d.geometry.PointCloud()
            pcd.points = o3d.utility.Vector3dVector(data[:, :3])
            if data.shape[1] >= 6:
                pcd.normals = o3d.utility.Vector3dVector(data[:, 3:6])
            return pcd
        else:
            # Load ply file
            return o3d.io.read_point_cloud(file_path)

    def __getitem__(self, idx):

        # read points
        sample = self.file_list[idx]
        # print(sample['file_path'])
        args = parser.get_args()
        use_crown = args.use_crown
        Preparation_Name = 'Preparation'; Antagonist_Name = 'Antagonist'; Crown_Name = 'Crown'
        if len(args.file_key_words) == 3:
            Preparation_Name = args.file_key_words[0]; Antagonist_Name = args.file_key_words[1]; Crown_Name = args.file_key_words[2]

        # 获取所有数据文件的文件名
        data_file_lists = os.listdir(os.path.join(self.pc_path, sample['file_path']))
        preparation_ply_name = next((item for item in data_file_lists if Preparation_Name in item), None)
        antagonist_ply_name = next((item for item in data_file_lists if Antagonist_Name in item), None)
        crown_ply_name = next((item for item in data_file_lists if Crown_Name in item), None)
        psr_npz_name = next((item for item in data_file_lists if 'psr' in item), None)

        # 检查数据
        """
        训练模式下: 1. 有真实冠和psr.npz数据
                  2. 有真实冠，无psr.npz数据
                  
        测试模式下: 1. 有真实冠，无 psr.npz 数据
                  2. 无真实冠，无 psr.npz 数据
        """
        # 1. 加载牙冠数据
        if not use_crown: # 不使用本地真实牙冠数据
            shellP = np.float32([0.0,0.0,0.0])
            shell_center = args.shell_center  # 改值由外界传入
            print("不使用真实牙冠数据信息, shell_center 参数从args中获取，shell_center：",shell_center)
        else:
            shell = self.load_point_cloud(os.path.join(self.pc_path, sample['file_path'], crown_ply_name))
            shellP = np.asarray(shell.points)
            shell_center = np.mean(shellP, axis=0)

        # 2. 加载上下颚牙齿数据、组合、并裁剪出中心(依据 shell_center)
        main = self.load_point_cloud(os.path.join(self.pc_path, sample['file_path'], preparation_ply_name))
        opposing = self.load_point_cloud(os.path.join(self.pc_path, sample['file_path'], antagonist_ply_name))
        main_opposing = np.concatenate((main.points, opposing.points), axis=0)
        # main_opposing_partial = self.crop_point_cloud(main_opposing, shellP) # 基于真实冠
        # main_opposing_partial = self.crop_point_cloud_cuboid(main_opposing, shell_center) # 真实冠立方体
        main_opposing_partial = self.crop_point_cloud_sphere(main_opposing, shell_center) # 真实冠球体
        # np.savetxt('main_opposing_partial.txt', main_opposing_partial, delimiter=' ', fmt='%.6f')
        # 使用上下牙的最大值和最小值
        min_gt = np.min(main_opposing_partial).astype(np.float32)
        max_gt = np.max(main_opposing_partial).astype(np.float32)

        # 3. 获取 psr.npz 数据, 如果没有就自动生成
        shell_grid = np.float32([])
        # 当 没有 psr.npz 文件，有 真实冠 文件，确定要使用 真实冠数据
        if psr_npz_name is None and crown_ply_name is not None and use_crown:
            resolution = 128
            dpsr = DPSR(res=(resolution, resolution, resolution), sig = 2)
            points = np.asarray(shell.points, dtype=np.float32)
            normals = np.asarray(shell.normals, dtype=np.float32)
            # mesh = o3d.io.read_triangle_mesh(os.path.join(self.pc_path, sample['file_path'], crown_ply_name))
            # points = np.asarray(mesh.vertices, dtype=np.float32)
            # normals = np.asarray(mesh.compute_vertex_normals().vertex_normals, dtype=np.float32)
            points = (points - min_gt) / (max_gt - min_gt + 1.0)
            points = np.clip(points, 0, 1) # 裁剪到有效范围, 避免box定义太小造成错误
            points_t = torch.from_numpy(points).unsqueeze(0)  # [1, N, 3]
            normals_t = torch.from_numpy(normals).unsqueeze(0)  # [1, N, 3]
            shell_grid = dpsr(points_t, normals_t).squeeze().cpu().numpy().astype(np.float32)

            # 保存为 npz 文件
            out_path = os.path.join(self.pc_path, sample['file_path'], 'psr.npz')
            np.savez(out_path, psr=shell_grid)
            print(f"已保存: {out_path}")

        # 当 有 psr.npz 文件，有 真实冠 文件，确定要使用 真实冠数据
        if psr_npz_name is not None and use_crown:
            shell_grid = np.load(os.path.join(self.pc_path, sample['file_path'], 'psr.npz'))
            psr = shell_grid['psr']
            shell_grid = psr.astype(np.float32)

        # #打印出生成psr查看效果
        # shell_grid_temp = torch.from_numpy(np.asarray(shell_grid)).float().view(1, resolution, resolution, resolution)
        # v, f, _ = mc_from_psr(shell_grid_temp, zero_level= 0)
        # # denormalize 反归一化
        # de_p = (v * (max_gt + 1.0 - min_gt) + min_gt)
        # mesh_dir = './build_psr_out'
        # os.makedirs(mesh_dir, exist_ok=True)
        # mesh_out_file = os.path.join(mesh_dir, str(sample['file_path']) + '_chack_psr-128-1-0.ply')
        # export_mesh(mesh_out_file, de_p, f)

        # 2. 进行下采样
        npoints = self.npoints

        if main_opposing_partial.shape[0] > npoints:
            main_only_tensor = torch.from_numpy(main_opposing_partial).float().unsqueeze(0)  #更改形状，转移到GPU上算
            main_opposing_partial = fps(main_only_tensor, self.npoints, device).squeeze(0).cpu() #变回原来的形状，回到cpu上
        else:
            main_opposing_partial = torch.from_numpy(main_opposing_partial).float()
            print("上下文数据异常：",sample['file_path'],"> 裁剪中心后点数量", main_opposing_partial.shape[0])
        if shellP.shape[0] > npoints:
            shellP_tensor = torch.from_numpy(shellP).float().unsqueeze(0)
            shellP = fps(shellP_tensor, self.npoints, device).squeeze(0).cpu()
        else:
            shellP = torch.from_numpy(shellP).float()
            print("牙冠数据异常数据：", sample['file_path'], "> 裁剪中心后点数量", main_opposing_partial.shape[0])

        # 3.归一化处理
        try:
            main_opposing_partial_only = main_opposing_partial.detach().clone()
            shell_only = shellP.detach().clone()
        except:
            print("Error:数据归一化异常（datasets/crowndataset.py,异常数据：",sample['file_path'])
        context_partial_only, shell_only, centroid, std_pc = self.normalize_points_mean_std(main_opposing_partial_only, shell_only)

        data_partial = context_partial_only.float()
        data_gt = shell_only.float()
        value_centroid = centroid.float()
        value_std_pc = std_pc.float()
        shell_grid_gt = torch.from_numpy(np.asarray(shell_grid)).float()

        return sample['taxonomy_id'], sample['model_id'], data_gt, data_partial, value_centroid, value_std_pc, shell_grid_gt, min_gt, max_gt

    def __len__(self):
        return len(self.file_list)

    # 基于冠中心裁剪出基牙立方体，基于中心裁剪固定长宽高的数据box
    def crop_point_cloud_cuboid(self, pc, shell_center):
        x_range = 10
        y_range = 10
        z_range = 8

        """根据指定中心裁剪点云"""
        pc_xyz = pc[:, :3]
        x_min, x_max, y_min, y_max, z_min, z_max = self.cla_max_min_bycenter(shell_center,x_range,y_range,z_range)
        # x_min, x_max = shell_center[0] - x_range, shell_center[0] + x_range
        # y_min, y_max = shell_center[1] - y_range, shell_center[1] + y_range
        # z_min, z_max = shell_center[2] - z_range, shell_center[2] + z_range
        mask = (
                (pc_xyz[:, 0] >= x_min) & (pc_xyz[:, 0] <= x_max) &
                (pc_xyz[:, 1] >= y_min) & (pc_xyz[:, 1] <= y_max) &
                (pc_xyz[:, 2] >= z_min) & (pc_xyz[:, 2] <= z_max)
        )
        cropped_pc = pc[mask]
        if cropped_pc.shape[0] == 0:
            raise Warning("裁剪后点云为空！建议调整裁剪范围或中心位置")
        return cropped_pc

    def cla_max_min_bycenter(self,shell_center,x_range,y_range,z_range):
        return (shell_center[0] - x_range, shell_center[0] + x_range,
                shell_center[1] - y_range, shell_center[1] + y_range,
                shell_center[2] - z_range, shell_center[2] + z_range)

    # 基于牙冠中心，裁剪出半径为10的球
    def crop_point_cloud_sphere(self, pc, shell_center):
        radius = 10
        """根据指定中心裁剪出半径为radius的球体内的点云"""
        pc_xyz = pc[:, :3]
        # 计算每个点到中心的距离平方（避免开方，提高效率）
        dist_sq = ((pc_xyz[:, 0] - shell_center[0]) ** 2 +
                   (pc_xyz[:, 1] - shell_center[1]) ** 2 +
                   (pc_xyz[:, 2] - shell_center[2]) ** 2)
        mask = dist_sq <= radius ** 2
        cropped_pc = pc[mask]
        if cropped_pc.shape[0] == 0:
            raise Warning("裁剪后点云为空！建议调整裁剪半径或中心位置")
        return cropped_pc

    # 基于牙冠数据来裁剪出基牙主要区域，基于真实冠自适应大小，前提有真实冠数据
    def crop_point_cloud(self, pc, shell_points=None):
        """
        基于牙冠中心点和尺寸裁剪点云
        shell_points: 牙冠点云，用于计算中心和裁剪范围
        """
        if shell_points is None or len(shell_points) == 0:
            raise ValueError("需要提供牙冠点云来计算裁剪范围")

        # 计算牙冠中心
        shell_center = shell_points.mean(axis=0)

        # 基于牙冠尺寸计算裁剪范围
        shell_size = shell_points.max(axis=0) - shell_points.min(axis=0)
        expand_ratio = getattr(self, 'crop_expand_ratio', 0.6)
        x_range = shell_size[0] * expand_ratio
        y_range = shell_size[1] * expand_ratio
        z_range = shell_size[2] * (expand_ratio)

        # 裁剪点云
        pc_xyz = pc[:, :3]
        x_min, x_max = shell_center[0] - x_range, shell_center[0] + x_range
        y_min, y_max = shell_center[1] - y_range, shell_center[1] + y_range
        z_min, z_max = shell_center[2] - z_range, shell_center[2] + z_range
        mask = (
                (pc_xyz[:, 0] >= x_min) & (pc_xyz[:, 0] <= x_max) &
                (pc_xyz[:, 1] >= y_min) & (pc_xyz[:, 1] <= y_max) &
                (pc_xyz[:, 2] >= z_min) & (pc_xyz[:, 2] <= z_max)
        )
        cropped_pc = pc[mask]
        if cropped_pc.shape[0] == 0:
            raise Warning("裁剪后点云为空！建议调整裁剪范围")
        return cropped_pc