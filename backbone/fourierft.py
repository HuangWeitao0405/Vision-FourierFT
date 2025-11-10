# Sheng Wang at Feb 22 2023
# Modified: Replaced LoRA with FourierFT - 将LoRA适配器替换为FourierFT适配器
# Updated: Implement complete FourierFT functionality with task-specific branches and weight merging

import math
import timm
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
from timm.models.vision_transformer import VisionTransformer as timm_ViT
from torch import Tensor
from torch.nn.parameter import Parameter
from backbone.base_vit import ViT
import os
from backbone.linears import SimpleLinear
import gc
import torch.nn.utils as utils
import copy
import json
from typing import Dict, List, Optional, Set, Tuple
import warnings

# 类级别索引存储，确保所有实例共享相同的索引管理
class GlobalIndicesStorage:
    _global_indices = {}
    _local_indices = {}

class FourierFTLayer(nn.Module):
    """
    完整的FourierFT层实现，对应材料1中的FourierFTLayer
    实现任务特定分支、索引不重叠和权重合并策略
    """
    def __init__(self, base_layer: nn.Module, n_frequency: int = 1000, 
                 n_frequency_non_trainable: int = 0, scaling: float = 150.0,
                 random_loc_seed: int = 777, init_weights: bool = False,
                 prefer: bool = False, init_fc: int = 0, task_id: int = 0,
                 round_number: int = 0, reinit_num: int = 0,
                 indices_file_path: Optional[str] = None,
                 adapter_name: str = "default"):
        super().__init__()
        self.base_layer = base_layer.requires_grad_(False)
        self.adapter_name = adapter_name
        self.merged = False  # 添加merged参数控制是否输出增量权重
        
        # 频率参数管理
        self.n_frequency = n_frequency
        self.n_frequency_non_trainable = n_frequency_non_trainable
        self.scaling = scaling
        self.random_loc_seed = random_loc_seed
        self.init_weights = init_weights
        self.prefer = prefer
        self.init_fc = init_fc
        self.task_id = task_id
        self.round_number = round_number
        self.reinit_num = reinit_num
        self.indices_file_path = indices_file_path
        
        # 获取基础层维度
        if isinstance(base_layer, nn.Linear):
            self.in_features, self.out_features = base_layer.in_features, base_layer.out_features
        else:
            raise ValueError(f"Unsupported layer type {type(base_layer)}")
        
        # 频谱参数
        self.fourierft_spectrum = nn.ParameterDict({})
        self.fourierft_spectrum_non_trainable = nn.ParameterDict({})
        
        # 索引管理（实现任务特定分支和避免重叠）
        self.indices = {}
        self.non_trainable_indices = {}
        self.used_indices_history = set()  # 记录历史使用的索引
        
        # 初始化适配器
        self.update_layer(adapter_name)
            
    def _generate_task_specific_indices(self, adapter_name: str):
        """
        生成任务特定的频率索引，确保不与其他任务重叠
        对应材料1中的任务特定分支实现
        """
        total_positions = self.out_features * self.in_features
        
        # 全局索引管理 - 使用类级别存储，确保所有实例共享相同的索引
        global_indices_key = "global"
        
        # 收集所有可能的其他任务局部索引
        other_rounds_indices = []
        used_indices = set()
        
        if self.round_number == -1:
            # 基础轮次：只使用全局频率
            # 生成新的全局随机排列
            generator = torch.Generator().manual_seed(self.random_loc_seed)
            permutation = torch.randperm(total_positions, generator=generator)
            training_indices = permutation[:self.n_frequency]
            GlobalIndicesStorage._global_indices[global_indices_key] = training_indices
            
            # 即使是基础轮次，也尝试收集已存在的其他任务局部索引
            for r in range(max(1, self.task_id + 1)):
                if r != self.task_id:
                    local_indices_key = f"{adapter_name}_local_{r}"
                    if local_indices_key in GlobalIndicesStorage._local_indices:
                        round_indices = GlobalIndicesStorage._local_indices[local_indices_key]
                        other_rounds_indices.append(round_indices)
                        used_indices.update(round_indices.tolist())
            
            non_training_indices = torch.cat(other_rounds_indices) if other_rounds_indices else torch.tensor([], dtype=torch.long)
        else:
            # 多轮情况：全局频率 + 当前轮次的局部频率
            
            # 1. 生成全局索引（如果不存在）
            # 从全局存储中查找，或者生成新的
            if global_indices_key in GlobalIndicesStorage._global_indices:
                # 从存储中获取已有的全局索引
                global_indices = GlobalIndicesStorage._global_indices[global_indices_key][:self.n_frequency]
            else:
                # 生成新的全局索引并存储到全局存储
                generator = torch.Generator().manual_seed(self.random_loc_seed)
                permutation = torch.randperm(total_positions, generator=generator)
                global_indices = permutation[:self.n_frequency]
                
                # 存储到全局索引存储
                GlobalIndicesStorage._global_indices[global_indices_key] = global_indices
            
            # 2. 收集其他轮次的局部索引（从全局存储中）
            used_indices = set(global_indices.tolist())
            
            # 扩展查找范围，查找所有与当前adapter_name相关的局部索引    
            # 尝试所有可能的局部索引键格式
            collected_any = False
            for key in GlobalIndicesStorage._local_indices.keys():
                # 匹配任何包含'local_'的键，不仅仅是那些以当前adapter_name开头的
                if 'local_' in key:
                    try:
                        # 尝试从键中提取任务ID（最后一个下划线后面的部分）
                        task_part = key.split('_')[-1]
                        if task_part.isdigit():
                            r = int(task_part)
                            # 只收集不是当前任务的索引
                            if r != self.task_id:
                                round_indices = GlobalIndicesStorage._local_indices[key]
                                other_rounds_indices.append(round_indices)
                                used_indices.update(round_indices.tolist())
                                collected_any = True
                    except (ValueError, IndexError) as e:
                        continue
            
            # 3. 为当前轮次生成新的局部索引，确保不与已使用的索引重叠
            current_round_indices_key = f"{adapter_name}_local_{self.task_id}"
            
            # 检查当前轮次是否已有索引
            if current_round_indices_key in GlobalIndicesStorage._local_indices:
                current_round_indices = GlobalIndicesStorage._local_indices[current_round_indices_key]
            else:
                generator = torch.Generator().manual_seed(self.random_loc_seed + self.task_id)
                available_positions = list(set(range(total_positions)) - used_indices)
                
                # 如果可用位置不足，使用随机采样并处理冲突
                if len(available_positions) < self.reinit_num:
                    # 先使用所有可用位置
                    current_round_indices = torch.tensor(available_positions)
                    # 再随机采样剩余需要的位置（允许重复）
                    remaining_needed = self.reinit_num - len(available_positions)
                    if remaining_needed > 0:
                        random_indices = torch.randint(0, total_positions, (remaining_needed,), generator=generator)
                        current_round_indices = torch.cat([current_round_indices, random_indices])
                else:
                    # 从可用位置中随机选择
                    permuted_available = torch.randperm(len(available_positions), generator=generator)
                    current_round_indices = torch.tensor(available_positions)[permuted_available[:self.reinit_num]]
                
                # 存储当前轮次的局部索引到全局存储
                GlobalIndicesStorage._local_indices[current_round_indices_key] = current_round_indices
            
            # 组合训练索引
            training_indices = torch.cat([global_indices, current_round_indices])
            
            # 组合非训练索引（其他轮次的局部索引）
        if other_rounds_indices:
            non_training_indices = torch.cat(other_rounds_indices)
        else:
            non_training_indices = torch.tensor([], dtype=torch.long)
        
        # 转换为二维坐标
        training_indices_2d = torch.stack([
            training_indices // self.in_features, 
            training_indices % self.in_features
        ], dim=0)
        
        non_training_indices_2d = torch.stack([
            non_training_indices // self.in_features, 
            non_training_indices % self.in_features
        ], dim=0) if len(non_training_indices) > 0 else torch.tensor([], dtype=torch.long)
        
        return training_indices_2d, non_training_indices_2d

    def update_layer(self, adapter_name: str):
        """更新或创建适配器层"""
        # 生成任务特定的索引
        training_indices, non_training_indices = self._generate_task_specific_indices(adapter_name)
        # 存储当前任务的可训练索引（全局索引+当前任务局部索引）
        self.indices[adapter_name] = training_indices
        
        # 存储不可训练索引（其他任务的局部索引）
        self.non_trainable_indices[adapter_name] = non_training_indices
        
        # 计算不可训练频率数量
        n_non_trainable = len(non_training_indices[0]) if len(non_training_indices) > 0 else 0
        
        # 初始化可训练频谱参数（只包含全局索引和当前任务局部索引）
        # n_trainable = len(training_indices[0])
        # if not self.prefer:
        #     self.fourierft_spectrum[adapter_name] = nn.Parameter(
        #         torch.randn(n_trainable), requires_grad=True
        #     )
        # else:
        #     self.fourierft_spectrum[adapter_name] = nn.Parameter(
        #         torch.zeros(n_trainable), requires_grad=True
        #     )

        # 计算可训练频率数量（全局索引 + 当前任务局部索引）
        n_trainable = len(training_indices[0])
        
        # 获取全局索引的数量（前n_frequency个位置）
        n_global = min(self.n_frequency, n_trainable)
        
        # 初始化频谱参数
        if not self.prefer:
            new_spectrum = torch.randn(n_trainable)
        else:
            new_spectrum = torch.zeros(n_trainable)
        
        # 查找上一个任务的适配器来继承全局索引的频谱参数
        prev_adapter_found = False
        if self.task_id > 0:
            # 尝试查找上一个任务的适配器名称
            prev_adapter_candidates = []
            for existing_adapter in list(self.fourierft_spectrum.keys()):
                # 检查是否是上一个任务的适配器（通过任务ID推断）
                if existing_adapter != adapter_name:
                    # 假设适配器名称包含任务信息，如"task_0", "task_1"等
                    if f"task_{self.task_id-1}" in existing_adapter or f"_{self.task_id-1}" in existing_adapter:
                        prev_adapter_candidates.append(existing_adapter)
                    # 或者通过其他方式识别上一个任务
                    elif hasattr(self, 'prev_task_adapter_name') and existing_adapter == self.prev_task_adapter_name:
                        prev_adapter_candidates.append(existing_adapter)
            
            # 如果有候选适配器，使用第一个找到的
            if prev_adapter_candidates:
                prev_adapter_name = prev_adapter_candidates[0]
                if prev_adapter_name in self.fourierft_spectrum:
                    prev_spectrum = self.fourierft_spectrum[prev_adapter_name].data
                    prev_indices = self.indices.get(prev_adapter_name, None)
                    
                    if prev_indices is not None and len(prev_spectrum) >= n_global:
                        # 获取上一个适配器的全局索引部分
                        prev_global_indices = prev_indices[:, :n_global]  # 前n_global个是全局索引
                        
                        # 获取当前适配器的全局索引部分
                        current_global_indices = training_indices[:, :n_global]
                        
                        # 将索引转换为一维坐标以进行匹配
                        def indices_to_1d(indices_2d):
                            return indices_2d[0] * self.in_features + indices_2d[1]
                        
                        prev_global_1d = indices_to_1d(prev_global_indices)
                        current_global_1d = indices_to_1d(current_global_indices)
                        
                        # 创建映射：当前全局索引位置 -> 对应的频谱值
                        spectrum_map = {}
                        for i, idx_1d in enumerate(prev_global_1d):
                            spectrum_map[idx_1d.item()] = prev_spectrum[i].item()
                        
                        # 将对应的频谱值复制到新适配器
                        for i, idx_1d in enumerate(current_global_1d):
                            idx_val = idx_1d.item()
                            if idx_val in spectrum_map:
                                new_spectrum[i] = spectrum_map[idx_val]
                        
                        prev_adapter_found = True
        
        # 如果没有找到上一个适配器，使用随机初始化但记录信息
        if not prev_adapter_found and self.task_id > 0:
            print(f"警告：未找到任务 {self.task_id-1} 的适配器，全局索引将使用随机初始化")
        
        # 创建参数
        self.fourierft_spectrum[adapter_name] = nn.Parameter(
            new_spectrum, requires_grad=True
        )
        
        # 初始化不可训练频谱参数（保存其他任务的局部索引参数）
        if n_non_trainable > 0:
            # 创建不可训练频谱参数张量
            non_trainable_spectrum = torch.zeros(n_non_trainable)
            
            if non_training_indices.numel() > 0:
                # 将非训练索引转换为一维坐标
                non_training_indices_1d = non_training_indices[0] * self.in_features + non_training_indices[1]
                
                # 从所有现有适配器中查找匹配的频谱参数
                all_adapters = list(self.fourierft_spectrum.keys()) + list(self.fourierft_spectrum_non_trainable.keys())
                
                for existing_adapter in all_adapters:
                    if existing_adapter == adapter_name:
                        continue
                        
                    # 检查可训练频谱
                    if existing_adapter in self.fourierft_spectrum and existing_adapter in self.indices:
                        existing_indices = self.indices[existing_adapter]
                        existing_spectrum = self.fourierft_spectrum[existing_adapter].data
                        
                        existing_indices_1d = existing_indices[0] * self.in_features + existing_indices[1]
                        
                        # 查找匹配的索引
                        for i, idx_1d in enumerate(non_training_indices_1d):
                            matching_pos = (existing_indices_1d == idx_1d).nonzero()
                            if len(matching_pos) > 0:
                                non_trainable_spectrum[i] = existing_spectrum[matching_pos[0, 0]]
                    
                    # 检查不可训练频谱（重要修复）
                    elif existing_adapter in self.fourierft_spectrum_non_trainable and existing_adapter in self.non_trainable_indices:
                        existing_non_train_indices = self.non_trainable_indices[existing_adapter]
                        existing_non_train_spectrum = self.fourierft_spectrum_non_trainable[existing_adapter].data
                        
                        existing_non_train_indices_1d = existing_non_train_indices[0] * self.in_features + existing_non_train_indices[1]
                        
                        # 查找匹配的索引
                        for i, idx_1d in enumerate(non_training_indices_1d):
                            matching_pos = (existing_non_train_indices_1d == idx_1d).nonzero()
                            if len(matching_pos) > 0:
                                non_trainable_spectrum[i] = existing_non_train_spectrum[matching_pos[0, 0]]

            # 创建参数
            self.fourierft_spectrum_non_trainable[adapter_name] = nn.Parameter(
                non_trainable_spectrum, requires_grad=False
            )
        
        # 将旧任务的频率参数设置为不可训练
        for existing_adapter in list(self.fourierft_spectrum.keys()):
            # 跳过当前任务的适配器
            if existing_adapter == adapter_name:
                continue
            
            # 获取旧任务的参数并设置为不可训练
            param = self.fourierft_spectrum[existing_adapter]
            if param.requires_grad:
                # 创建一个新的参数张量，保留值但设置为不可训练
                new_param = nn.Parameter(param.data.detach().clone(), requires_grad=False)
                self.fourierft_spectrum[existing_adapter] = new_param
        
        # 记录使用的索引到历史记录
        self._record_used_indices(adapter_name)
        
        # 移动到相同设备
        self._move_adapter_to_device_of_base_layer(adapter_name)

    def _record_used_indices(self, adapter_name: str):
        """记录使用的索引到历史记录，用于避免重叠"""
        indices = self.indices[adapter_name]
        for i in range(indices.shape[1]):
            pos = (indices[0, i].item(), indices[1, i].item())
            self.used_indices_history.add(pos)

    def _move_adapter_to_device_of_base_layer(self, adapter_name: str):
        """将适配器参数移动到基础层所在的设备"""
        device = self.base_layer.weight.device
        if adapter_name in self.fourierft_spectrum:
            self.fourierft_spectrum[adapter_name].data = self.fourierft_spectrum[adapter_name].data.to(device)
        if adapter_name in self.fourierft_spectrum_non_trainable:
            self.fourierft_spectrum_non_trainable[adapter_name].data = self.fourierft_spectrum_non_trainable[adapter_name].data.to(device)

    def get_delta_weight(self, adapter_name: str) -> torch.Tensor:
        """计算增量权重"""
        # 确保适配器名称存在
        if adapter_name not in self.fourierft_spectrum:
            raise KeyError(f"Adapter '{adapter_name}' not found")
            
        spectrum = self.fourierft_spectrum[adapter_name]
        indices = self.indices[adapter_name].to(spectrum.device)
        
        # 创建稠密频谱矩阵
        dense_spectrum = torch.zeros(
            self.out_features, self.in_features, 
            device=spectrum.device, dtype=spectrum.dtype
        )
        
        # 填充可训练频谱（全局索引和当前任务局部索引）
        dense_spectrum[indices[0, :], indices[1, :]] = spectrum
        
        # 填充不可训练频谱（其他任务的局部索引）
        if adapter_name in self.non_trainable_indices and adapter_name in self.fourierft_spectrum_non_trainable:
            non_trainable_indices = self.non_trainable_indices[adapter_name].to(spectrum.device)
            dense_spectrum[non_trainable_indices[0, :], non_trainable_indices[1, :]] = \
                self.fourierft_spectrum_non_trainable[adapter_name]
        
        # 逆傅里叶变换
        dense_spectrum = dense_spectrum.to(torch.float32).to(torch.complex64)
        delta_weight = torch.fft.ifft2(dense_spectrum).real * self.scaling
        
        return delta_weight

    def merge_adapters(self, adapter_names: List[str], strategy: str = "max") -> torch.Tensor:
        """
        合并多个适配器的权重，对应材料1中的权重合并策略
        """
        if not adapter_names:
            return torch.zeros_like(self.base_layer.weight)
        
        # 计算每个适配器的增量权重
        delta_weights = []
        for adapter_name in adapter_names:
            if adapter_name in self.fourierft_spectrum:
                delta_weights.append(self.get_delta_weight(adapter_name))
        
        if not delta_weights:
            return torch.zeros_like(self.base_layer.weight)
        
        # 合并策略
        if strategy == "mean":
            # 平均策略
            merged_delta = torch.stack(delta_weights).mean(dim=0)
        elif strategy == "max":
            # 最大绝对值策略
            merged_delta = delta_weights[0]
            for delta in delta_weights[1:]:
                merged_delta = torch.where(
                    delta.abs() >= merged_delta.abs(),
                    delta,
                    merged_delta
                )
        else:
            raise ValueError(f"Unsupported merge strategy: {strategy}")
        
        return merged_delta

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """前向传播 - 使用全部适配器"""
        base_output = self.base_layer(x)
        
        # 如果merged为True，直接返回基础层输出（增量已合并到基础层中）
        if self.merged:
            return torch.zeros_like(base_output, device=x.device, dtype=x.dtype)
        
        # 否则，应用增量权重
        delta_output = torch.zeros_like(base_output, device=x.device, dtype=x.dtype)
        # 只使用当前适配器
        if self.adapter_name in self.fourierft_spectrum:
            delta_w = self.get_delta_weight(self.adapter_name)
            x_transformed = x.to(delta_w.dtype)
            delta_output += F.linear(x_transformed, delta_w)
        
        return delta_output

class FourierFTLayerWrapper(FourierFTLayer):
    """
    FourierFT层包装器，保持向后兼容性
    """
    def __init__(self, base_layer, adapter_name, n_frequency, n_frequency_non_trainable, scaling,
                 random_loc_seed, init_weights, prefer, init_fc, task_id=0, round_number=1, reinit_num=0,
                 indices_file_path=None):
        super().__init__(
            base_layer=base_layer,
            n_frequency=n_frequency,
            n_frequency_non_trainable=n_frequency_non_trainable,
            scaling=scaling,
            random_loc_seed=random_loc_seed,
            init_weights=init_weights,
            prefer=prefer,
            init_fc=init_fc,
            task_id=task_id,
            round_number=round_number,
            reinit_num=reinit_num,
            indices_file_path=indices_file_path,
            adapter_name=adapter_name
        )

class FourierFT_ViT(nn.Module):
    """
    应用FourierFT到Vision Transformer模型
    """
    def __init__(self, vit_model: ViT, n_frequency: int = 1000, n_frequency_non_trainable: int = 0,
                 scaling: float = 150.0, random_loc_seed: int = 777, init_weights: bool = False,
                 prefer: bool = False, init_fc: int = 0, num_classes: int = 0, fourier_layer=None,
                 task_id: int = 0, round_number: int = 1, reinit_num: int = 0, indices_file_path=None):
        super(FourierFT_ViT, self).__init__()

        base_vit_dim = vit_model.transformer.blocks[0].attn.proj_q.in_features
        dim = base_vit_dim
        
        # 设置傅里叶层
        if fourier_layer:
            self.fourier_layer = fourier_layer
        else:
            self.fourier_layer = list(range(len(vit_model.transformer.blocks)))
        
        # 冻结基础模型
        for param in vit_model.parameters():
            param.requires_grad = False

        # 应用FourierFT适配器
        self.fourier_wrappers = []
        for t_layer_i, blk in enumerate(vit_model.transformer.blocks):
            if t_layer_i not in self.fourier_layer:
                continue
                
            # 替换查询和值投影层
            w_q_linear = blk.attn.proj_q
            w_v_linear = blk.attn.proj_v
            
            fourier_q = FourierFTLayerWrapper(
                w_q_linear, f"fourier_q_{t_layer_i}", n_frequency, n_frequency_non_trainable,
                scaling, random_loc_seed, init_weights, prefer, init_fc,
                task_id, round_number, reinit_num, indices_file_path
            )
            fourier_v = FourierFTLayerWrapper(
                w_v_linear, f"fourier_v_{t_layer_i}", n_frequency, n_frequency_non_trainable,
                scaling, random_loc_seed, init_weights, prefer, init_fc,
                task_id, round_number, reinit_num, indices_file_path
            )
            
            self.fourier_wrappers.extend([fourier_q, fourier_v])
            blk.attn.proj_q = fourier_q
            blk.attn.proj_v = fourier_v

        self.fourier_vit = vit_model
        if num_classes > 0:
            self.fourier_vit.fc = nn.Linear(vit_model.fc.in_features, num_classes)

    def forward(self, x: Tensor) -> Tensor:
        return self.fourier_vit(x)

    def merge_adapters(self, adapter_names: List[str], strategy: str = "max"):
        """合并所有适配器权重"""
        for wrapper in self.fourier_wrappers:
            # 获取基础层权重
            base_weight = wrapper.base_layer.weight.data
            # 计算合并的增量权重
            merged_delta = wrapper.merge_adapters(adapter_names, strategy)
            # 应用合并的权重
            wrapper.base_layer.weight.data = base_weight + merged_delta

class _FourierFT_qkv_timm(nn.Module):
    """FourierFT包装器，用于timm ViT的qkv层"""
    def __init__(self, qkv, adapter_name, n_frequency, n_frequency_non_trainable, scaling,
                 random_loc_seed, init_weights, prefer, init_fc, task_id=0, round_number=0, reinit_num=0,
                 indices_file_path=None):
        super().__init__()
        self.qkv = qkv
        self.adapter_name = adapter_name
        self.dim = qkv.in_features
        
        # 为Q和V创建FourierFT适配器
        self.fourier_q = FourierFTLayerWrapper(
            nn.Linear(self.dim, self.dim, bias=False), f"{adapter_name}", 
            n_frequency, n_frequency_non_trainable, scaling, random_loc_seed, 
            init_weights, prefer, init_fc, task_id, round_number, reinit_num, indices_file_path
        )
        self.fourier_v = FourierFTLayerWrapper(
            nn.Linear(self.dim, self.dim, bias=False), f"{adapter_name}", 
            n_frequency, n_frequency_non_trainable, scaling, random_loc_seed, 
            init_weights, prefer, init_fc, task_id, round_number, reinit_num, indices_file_path
        )

    def forward(self, x):
        qkv = self.qkv(x)
        new_q = self.fourier_q(x)
        new_v = self.fourier_v(x)
        
        # 将FourierFT调整添加到Q和V组件
        qkv[:, :, : self.dim] += new_q
        qkv[:, :, -self.dim:] += new_v
        
        return qkv

class FourierFT_ViT_timm(nn.Module):
    """基于timm库的Vision Transformer的FourierFT实现"""
    def __init__(self, vit_model: timm_ViT, n_frequency: int = 1000, n_frequency_non_trainable: int = 0,
                 scaling: float = 150.0, random_loc_seed: int = 777, init_weights: bool = False,
                 prefer: bool = False, init_fc: int = 0, num_classes: int = 0, increment=10, 
                 filepath='./', fourier_layer=None, eval=False, cur_task_index=0,
                 round_number: int = 0, reinit_num: int = 0, indices_file_path=None):
        super(FourierFT_ViT_timm, self).__init__()

        assert n_frequency > 0
        self.n_frequency = n_frequency
        self.base_vit = copy.deepcopy(vit_model)

        if not eval:
            self.save_file = indices_file_path
            self.increment = increment
            print('save_file', self.save_file)

        if fourier_layer:
            self.fourier_layer = fourier_layer
        else:
            self.fourier_layer = list(range(len(vit_model.blocks)))
        
        if cur_task_index is not None:
            self.task_id = cur_task_index

        self.round_number = round_number

        # 冻结基础模型
        for param in self.base_vit.parameters():
            param.requires_grad = False
        for param in vit_model.parameters():
            param.requires_grad = False

        # 应用FourierFT适配器
        for t_layer_i, blk in enumerate(vit_model.blocks):
            if t_layer_i not in self.fourier_layer:
                continue
                
            w_qkv_linear = blk.attn.qkv
            blk.attn.qkv = _FourierFT_qkv_timm(
                w_qkv_linear, f"task_{self.task_id}", n_frequency, n_frequency_non_trainable,
                scaling, random_loc_seed, init_weights, prefer, init_fc,
                self.task_id, self.round_number, reinit_num, indices_file_path
            )

        self.fourier_vit = vit_model
        if not eval:
            self.fourier_vit.head = torch.nn.Identity()
        else:
            self.reset_fourier_vit_head()

    def reset_fourier_vit_head(self):
        """重置分类头"""
        task_incremental = self.increment
        self.fourier_vit.head = self.generate_fc(768, (self.task_id) * task_incremental).cuda()
        
        if self.task_id > 0:
            temp_weights = torch.load(self.save_file + 'CLs_weight' + str(self.task_id - 1) + '.pt') 
            temp_bias = torch.load(self.save_file + 'CLs_bias' + str(self.task_id - 1) + '.pt') 
            self.fourier_vit.head.weight.data = temp_weights.data.cuda()
            self.fourier_vit.head.bias.data = temp_bias.data.cuda()

    def generate_fc(self, in_dim, out_dim):
        return SimpleLinear(in_dim, out_dim)

    def forward(self, x: Tensor, loss=False, eval=False) -> Tensor:
        if eval:
            return self.fourier_vit(x)
        elif loss:
            return self.fourier_vit(x), torch.tensor(0.0).to(x.device)
        else:
            return self.fourier_vit(x)

    def merge_adapters(self, adapter_names: List[str], strategy: str = "max"):
        """合并适配器权重到实际的qkv层权重中"""
        # 遍历所有模块，寻找_FourierFT_qkv_timm实例
        for name, module in self.fourier_vit.named_modules():
            if isinstance(module, _FourierFT_qkv_timm):
                # 计算q适配器的增量权重
                q_delta = module.fourier_q.merge_adapters(adapter_names, strategy)
                # 计算v适配器的增量权重
                v_delta = module.fourier_v.merge_adapters(adapter_names, strategy)
                
                # 从_FourierFT_qkv_timm实例获取dim属性
                dim = module.dim
                
                # 将增量权重应用到实际的qkv层权重中
                # q部分对应qkv.weight的前dim列
                # 修改后
                module.qkv.weight.data[:dim, :dim] += q_delta.to(module.qkv.weight.device)
                module.qkv.weight.data[-dim:, -dim:] += v_delta.to(module.qkv.weight.device)
                
                # 设置merged为True，表示增量已合并到基础层
                module.fourier_q.merged = True
                module.fourier_v.merged = True
                    
    def reset_weights_to_base(self):
        """重置所有层的权重到基础状态，移除之前融合的增量权重"""
        reset_count = 0
        
        # 遍历所有模块，寻找_FourierFT_qkv_timm实例
        for name, module in self.fourier_vit.named_modules():
            if isinstance(module, _FourierFT_qkv_timm):
                # 查找base_vit中对应的qkv层
                base_module = self._find_corresponding_base_module(name)
                if base_module is not None and hasattr(base_module, 'qkv') and hasattr(base_module.qkv, 'weight'):
                    # 记录重置前的权重差异用于验证
                    before_reset_diff = torch.sum(torch.abs(module.qkv.weight.data - base_module.qkv.weight.data))
                    
                    # 重置qkv层权重到基础状态
                    module.qkv.weight.data = copy.deepcopy(base_module.qkv.weight.data)
                    
                    # 记录重置后的权重差异用于验证
                    after_reset_diff = torch.sum(torch.abs(module.qkv.weight.data - base_module.qkv.weight.data))
                    
                    # 验证输出
                    # print(f"重置模块 {name} 的qkv层权重:")
                    # print(f"  重置前差异: {before_reset_diff.item():.6f}")
                    # print(f"  重置后差异: {after_reset_diff.item():.6f}")
                    # print(f"  重置成功: {after_reset_diff.item() < 1e-6}")
                    
                    # 设置merged为False，表示增量未合并到基础层
                    module.fourier_q.merged = False
                    module.fourier_v.merged = False
                    
                    reset_count += 1
        
        print(f"\n总共重置了 {reset_count} 个模块的qkv层权重")
                    
    def _find_corresponding_base_module(self, module_name):
        """查找base_vit中对应的模块"""
        # 移除可能的wrapper名称前缀
        parts = module_name.split('.')
        base_parts = []
        for part in parts:
            if part == 'fourier_v' or part == 'fourier_q' or part == 'qkv':
                continue
            base_parts.append(part)
        
        # 构建base_vit中的路径
        base_module = self.base_vit
        for part in base_parts:
            if hasattr(base_module, part):
                base_module = getattr(base_module, part)
            else:
                # 如果找不到对应的模块，尝试查找子模块
                found = False
                for child_name, child_module in base_module.named_children():
                    if child_name == part:
                        base_module = child_module
                        found = True
                        break
                if not found:
                    return None
        
        return base_module