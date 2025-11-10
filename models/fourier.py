# 导入必要的库和模块
import logging
import json
import numpy as np
import torch
from torch import nn
from torch.serialization import load
from tqdm import tqdm
from torch import optim
from torch.nn import functional as F
from torch.utils.data import DataLoader
from utils.inc_net import IncrementalNet
from models.base import BaseLearner
from utils.toolkit import target2onehot, tensor2numpy

import timm
from backbone.fourierft import FourierFT_ViT_timm, GlobalIndicesStorage
import torch.distributed as dist
import os

num_workers = 8

class Learner(BaseLearner):
    """
    实现完整傅里叶微调的学习器，包含任务特定分支和权重合并策略
    """
    def __init__(self, args):
        super().__init__(args)
        self._network = IncrementalNet(args, True)
        
        # 傅里叶微调特定参数
        self.adapters_history = []  # 记录历史适配器
        # 存储傅里叶配置参数
        self.fourier_params = {
            'n_frequency': args['fourierft']['n_frequency'],
            'n_frequency_non_trainable': args['fourierft']['n_frequency_non_trainable'],
            'scaling': args['fourierft']['scaling'],
            'random_loc_seed': args['fourierft']['random_loc_seed'],
            'init_weights': args['fourierft']['init_weights'],
            'prefer': args['fourierft']['prefer'],
            'init_fc': args['fourierft']['init_fc'],
            'round_number': args['round_number'],
            'reinit_num': args['reinit_num'],
            'indices_file_path': args['indices_file_path']
        }
    
    def _update_old_tasks_global_spectrum(self):
        """更新所有旧任务的全局频谱参数为当前任务的全局频谱参数"""
        current_adapter = f"task_{self._cur_task}"
        
        # 获取模型中的所有FourierFT层
        fourier_layers = []
        if hasattr(self._network, 'module'):
            # 处理DataParallel情况
            for module in self._network.module.modules():
                if hasattr(module, 'fourierft_spectrum') and isinstance(module.fourierft_spectrum, nn.ParameterDict):
                    fourier_layers.append(module)
        else:
            for module in self._network.modules():
                if hasattr(module, 'fourierft_spectrum') and isinstance(module.fourierft_spectrum, nn.ParameterDict):
                    fourier_layers.append(module)
        
        # 遍历所有FourierFT层
        for layer in fourier_layers:
            # 检查当前层是否有当前任务的频谱参数
            if current_adapter in layer.fourierft_spectrum:
                current_spectrum = layer.fourierft_spectrum[current_adapter].data.clone()
                
                # 获取全局索引（与adapter无关）
                global_indices_key = "global"
                if global_indices_key in GlobalIndicesStorage._global_indices:
                    global_indices = GlobalIndicesStorage._global_indices[global_indices_key]
                    
                    # 更新所有旧任务的全局频谱参数
                    for old_adapter in self.adapters_history[:-1]:  # 除了当前任务
                        if old_adapter in layer.fourierft_spectrum:
                            old_spectrum = layer.fourierft_spectrum[old_adapter].data
                            
                            # 获取全局索引（与adapter无关）
                            old_global_key = "global"
                            if old_global_key in GlobalIndicesStorage._global_indices:
                                old_global_indices = GlobalIndicesStorage._global_indices[old_global_key]
                                
                            # 确保索引长度匹配
                            min_len = min(len(global_indices), len(old_global_indices))
                                
                            # 更新旧任务的全局频谱参数
                            if len(old_spectrum) >= min_len:
                                # 复制当前任务的全局频谱参数到旧任务的全局索引位置
                                with torch.no_grad():
                                    old_spectrum[:min_len] = current_spectrum[:min_len]
        
        print(f"成功更新所有旧任务的全局频谱参数为任务_{self._cur_task}的全局频谱参数")


    def incremental_train(self, data_manager):
        """增量训练主流程"""
        self._total_classes = self._known_classes + data_manager.get_task_size(self._cur_task)
        self._network.update_fc(self._total_classes)
        
        logging.info("Learning on {}-{}".format(self._known_classes, self._total_classes))

        # 获取数据
        train_dataset = data_manager.get_dataset(
            np.arange(self._known_classes, self._total_classes), source="train", mode="train"
        )
        self.train_loader = DataLoader(
            train_dataset, batch_size=self.args["batch_size"], shuffle=True, num_workers=num_workers
        )
        
        test_dataset = data_manager.get_dataset(
            np.arange(0, self._total_classes), source="test", mode="test"
        )
        self.test_loader = DataLoader(
            test_dataset, batch_size=self.args["batch_size"], shuffle=False, num_workers=num_workers
        )

        # 多GPU支持
        if len(self._multiple_gpus) > 1:
            self._network = nn.DataParallel(self._network, self._multiple_gpus)

        # 更新傅里叶参数的任务ID和轮次编号
        self.fourier_params['task_id'] = self._cur_task
        self.fourier_params['round_number'] = self._cur_task  # round_number随增量任务递增
        
        # 训练
        self._train(self.train_loader, self.test_loader)

        # 恢复单GPU模式
        if len(self._multiple_gpus) > 1:
            self._network = self._network.module

    def _get_dynamic_n_frequency(self):
        """根据任务复杂度动态调整频率数量"""
        if self._cur_task >= 4 and self._cur_task < 8:
            return 3 * self.fourier_params['n_frequency']  # 中等任务复杂度
        elif self._cur_task >= 8:
            return 2 * self.fourier_params['n_frequency']  # 高任务复杂度
        else:
            return self.fourier_params['n_frequency']  # 默认

    def _train(self, train_loader, test_loader):
        """训练主循环"""
        self._network.to(self._device)
        
        # 为当前任务创建新的适配器名称
        current_adapter = f"task_{self._cur_task}"
        
        if self._cur_task == 0:
            # 初始任务训练
            optimizer = optim.SGD(
                self._network.parameters(),
                momentum=0.9,
                lr=self.args["init_lr"],
            )
            scheduler = optim.lr_scheduler.CosineAnnealingLR(
                optimizer=optimizer,
                T_max=self.args["init_epoch"],
                eta_min=0.00001  # 最小学习率
            )
            self._init_train(train_loader, test_loader, optimizer, scheduler)
        else:
            # 增量任务训练
            if len(self._multiple_gpus) > 1:
                self._network = self._network.module
            
            # 直接更新backbone的任务相关参数
            if hasattr(self._network.backbone, 'task_id'):
                self._network.backbone.task_id = self._cur_task
            if hasattr(self._network.backbone, 'round_number'):
                self._network.backbone.round_number = self._cur_task
            
            # 训练前重置权重到基础状态，移除之前融合的增量权重
            self._network.backbone.reset_weights_to_base()
            
            # 遍历所有模块，更新任务ID并为新任务创建适配器
            if hasattr(self._network.backbone, 'fourier_vit'):
                for module in self._network.backbone.fourier_vit.modules():
                    if hasattr(module, 'task_id'):
                        module.task_id = self._cur_task
                    if hasattr(module, 'round_number'):
                        module.round_number = self._cur_task
                    # 为每个具有update_layer方法的模块调用该方法，创建新任务的适配器
                    if hasattr(module, 'update_layer'):
                        module.update_layer(current_adapter)
            
            if len(self._multiple_gpus) > 1:
                self._network = nn.DataParallel(self._network, self._multiple_gpus)
            
            self._network.to(self._device)

            optimizer = optim.SGD(
                self._network.parameters(),
                lr=self.args["lrate"],
                momentum=0.9,
            )
            scheduler = optim.lr_scheduler.CosineAnnealingLR(
                optimizer=optimizer,
                T_max=self.args["epochs"],
                eta_min=0.00001  # 最小学习率
            )
            
            # 训练当前任务
            self._update_representation(train_loader, test_loader, optimizer, scheduler)
        
        save_fourierft_name = self.fourier_params['indices_file_path']
        if len(self._multiple_gpus) > 1:
            self._network.module.save_fc(save_fourierft_name, self._cur_task)
        else:
            self._network.save_fc(save_fourierft_name, self._cur_task)
        
        # 任务结束后处理，记录适配器信息并更新旧任务的全局频谱参数
        self._known_classes = self._total_classes
        
        # 记录当前任务的适配器，确保与训练前创建的适配器名称一致
        self.adapters_history.append(current_adapter)
        
        # 更新所有旧任务的全局频谱参数为当前任务的全局频谱参数
        # if len(self.adapters_history) > 1:
        #     self._update_old_tasks_global_spectrum()
        
        # 训练结束后，合并之前任务的适配器增量权重
        # if len(self.adapters_history) > 0:
        #     # 获取之前所有任务的适配器名称
        #     previous_adapters = self.adapters_history[:]
                
        #     # 合并策略
        #     strategy = self.args.get('merge_strategy', 'max')
                
        #     # 应用合并
        #     if len(self._multiple_gpus) > 1:
        #         self._network.module.backbone.merge_adapters(previous_adapters, strategy)
        #     else:
        #         self._network.backbone.merge_adapters(previous_adapters, strategy)
                
        #     # 合并后在测试集上进行测试
        #     test_acc = self._compute_accuracy(self._network, test_loader)
        #     info = "Task {}, After merge adapters => Test_accy {:.2f}".format(
        #         self._cur_task, test_acc
        #     )
        #     logging.info(info)
        
        self._cur_task += 1

    def get_optimizer(self):
        """获取优化器"""
        if self.args['optimizer'] == 'sgd':
            optimizer = optim.SGD(
                filter(lambda p: p.requires_grad, self._network.parameters()),
                momentum=0.9,
                lr=self.init_lr,
                weight_decay=self.weight_decay
            )
        elif self.args['optimizer'] == 'adam':
            optimizer = optim.Adam(
                filter(lambda p: p.requires_grad, self._network.parameters()),
                lr=self.args["lrate"],
                betas=(0.9, 0.999)
            )
        elif self.args['optimizer'] == 'adamw':
            optimizer = optim.AdamW(
                filter(lambda p: p.requires_grad, self._network.parameters()),
                lr=self.init_lr,
                weight_decay=self.weight_decay
            )
        return optimizer

    def get_scheduler(self, optimizer):
        """获取学习率调度器"""
        if self.args["scheduler"] == 'cosine':
            scheduler = optim.lr_scheduler.CosineAnnealingLR(
                optimizer=optimizer,
                T_max=self.args['tuned_epoch'],
                eta_min=self.args['min_lr']
            )
        elif self.args["scheduler"] == 'steplr':
            scheduler = optim.lr_scheduler.MultiStepLR(
                optimizer=optimizer,
                milestones=self.args["init_milestones"],
                gamma=self.args["init_lr_decay"]
            )
        elif self.args["scheduler"] == 'constant':
            scheduler = None
        return scheduler

    def _init_train(self, train_loader, test_loader, optimizer, scheduler):
        """初始任务训练"""
        prog_bar = tqdm(range(self.args["init_epoch"]))
        for _, epoch in enumerate(prog_bar):
            self._network.train()
            losses = 0.0
            correct, total = 0, 0
            
            for i, (_, inputs, targets) in enumerate(train_loader):
                inputs, targets = inputs.to(self._device), targets.to(self._device)
                logits = self._network(inputs)["logits"]
                loss_clf = F.cross_entropy(logits, targets)
                
                # 添加L2正则化项
                l2_reg = 0.0
                l2_lambda = 1e-5  # 正则化系数
                for param in self._network.parameters():
                    if param.requires_grad:
                        l2_reg += torch.norm(param, 2)
                
                loss = loss_clf + l2_lambda * l2_reg
                
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                
                losses += loss.item()
                _, preds = torch.max(logits, dim=1)
                correct += preds.eq(targets.expand_as(preds)).cpu().sum()
                total += len(targets)

            scheduler.step()
            train_acc = np.around(tensor2numpy(correct) * 100 / total, decimals=2)
            
            if epoch % 5 == 0:
                test_acc = self._compute_accuracy(self._network, test_loader)
                info = "Task {}, Epoch {}/{} => Loss {:.3f}, Train_accy {:.2f}, Test_accy {:.2f}".format(
                    self._cur_task, epoch + 1, self.args["init_epoch"],
                    losses / len(train_loader), train_acc, test_acc
                )
            else:
                info = "Task {}, Epoch {}/{} => Loss {:.3f}, Train_accy {:.2f}".format(
                    self._cur_task, epoch + 1, self.args["init_epoch"],
                    losses / len(train_loader), train_acc
                )
            
            prog_bar.set_description(info)
        
        logging.info(info)

    def _update_representation(self, train_loader, test_loader, optimizer, scheduler):
        """增量表示学习"""
        prog_bar = tqdm(range(self.args["epochs"]))
        for _, epoch in enumerate(prog_bar):
            self._network.train()
            losses = 0.0
            correct, total = 0, 0
            
            for i, (_, inputs, targets) in enumerate(train_loader):
                inputs, targets = inputs.to(self._device), targets.to(self._device)
                logits = self._network(inputs)["logits"]
                
                # 只对新类别计算损失
                fake_targets = targets - self._known_classes
                loss_clf = F.cross_entropy(
                    logits[:, self._known_classes:],
                    fake_targets
                )
                
                # 添加L2正则化项
                l2_reg = 0.0
                l2_lambda = 1e-4  # 正则化系数
                for param in self._network.parameters():
                    if param.requires_grad:
                        l2_reg += torch.norm(param, 2)
                
                # 傅里叶微调没有正交损失，但添加L2正则
                loss = loss_clf + l2_lambda * l2_reg

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                
                losses += loss.item()
                _, preds = torch.max(logits, dim=1)
                correct += preds.eq(targets.expand_as(preds)).cpu().sum()
                total += len(targets)

            scheduler.step()
            train_acc = np.around(tensor2numpy(correct) * 100 / total, decimals=2)
            
            if epoch % 5 == 0:
                test_acc = self._compute_accuracy(self._network, test_loader)
                info = "Task {}, Epoch {}/{} => Loss {:.3f}, Train_accy {:.2f}, Test_accy {:.2f}".format(
                    self._cur_task, epoch + 1, self.args["epochs"],
                    losses / len(train_loader), train_acc, test_acc
                )
            else:
                info = "Task {}, Epoch {}/{} => Loss {:.3f}, Train_accy {:.2f}".format(
                    self._cur_task, epoch + 1, self.args["epochs"],
                    losses / len(train_loader), train_acc
                )
            
            prog_bar.set_description(info)
        
        logging.info(info)

    def _compute_accuracy(self, model, loader):
        """计算准确率"""
        model.eval()
        correct, total = 0, 0
        with torch.no_grad():
            for i, (_, inputs, targets) in enumerate(loader):
                inputs = inputs.to(self._device)
                outputs = model(inputs)["logits"]
                _, preds = torch.max(outputs, dim=1)
                correct += (preds.cpu() == targets).sum()
                total += len(targets)
        return np.around(tensor2numpy(correct) * 100 / total, decimals=2)