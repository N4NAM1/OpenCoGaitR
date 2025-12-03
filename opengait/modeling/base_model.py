"""The base model definition.

This module defines the abstract meta model class and base model class. In the base model,
 we define the basic model functions, like get_loader, build_network, and run_train, etc.
 The api of the base model is run_train and run_test, they are used in `opengait/main.py`.

Typical usage:

BaseModel.run_train(model)
BaseModel.run_test(model)
"""
import torch
import numpy as np
import os.path as osp
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as tordata

from tqdm import tqdm
from torch.cuda.amp import autocast
from torch.cuda.amp import GradScaler
from abc import ABCMeta
from abc import abstractmethod

from . import backbones
from .loss_aggregator import LossAggregator
from data.transform import get_transform
from data.collate_fn import get_collate_fn
from data.dataset import GaitCIRDataset as DataSet
import data.sampler as Samplers
from utils import Odict, mkdir, ddp_all_gather
from utils import get_valid_args, is_list, is_dict, np2var, ts2np, list2var, get_attr_from
from evaluation import evaluator as eval_functions
from utils import NoOp
from utils import get_msg_mgr
from transformers import CLIPProcessor

__all__ = ['BaseModel']


class MetaModel(metaclass=ABCMeta):
    """The necessary functions for the base model.

    This class defines the necessary functions for the base model, in the base model, we have implemented them.
    """
    @abstractmethod
    def get_loader(self, data_cfg):
        """Based on the given data_cfg, we get the data loader."""
        raise NotImplementedError

    @abstractmethod
    def build_network(self, model_cfg):
        """Build your network here."""
        raise NotImplementedError

    @abstractmethod
    def init_parameters(self):
        """Initialize the parameters of your network."""
        raise NotImplementedError

    @abstractmethod
    def get_optimizer(self, optimizer_cfg):
        """Based on the given optimizer_cfg, we get the optimizer."""
        raise NotImplementedError

    @abstractmethod
    def get_scheduler(self, scheduler_cfg):
        """Based on the given scheduler_cfg, we get the scheduler."""
        raise NotImplementedError

    @abstractmethod
    def save_ckpt(self, iteration):
        """Save the checkpoint, including model parameter, optimizer and scheduler."""
        raise NotImplementedError

    @abstractmethod
    def resume_ckpt(self, restore_hint):
        """Resume the model from the checkpoint, including model parameter, optimizer and scheduler."""
        raise NotImplementedError

    @abstractmethod
    def inputs_pretreament(self, inputs):
        """Transform the input data based on transform setting."""
        raise NotImplementedError

    @abstractmethod
    def train_step(self, loss_num) -> bool:
        """Do one training step."""
        raise NotImplementedError

    @abstractmethod
    def inference(self):
        """Do inference (calculate features.)."""
        raise NotImplementedError

    @abstractmethod
    def run_train(model):
        """Run a whole train schedule."""
        raise NotImplementedError

    @abstractmethod
    def run_test(model):
        """Run a whole test schedule."""
        raise NotImplementedError


class BaseModel(MetaModel, nn.Module):
    """Base model.

    This class inherites the MetaModel class, and implements the basic model functions, like get_loader, build_network, etc.

    Attributes:
        msg_mgr: the massage manager.
        cfgs: the configs.
        iteration: the current iteration of the model.
        engine_cfg: the configs of the engine(train or test).
        save_path: the path to save the checkpoints.

    """

    def __init__(self, cfgs, training):
        """Initialize the base model.

        Complete the model initialization, including the data loader, the network, the optimizer, the scheduler, the loss.

        Args:
        cfgs:
            All of the configs.
        training:
            Whether the model is in training mode.
        """

        super(BaseModel, self).__init__()
        self.msg_mgr = get_msg_mgr()
        self.cfgs = cfgs
        self.iteration = 0
        self.engine_cfg = cfgs['trainer_cfg'] if training else cfgs['evaluator_cfg']
        if self.engine_cfg is None:
            raise Exception("Initialize a model without -Engine-Cfgs-")

        if training and self.engine_cfg['enable_float16']:
            self.Scaler = GradScaler()
        self.save_path = osp.join('output/', cfgs['data_cfg']['dataset_name'],
                                  cfgs['model_cfg']['model'], self.engine_cfg['save_name'])

        self.build_network(cfgs['model_cfg'])
        self.init_parameters()
        self.trainer_trfs = get_transform(cfgs['trainer_cfg']['transform'])

        self.msg_mgr.log_info(cfgs['data_cfg'])
        if training:
            self.train_loader = self.get_loader(
                cfgs['data_cfg'], train=True)
        if not training or self.engine_cfg['with_test']:
            self.test_loader = self.get_loader(
                cfgs['data_cfg'], train=False)
            self.evaluator_trfs = get_transform(
                cfgs['evaluator_cfg']['transform'])

        self.device = torch.distributed.get_rank()
        torch.cuda.set_device(self.device)
        self.to(device=torch.device(
            "cuda", self.device))

        if training:
            self.loss_aggregator = LossAggregator(cfgs['loss_cfg'])
            self.optimizer = self.get_optimizer(self.cfgs['optimizer_cfg'])
            self.scheduler = self.get_scheduler(cfgs['scheduler_cfg'])
        self.train(training)
        restore_hint = self.engine_cfg['restore_hint']
        if restore_hint != 0:
            self.resume_ckpt(restore_hint)

    def get_backbone(self, backbone_cfg):
        """Get the backbone of the model."""
        if is_dict(backbone_cfg):
            Backbone = get_attr_from([backbones], backbone_cfg['type'])
            valid_args = get_valid_args(Backbone, backbone_cfg, ['type'])
            return Backbone(**valid_args)
        if is_list(backbone_cfg):
            Backbone = nn.ModuleList([self.get_backbone(cfg)
                                      for cfg in backbone_cfg])
            return Backbone
        raise ValueError(
            "Error type for -Backbone-Cfg-, supported: (A list of) dict.")

    def build_network(self, model_cfg):
        if 'backbone_cfg' in model_cfg.keys():
            self.Backbone = self.get_backbone(model_cfg['backbone_cfg'])

    def init_parameters(self):
        for m in self.modules():
            if isinstance(m, (nn.Conv3d, nn.Conv2d, nn.Conv1d)):
                nn.init.xavier_uniform_(m.weight.data)
                if m.bias is not None:
                    nn.init.constant_(m.bias.data, 0.0)
            elif isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight.data)
                if m.bias is not None:
                    nn.init.constant_(m.bias.data, 0.0)
            elif isinstance(m, (nn.BatchNorm3d, nn.BatchNorm2d, nn.BatchNorm1d)):
                if m.affine:
                    nn.init.normal_(m.weight.data, 1.0, 0.02)
                    nn.init.constant_(m.bias.data, 0.0)

    def get_loader(self, data_cfg, train=True):
        """
        [GaitCIR 适配版]
        1. 从 trainer_cfg/evaluator_cfg 的 sampler 中读取 batch_size 和帧数。
        2. 使用 BatchSampler 包装 DistributedSampler。
        """
        mode_str = 'train' if train else 'test'
        
        # 1. 获取对应的引擎配置 (Trainer 或 Evaluator)
        engine_cfg = self.cfgs['trainer_cfg'] if train else self.cfgs['evaluator_cfg']
        sampler_cfg = engine_cfg['sampler']
        
        # 🔥 FIX: 从 sampler 配置中读取 batch_size
        batch_size = sampler_cfg['batch_size']
        
        # 2. 将帧数参数注入到 data_cfg 副本中 (Dataset 需要这些参数)
        data_cfg_copy = data_cfg.copy()
        if mode_str == 'train':
            # 读取 train_max_frames (默认 30)
            data_cfg_copy['train_max_frames'] = sampler_cfg.get('train_max_frames', 30)
        else:
            # 读取 test_max_frames (默认 all)
            data_cfg_copy['test_max_frames'] = sampler_cfg.get('test_max_frames', 'all')
            
        # 3. 实例化 Dataset
        dataset = DataSet(data_cfg_copy, mode_str)
        
        # 4. 准备 Processor
        backbone_name = self.cfgs['model_cfg'].get('backbone', "openai/clip-vit-base-patch32")
        processor = CLIPProcessor.from_pretrained(backbone_name)
        
        # 5. 构建 Sampler
        base_sampler = tordata.distributed.DistributedSampler(dataset, shuffle=train)
        
        # 6. 构建 BatchSampler (显式批处理)
        batch_sampler = tordata.BatchSampler(
            sampler=base_sampler,
            batch_size=batch_size,
            drop_last=train
        )
        
        # 7. 构建 DataLoader
        loader = tordata.DataLoader(
            dataset=dataset,
            batch_sampler=batch_sampler,
            collate_fn=get_collate_fn(processor, mode=mode_str),
            num_workers=data_cfg['num_workers'],
            pin_memory=True
        )
        
        return loader

    def get_optimizer(self, optimizer_cfg):
        self.msg_mgr.log_info(optimizer_cfg)
        optimizer = get_attr_from([optim], optimizer_cfg['solver'])
        valid_arg = get_valid_args(optimizer, optimizer_cfg, ['solver'])
        optimizer = optimizer(
            filter(lambda p: p.requires_grad, self.parameters()), **valid_arg)
        return optimizer

    def get_scheduler(self, scheduler_cfg):
            """ [GaitCIR] 动态调度器构建，修复了 'cosine' 调度器冲突 """
            sch_cfg = scheduler_cfg.copy()
            sch_name = sch_cfg.pop('scheduler', 'MultiStepLR')
            
            self.msg_mgr.log_info(sch_cfg)

            # 1. 🔥 修复：直接处理 HuggingFace 风格的调度器 (如 cosine, linear)
            if sch_name == 'cosine' or sch_name == 'linear':
                from transformers import get_scheduler
                
                # total_iter (或 num_training_steps) 需要从 trainer_cfg 中获取
                total_steps = self.cfgs['trainer_cfg'].get('total_iter', 60000)
                
                scheduler = get_scheduler(
                    sch_name, 
                    self.optimizer, 
                    num_warmup_steps=sch_cfg.get('warmup_steps', 0),
                    num_training_steps=total_steps
                )
                
            # 2. 兼容 OpenGait 的 PyTorch 原生调度器
            else:
                # 这一步会使用 OpenGait 的 get_attr_from 工具查找
                Scheduler = get_attr_from([torch.optim.lr_scheduler], sch_name) 
                valid_args = self._get_valid_args(Scheduler, sch_cfg)
                scheduler = Scheduler(self.optimizer, **valid_args)

            self.msg_mgr.log_info(f"Scheduler: {sch_name}")
            return scheduler

    def save_ckpt(self, iteration):
            if torch.distributed.get_rank() == 0:
                mkdir(osp.join(self.save_path, "checkpoints/"))
                save_name = self.engine_cfg['save_name']
                
                # 🔥 核心修改：过滤掉冻结的 CLIP 参数
                # 获取完整状态字典
                raw_state_dict = self.state_dict()
                # 只保留不以 'clip.' 开头的参数 (即保留 combiner 和 logit_scale)
                # 这样文件会变得非常小！
                filtered_state_dict = {k: v for k, v in raw_state_dict.items() if not k.startswith('clip.')}
                
                checkpoint = {
                    'model': filtered_state_dict,  # ✅ 存瘦身后的字典
                    'optimizer': self.optimizer.state_dict(),
                    'scheduler': self.scheduler.state_dict(),
                    'iteration': iteration
                }
                
                save_path = osp.join(self.save_path, 'checkpoints/{}-{:0>5}.pt'.format(save_name, iteration))
                
                # 禁用压缩以提高稳定性和速度
                torch.save(checkpoint, save_path, _use_zipfile=False)
                self.msg_mgr.log_info(f"💾 Checkpoint saved (Lightweight): {save_path}")

    def _load_ckpt(self, save_name):
        load_ckpt_strict = self.engine_cfg['restore_ckpt_strict']

        checkpoint = torch.load(save_name, map_location=torch.device(
            "cuda", self.device))
        model_state_dict = checkpoint['model']

        if not load_ckpt_strict:
            self.msg_mgr.log_info("-------- Restored Params List --------")
            self.msg_mgr.log_info(sorted(set(model_state_dict.keys()).intersection(
                set(self.state_dict().keys()))))

        self.load_state_dict(model_state_dict, strict=load_ckpt_strict)
        if self.training:
            if not self.engine_cfg["optimizer_reset"] and 'optimizer' in checkpoint:
                self.optimizer.load_state_dict(checkpoint['optimizer'])
            else:
                self.msg_mgr.log_warning(
                    "Restore NO Optimizer from %s !!!" % save_name)
            if not self.engine_cfg["scheduler_reset"] and 'scheduler' in checkpoint:
                self.scheduler.load_state_dict(
                    checkpoint['scheduler'])
            else:
                self.msg_mgr.log_warning(
                    "Restore NO Scheduler from %s !!!" % save_name)
        self.msg_mgr.log_info("Restore Parameters from %s !!!" % save_name)

    def resume_ckpt(self, restore_hint):
            """
            从检查点恢复模型权重、优化器和调度器状态。
            [GaitCIR 适配] 核心修改：使用 strict=False 以允许加载部分权重（因为我们没存冻结的 CLIP 参数）。
            """
            # 1. 解析 restore_hint，确定文件路径
            if isinstance(restore_hint, int):
                save_name = self.engine_cfg['save_name']
                save_name = osp.join(
                    self.save_path, 'checkpoints/{}-{:0>5}.pt'.format(save_name, restore_hint))
                self.iteration = restore_hint
            elif isinstance(restore_hint, str):
                save_name = restore_hint
                self.iteration = 0
            else:
                raise ValueError("Error type for -Restore_Hint-, supported: int or string.")

            # 2. 加载文件
            if not osp.exists(save_name):
                raise FileNotFoundError(f"Checkpoint not found: {save_name}")
                
            checkpoint = torch.load(save_name, map_location=torch.device("cuda", self.device))

            # 3. 加载模型权重
            # 🔥 核心修改：strict=False
            # 这样即使 checkpoint 里没有 clip.xxx 的参数，也不会报错
            # (因为 CLIP 参数在 build_network 时已经由 from_pretrained 加载好了)
            missing_keys, unexpected_keys = self.load_state_dict(checkpoint['model'], strict=False)
            
            # 打印加载情况 (可选调试用)
            if len(missing_keys) > 0:
                # 过滤掉预期中缺失的 clip 参数，只打印真正异常的缺失
                real_missing = [k for k in missing_keys if not k.startswith("clip.")]
                if real_missing:
                    self.msg_mgr.log_warning(f"Missing keys: {real_missing}")
                else:
                    self.msg_mgr.log_info(f"Restored model (CLIP parameters skipped as expected).")

            # 4. 加载优化器和调度器 (仅在训练模式下)
            if self.training:
                if 'optimizer' in checkpoint and not self.engine_cfg.get("optimizer_reset", False):
                    self.optimizer.load_state_dict(checkpoint['optimizer'])
                else:
                    self.msg_mgr.log_warning("Restore NO Optimizer from %s !!!" % save_name)
                    
                if 'scheduler' in checkpoint and not self.engine_cfg.get("scheduler_reset", False):
                    self.scheduler.load_state_dict(checkpoint['scheduler'])
                else:
                    self.msg_mgr.log_warning("Restore NO Scheduler from %s !!!" % save_name)

            self.msg_mgr.log_info("Restored from %s" % save_name)

    def fix_BN(self):
        for module in self.modules():
            classname = module.__class__.__name__
            if classname.find('BatchNorm') != -1:
                module.eval()

    def inputs_pretreament(self, inputs):
        """
        [GaitCIR 修复] 简化版预处理：只负责将 Tensor 搬运到 GPU。
        输入 outputs: (ref, tar, txt, mask, inv, inv_mask) - 6 elements
        """
        # 遍历 inputs 中的所有元素。如果是 Tensor，则搬运到 GPU。
        # 注意：这里我们使用 .cuda() 和 non_blocking=True 来实现高效的异步数据传输。
        processed_inputs = []
        for i in inputs:
            if isinstance(i, torch.Tensor):
                processed_inputs.append(i.cuda(non_blocking=True))
            else:
                # 字符串/List/Dict 等非 Tensor 数据原样返回 (如 tasks, metas)
                processed_inputs.append(i)
                
        # 返回处理后的列表，model.forward() 会接收它
        return processed_inputs
    
    def train_step(self, loss_sum) -> bool:
        """Conduct loss_sum.backward(), self.optimizer.step() and self.scheduler.step().

        Args:
            loss_sum:The loss of the current batch.
        Returns:
            bool: True if the training is finished, False otherwise.
        """

        self.optimizer.zero_grad()
        if loss_sum <= 1e-9:
            self.msg_mgr.log_warning(
                "Find the loss sum less than 1e-9 but the training process will continue!")

        if self.engine_cfg['enable_float16']:
            self.Scaler.scale(loss_sum).backward()
            self.Scaler.step(self.optimizer)
            scale = self.Scaler.get_scale()
            self.Scaler.update()
            # Warning caused by optimizer skip when NaN
            # https://discuss.pytorch.org/t/optimizer-step-before-lr-scheduler-step-error-using-gradscaler/92930/5
            if scale != self.Scaler.get_scale():
                self.msg_mgr.log_debug("Training step skip. Expected the former scale equals to the present, got {} and {}".format(
                    scale, self.Scaler.get_scale()))
                return False
        else:
            loss_sum.backward()
            self.optimizer.step()

        self.iteration += 1
        self.scheduler.step()
        return True

    def inference(self, rank):
        """Inference all the test data.

        Args:
            rank: the rank of the current process.Transform
        Returns:
            Odict: contains the inference results.
        """
        total_size = len(self.test_loader)
        if rank == 0:
            pbar = tqdm(total=total_size, desc='Transforming')
        else:
            pbar = NoOp()
        batch_size = self.test_loader.batch_sampler.batch_size
        rest_size = total_size
        info_dict = Odict()
        for inputs in self.test_loader:
            ipts = self.inputs_pretreament(inputs)
            with autocast(enabled=self.engine_cfg['enable_float16']):
                retval = self.forward(ipts)
                inference_feat = retval['inference_feat']
                for k, v in inference_feat.items():
                    inference_feat[k] = ddp_all_gather(v, requires_grad=False)
                del retval
            for k, v in inference_feat.items():
                inference_feat[k] = ts2np(v)
            info_dict.append(inference_feat)
            rest_size -= batch_size
            if rest_size >= 0:
                update_size = batch_size
            else:
                update_size = total_size % batch_size
            pbar.update(update_size)
        pbar.close()
        for k, v in info_dict.items():
            v = np.concatenate(v)[:total_size]
            info_dict[k] = v
        return info_dict

    @staticmethod
    def run_train(model):
        """
        [GaitCIR 修复版] 核心训练循环：直接从模型获取 Loss，并手动构造日志字典。
        """
        model.train()
        model.msg_mgr.log_info(f"🚀 Start training ...")
        
        # 1. 获取 Epoch 长度
        total_iter = model.engine_cfg['total_iter'] 
        loader = model.train_loader
        
        # 兼容 OpenGait 的基于 Iteration 的进度条
        pbar = tqdm(total=total_iter, initial=model.iteration, desc='Training') if torch.distributed.get_rank() == 0 else None
        
        # 2. 开始训练循环
        for i, inputs in enumerate(loader):
            # 2.1. 搬运数据
            ipts = model.inputs_pretreament(inputs)
            
            # 2.2. 前向计算
            with autocast(enabled=model.engine_cfg['enable_float16']):
                # 🔥 FIX: 模型返回 {'loss': ..., 'acc_loss': ..., 'inv_loss': ...}
                retval = model(ipts)
                loss_sum = retval['loss'] # 直接从模型输出中提取总 Loss

            # 2.3. 反向更新
            ok = model.train_step(loss_sum)
            
            if not ok: # 混合精度训练可能跳过
                continue

            # 2.4. 🔥 FIX: 构造日志信息 (Log & Tensorboard)
            visual_summary = {
                'scalar/learning_rate': model.optimizer.param_groups[0]['lr']
            }
            loss_info = {}
            
            # 将模型返回的所有 Loss 分量用于日志记录
            for k, v in retval.items():
                if isinstance(v, torch.Tensor) and v.numel() == 1:
                    loss_info[k] = v.item() # Loss Aggregator 需要这个字典
                    visual_summary[f'scalar/{k}'] = v.item() # Tensorboard 需要这个字典

            # 2.5. 日志记录
            model.msg_mgr.train_step(loss_info, visual_summary)
            
            if pbar: pbar.update(1)

            # 2.6. 保存与测试
            if model.iteration % model.engine_cfg['save_iter'] == 0:
                model.save_ckpt(model.iteration)
                if model.engine_cfg['with_test']:
                    model.eval()
                    BaseModel.run_test(model) # 你的测试逻辑
                    model.train()
            
            if model.iteration >= total_iter:
                if pbar: pbar.close()
                break

    @ staticmethod
    def run_test(model):
        """Accept the instance object(model) here, and then run the test loop."""
        evaluator_cfg = model.cfgs['evaluator_cfg']
        if torch.distributed.get_world_size() != evaluator_cfg['sampler']['batch_size']:
            raise ValueError("The batch size ({}) must be equal to the number of GPUs ({}) in testing mode!".format(
                evaluator_cfg['sampler']['batch_size'], torch.distributed.get_world_size()))
        rank = torch.distributed.get_rank()
        with torch.no_grad():
            info_dict = model.inference(rank)
        if rank == 0:
            loader = model.test_loader
            label_list = loader.dataset.label_list
            types_list = loader.dataset.types_list
            views_list = loader.dataset.views_list

            info_dict.update({
                'labels': label_list, 'types': types_list, 'views': views_list})

            if 'eval_func' in evaluator_cfg.keys():
                eval_func = evaluator_cfg["eval_func"]
            else:
                eval_func = 'identification'
            eval_func = getattr(eval_functions, eval_func)
            valid_args = get_valid_args(
                eval_func, evaluator_cfg, ['metric'])
            try:
                dataset_name = model.cfgs['data_cfg']['test_dataset_name']
            except:
                dataset_name = model.cfgs['data_cfg']['dataset_name']
            return eval_func(info_dict, dataset_name, **valid_args)
