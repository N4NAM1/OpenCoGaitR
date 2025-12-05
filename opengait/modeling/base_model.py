import torch
import numpy as np
import os.path as osp
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as tordata
import inspect
import gc  # 引入 gc 用于内存回收

from tqdm import tqdm
from torch.cuda.amp import autocast
from torch.cuda.amp import GradScaler
from abc import ABCMeta
from abc import abstractmethod

from utils import Odict, mkdir, ddp_all_gather
from utils import get_valid_args, is_list, is_dict, np2var, ts2np, list2var, get_attr_from
from utils import get_msg_mgr

# [GaitCIR] 核心导入
from transformers import CLIPProcessor
from data.dataset import GaitCIRDataset as DataSet 
from data.collate_fn import get_collate_fn
from evaluation.evaluator import evaluate_GaitCIR

__all__ = ['BaseModel']

class BaseModel(nn.Module, metaclass=ABCMeta):
    """
    Base model for GaitCIR in OpenGait.
    """
    def __init__(self, cfgs, training):
        super(BaseModel, self).__init__()
        self.msg_mgr = get_msg_mgr()
        self.cfgs = cfgs
        self.training = training
        self.iteration = 0
        self.engine_cfg = cfgs['trainer_cfg'] if training else cfgs['evaluator_cfg']
        
        # 1. 混合精度设置 (CLIP 训练必须关闭 FP16)
        # 建议在 yaml 里把 enable_float16 设为 false
        if self.engine_cfg.get('enable_float16', False) and training:
            self.Scaler = GradScaler()
        else:
            self.Scaler = None
            
        self.save_path = osp.join('output/', cfgs['data_cfg']['dataset_name'],
                                  cfgs['model_cfg']['model'], self.engine_cfg['save_name'])

        # 2. 构建网络
        self.build_network(cfgs['model_cfg'])
        
        # 3. 初始化参数 (保护 CLIP)
        self.init_parameters()
        
        # 4. 数据加载
        self.msg_mgr.log_info(cfgs['data_cfg'])
        if training:
            self.train_loader = self.get_loader(cfgs['data_cfg'], train=True)
            self.optimizer = self.get_optimizer(self.cfgs['optimizer_cfg'])
            self.scheduler = self.get_scheduler(cfgs['scheduler_cfg'])
        
        if not training or self.engine_cfg['with_test']:
            self.test_loader = self.get_loader(cfgs['data_cfg'], train=False)

        self.device = torch.distributed.get_rank()
        torch.cuda.set_device(self.device)
        self.to(device=torch.device("cuda", self.device))

        self.train(training)
        
        # 5. 恢复权重
        restore_hint = self.engine_cfg.get('restore_hint', 0)
        if restore_hint != 0:
            self.resume_ckpt(restore_hint)

    @abstractmethod
    def build_network(self, model_cfg):
        raise NotImplementedError

    def init_parameters(self):
        """
        [GaitCIR 修复] 初始化参数，但跳过冻结层 (如 CLIP)。
        """
        for name, m in self.named_modules():
            # 🔥 核心保护：如果模块名字包含 clip，或者不需要梯度，直接跳过
            # 这样就能保留 CLIP 的预训练权重，只初始化 Combiner
            if "clip" in name: 
                continue
                
            # 检查该模块是否有需要梯度的参数
            has_grad = False
            for p in m.parameters(recurse=False):
                if p.requires_grad:
                    has_grad = True
                    break
            if not has_grad:
                continue

            # 执行初始化
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
        mode_str = 'train' if train else 'test'
        engine_cfg = self.cfgs['trainer_cfg'] if train else self.cfgs['evaluator_cfg']
        sampler_cfg = engine_cfg['sampler']
        
        default_workers = data_cfg.get('num_workers', 8)
        num_workers = engine_cfg.get('num_workers', default_workers)

        batch_size = sampler_cfg['batch_size']
        
        data_cfg_copy = data_cfg.copy()
        if mode_str == 'train':
            data_cfg_copy['train_max_frames'] = sampler_cfg.get('train_max_frames', 30)
        else:
            data_cfg_copy['test_max_frames'] = sampler_cfg.get('test_max_frames', 'all')
            
        dataset = DataSet(data_cfg_copy, mode_str)
        
        backbone_name = self.cfgs['model_cfg'].get('backbone', "openai/clip-vit-base-patch32")
        processor = CLIPProcessor.from_pretrained(backbone_name)
        
        base_sampler = tordata.distributed.DistributedSampler(dataset, shuffle=train)
        
        # 使用 BatchSampler 显式批处理
        batch_sampler = tordata.BatchSampler(
            sampler=base_sampler, 
            batch_size=batch_size, 
            drop_last=train 
        )
        
        loader = tordata.DataLoader(
            dataset=dataset,
            batch_sampler=batch_sampler, 
            collate_fn=get_collate_fn(processor, mode=mode_str),
            num_workers=num_workers,
            pin_memory=False # 关闭 Pin Memory 减少 OOM 风险
        )
        return loader

    def _get_valid_args(self, cls, kwargs):
        valid_keys = inspect.signature(cls.__init__).parameters.keys()
        return {k: v for k, v in kwargs.items() if k in valid_keys}

    def get_optimizer(self, optimizer_cfg):
        opt_cfg = optimizer_cfg.copy()
        solver_name = opt_cfg.pop('solver', 'AdamW')
        try:
            OptClass = getattr(optim, solver_name)
        except AttributeError:
            raise ValueError(f"Optimizer {solver_name} not found")
            
        valid_args = self._get_valid_args(OptClass, opt_cfg)
        optimizer = OptClass(filter(lambda p: p.requires_grad, self.parameters()), **valid_args)
        self.msg_mgr.log_info(f"Optimizer: {solver_name} {valid_args}")
        return optimizer

    def get_scheduler(self, scheduler_cfg):
        sch_cfg = scheduler_cfg.copy()
        sch_name = sch_cfg.pop('scheduler', 'MultiStepLR')
        
        # total_iter for HF schedulers
        total_steps = self.cfgs['trainer_cfg'].get('total_iter', 60000)

        if sch_name == 'cosine' or sch_name == 'linear':
            from transformers import get_scheduler
            scheduler = get_scheduler(
                sch_name, 
                self.optimizer, 
                num_warmup_steps=sch_cfg.get('warmup_steps', 0),
                num_training_steps=total_steps
            )
        else:
            Scheduler = get_attr_from([torch.optim.lr_scheduler], sch_name) 
            valid_args = self._get_valid_args(Scheduler, sch_cfg)
            scheduler = Scheduler(self.optimizer, **valid_args)

        self.msg_mgr.log_info(f"Scheduler: {sch_name}")
        return scheduler

    def save_ckpt(self, iteration):
        if torch.distributed.get_rank() == 0:
            mkdir(osp.join(self.save_path, "checkpoints/"))
            save_name = self.engine_cfg['save_name']
            
            raw_state_dict = self.state_dict()
            # 🔥 过滤 CLIP 参数，只存 Combiner
            filtered_state_dict = {k: v for k, v in raw_state_dict.items() if not k.startswith('clip.')}
            
            checkpoint = {
                'model': filtered_state_dict,
                'optimizer': self.optimizer.state_dict(),
                'scheduler': self.scheduler.state_dict(),
                'iteration': iteration
            }
            save_path = osp.join(self.save_path, 'checkpoints/{}-{:0>5}.pt'.format(save_name, iteration))
            # 禁用压缩
            torch.save(checkpoint, save_path, _use_new_zipfile_serialization=False)
            self.msg_mgr.log_info(f"💾 Checkpoint saved (Lightweight): {save_path}")

    def resume_ckpt(self, restore_hint):
        if isinstance(restore_hint, int):
            save_name = self.engine_cfg['save_name']
            save_name = osp.join(self.save_path, 'checkpoints/{}-{:0>5}.pt'.format(save_name, restore_hint))
            self.iteration = restore_hint
        elif isinstance(restore_hint, str):
            save_name = restore_hint
            self.iteration = 0
        else:
            raise ValueError("Error type for -Restore_Hint-")

        if not osp.exists(save_name):
            if self.training:
                self.msg_mgr.log_info(f"Checkpoint {save_name} not found, starting from scratch.")
                return
            else:
                raise FileNotFoundError(f"Checkpoint not found: {save_name}")

        checkpoint = torch.load(save_name, map_location=torch.device("cuda", self.device))
        
        # 🔥 strict=False，允许不加载 CLIP 参数
        is_strict = self.engine_cfg.get('restore_ckpt_strict', False)
        missing, unexpected = self.load_state_dict(checkpoint['model'], strict=is_strict)
        
        if not is_strict:
            real_missing = [k for k in missing if not k.startswith("clip.")]
            if real_missing:
                self.msg_mgr.log_warning(f"Missing keys: {real_missing}")
            else:
                self.msg_mgr.log_info(f"Restored model (CLIP parameters skipped as expected).")

        if self.training:
            if 'optimizer' in checkpoint and not self.engine_cfg.get("optimizer_reset", False):
                self.optimizer.load_state_dict(checkpoint['optimizer'])
            if 'scheduler' in checkpoint and not self.engine_cfg.get("scheduler_reset", False):
                self.scheduler.load_state_dict(checkpoint['scheduler'])

        self.msg_mgr.log_info("Restored from %s" % save_name)

    def inputs_pretreament(self, inputs):
        """ [GaitCIR] 搬运 Tensor 到 GPU """
        processed_inputs = []
        for i in inputs:
            if isinstance(i, torch.Tensor):
                processed_inputs.append(i.cuda(non_blocking=True))
            else:
                processed_inputs.append(i)
        return processed_inputs

    def train_step(self, loss) -> bool:
        self.optimizer.zero_grad()
        # 梯度裁剪
        max_grad_norm = self.engine_cfg.get('clip_grad_norm', -1)

        if self.Scaler is not None:
            self.Scaler.scale(loss).backward()
            if max_grad_norm > 0:
                self.Scaler.unscale_(self.optimizer)
                torch.nn.utils.clip_grad_norm_(self.parameters(), max_grad_norm)
            self.Scaler.step(self.optimizer)
            self.Scaler.update()
        else:
            loss.backward()
            if max_grad_norm > 0:
                torch.nn.utils.clip_grad_norm_(self.parameters(), max_grad_norm)
            self.optimizer.step()

        self.iteration += 1
        self.scheduler.step()
        return True

    @staticmethod
    def run_train(model):
        model.train()
        model.msg_mgr.log_info(f"🚀 Start training ...")
        
        total_iter = model.engine_cfg['total_iter'] 
        loader = model.train_loader
        
        # 无限循环加载器
        def infinite_loader(loader):
            while True:
                for batch in loader:
                    yield batch
        train_iter = infinite_loader(loader)

        pbar = tqdm(total=total_iter, initial=model.iteration, desc='Training') if torch.distributed.get_rank() == 0 else None
        
        while model.iteration < total_iter:
            try:
                inputs = next(train_iter)
            except StopIteration:
                train_iter = infinite_loader(loader)
                inputs = next(train_iter)

            # 1. 搬运
            ipts = model.inputs_pretreament(inputs)
            
            # 2. 前向
            with autocast(enabled=model.engine_cfg.get('enable_float16', False)):
                retval = model(ipts)
                loss_sum = retval['loss']

            # 3. 反向
            ok = model.train_step(loss_sum)
            if not ok: continue
            
            # 4. 内存回收 (解决 OOM 关键)
            if model.iteration % 100 == 0:
                gc.collect()

            # 5. 日志
            if model.iteration % model.engine_cfg['log_iter'] == 0:
                visual_summary = {'scalar/learning_rate': model.optimizer.param_groups[0]['lr']}
                loss_info = {}
                for k, v in retval.items():
                    if isinstance(v, torch.Tensor) and v.numel() == 1:
                        loss_info[k] = v.item()
                        visual_summary[f'scalar/{k}'] = v.item()
                model.msg_mgr.train_step(loss_info, visual_summary)
            
            if pbar: pbar.update(1)

            # 6. 保存与测试
            if model.iteration % model.engine_cfg['save_iter'] == 0:
                model.save_ckpt(model.iteration)
                if model.engine_cfg['with_test']:
                    model.eval()
                    BaseModel.run_test(model)
                    model.train()

        if pbar: pbar.close()

    @staticmethod
    def run_test(model):
        """
        [GaitCIR 修复版] 支持多卡 DDP 结果汇聚 (All-Gather)
        """
        model.eval()
        rank = torch.distributed.get_rank()
        world_size = torch.distributed.get_world_size()
        loader = model.test_loader
        
        # 1. 本地推理 (只跑当前卡分到的数据)
        local_q, local_t = [], []
        local_tasks, local_metas = [], []
        
        if rank == 0: pbar = tqdm(loader, desc="🔍 Inference")
        else: pbar = loader
            
        with torch.no_grad():
            for inputs in pbar:
                ipts = model.inputs_pretreament(inputs)
                
                # 开启 FP16 推理
                with autocast(enabled=model.engine_cfg.get('enable_float16', False)):
                    outputs = model(ipts) 
                
                # 必须转回 CPU，否则 gather 时显存会爆炸
                local_q.append(outputs['query_feat'].float().cpu())
                local_t.append(outputs['tar_feat'].float().cpu())
                local_tasks.extend(outputs['tasks'])
                local_metas.extend(outputs['metas'])
                
                if rank == 0: pbar.update(1)

        # 2. 整理本地数据
        local_data = {
            'q': torch.cat(local_q, dim=0),
            't': torch.cat(local_t, dim=0),
            'tasks': local_tasks,
            'metas': local_metas
        }

        # 3. DDP 汇聚 (Gather All)
        # 无论单卡还是多卡，统一走 Gather 逻辑保证一致性
        gathered_data = [None for _ in range(world_size)]
        
        # all_gather_object 可以汇聚任意 Python 对象 (Tensor, List, Dict)
        # 注意：这会涉及序列化，数据量极大时可能会稍慢，但最稳健
        torch.distributed.all_gather_object(gathered_data, local_data)

        # 4. 仅在主进程解包并评测
        if rank == 0:
            print(f"✅ Inference Done. Aggregating results from {world_size} GPUs...")
            
            # 解包并拼接所有卡的数据
            all_q_feats = []
            all_t_feats = []
            all_tasks_final = []
            all_metas_final = []
            
            for node_data in gathered_data:
                all_q_feats.append(node_data['q'])
                all_t_feats.append(node_data['t'])
                all_tasks_final.extend(node_data['tasks'])
                all_metas_final.extend(node_data['metas'])
            
            # 拼接 Tensor
            final_q = torch.cat(all_q_feats, dim=0)
            final_t = torch.cat(all_t_feats, dim=0)
            
            print(f"📊 Total Samples: {len(final_q)} (Local was {len(local_data['q'])})")
            print(f"📊 Computing Metrics...")
            
            # 打包给 Evaluator
            eval_data = {
                'q_feats': final_q,
                'g_feats': final_t,
                'q_metas': all_metas_final,
                'g_metas': all_metas_final,
                'tasks': all_tasks_final
            }
            
            eval_cfg = model.cfgs['evaluator_cfg']
            metric_cfg = eval_cfg.get('metric_cfg', {})
            dataset_name = model.cfgs['data_cfg']['dataset_name']

            evaluate_GaitCIR(eval_data, dataset_name, metric_cfg, save_path=model.save_path)