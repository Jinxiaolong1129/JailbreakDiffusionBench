import json
from pathlib import Path
import time
from dataclasses import dataclass, asdict
from typing import List, Dict, Any
from datetime import datetime
import pandas as pd
import numpy as np
import random
import torch
import string
from tqdm import tqdm
import torch.nn as nn
import gc
from typing import Optional
import torch.nn.functional as F
from .factory import AttackerFactory
from .base import BaseAttacker, AttackResult
from transformers.models.clip import modeling_clip



class CosineSimilarityLoss(nn.Module):
    def __init__(self, reduction='mean'):
        super(CosineSimilarityLoss, self).__init__()
        self.reduction = reduction

    def forward(self, x, y):
        cos_sim = nn.functional.cosine_similarity(x, y, dim=1, eps=1e-6)
        loss = 1 - cos_sim

        if self.reduction == 'mean':
            loss = loss.mean()
        elif self.reduction == 'sum':
            loss = loss.sum()

        return loss

class CosineSimilarityLoss_1d(nn.Module):
    def __init__(self, reduction=None):
        """
        Args:
            reduction (str): 可选值为 'mean', 'sum', 或 None。
                             - 'mean': 返回所有样本的平均损失。
                             - 'sum': 返回所有样本的损失总和。
                             - None: 返回每个样本的独立损失，形状为 [512]。
        """
        super(CosineSimilarityLoss_1d, self).__init__()
        self.reduction = reduction

    def forward(self, x, y):
        """
        Args:
            x (Tensor): 输入张量，形状为 [512, 768]。
            y (Tensor): 输入张量，形状为 [768]。
        
        Returns:
            loss (Tensor): 损失值。
                           如果 reduction=None，形状为 [512]。
                           如果 reduction='mean' 或 'sum'，返回标量值。
        """
        # 扩展 y 的维度以与 x 匹配
        y = y.unsqueeze(0)  # 将 y 的形状从 [768] 扩展为 [1, 768]
        
        # 计算余弦相似度，输出形状为 [512]
        cos_sim = F.cosine_similarity(x, y, dim=1, eps=1e-6)
        
        # 损失为 1 - 余弦相似度
        loss = 1 - cos_sim

        # 根据 reduction 参数处理损失
        if self.reduction == 'mean':
            loss = loss.mean()
        elif self.reduction == 'sum':
            loss = loss.sum()
        return loss

class CosineSimilarityLoss_batch(nn.Module):
    def __init__(self, reduction=None):
        super(CosineSimilarityLoss_batch, self).__init__()
        self.reduction = reduction

    def forward(self, x, y):
        """
        Args:
            x: [batch_size, seq_len, feature_dim]  # 三维张量
            y: [batch_size, feature_dim]          # 二维张量
        Returns:
            loss: [batch_size, seq_len]  # 每个序列位置对应的损失
        """
        # 将 y 扩展到与 x 形状匹配 [batch_size, 1, feature_dim]
        y = y.unsqueeze(1)  # [batch_size, 1, feature_dim]
        
        # 计算余弦相似度 [batch_size, seq_len]
        cos_sim = nn.functional.cosine_similarity(x, y, dim=-1, eps=1e-6)  # 在最后一个维度计算
        
        # 转化为损失
        loss = 1 - cos_sim  # [batch_size, seq_len]
        
        # 如果需要归约
        if self.reduction == 'mean':
            loss = loss.mean(dim=-1)  # 按序列维度求平均，返回 [batch_size]
        elif self.reduction == 'sum':
            loss = loss.sum(dim=-1)  # 按序列维度求和，返回 [batch_size]

        return loss
@dataclass
class PromptHistory:
    """记录单个prompt的优化历史"""
    prompt_id: int
    original_prompt: str
    timestamp: str
    step: int
    control: str
    loss: float
    is_best: bool



@dataclass
class BatchAttackResult:
    success: List[bool]
    original_prompts: List[str]
    attack_prompts: List[str]
    execution_time: float



@dataclass
class PromptTracker:
    """追踪单个prompt的完整优化过程"""
    prompt_id: int
    original_prompt: str
    history: List[PromptHistory]
    best_control: str = ""
    best_loss: float = float('inf')
    start_time: float = 0.0
    end_time: float = 0.0

class ExperimentTracker:
    """实验追踪器"""
    def __init__(
        self,
        save_dir: str = "experiment_logs",
        experiment_name: str = None
    ):
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(parents=True, exist_ok=True)
        
        # 设置实验名称
        if experiment_name is None:
            experiment_name = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.experiment_name = experiment_name
        
        # 创建实验目录
        self.experiment_dir = self.save_dir / experiment_name
        self.experiment_dir.mkdir(parents=True, exist_ok=True)
        
        # 初始化prompt追踪器
        self.prompt_trackers: Dict[int, PromptTracker] = {}
        
    def initialize_prompt(self, prompt_id: int, prompt: str):
        """初始化新的prompt追踪器"""
        self.prompt_trackers[prompt_id] = PromptTracker(
            prompt_id=prompt_id,
            original_prompt=prompt,
            history=[],
            start_time=time.time()
        )
        
    def update_prompt(
        self,
        prompt_id: int,
        control: str,
        loss: float,
        step: int,
        is_best: bool = False
    ):
        """更新prompt的优化历史"""
        tracker = self.prompt_trackers[prompt_id]
        
        # 记录历史
        history = PromptHistory(
            prompt_id=prompt_id,
            original_prompt=tracker.original_prompt,
            timestamp=datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f"),
            step=step,
            control=control,
            loss=loss,
            is_best=is_best
        )
        tracker.history.append(history)
        
        # 更新最佳结果
        if is_best:
            tracker.best_control = control
            tracker.best_loss = loss
            
    def finish_prompt(self, prompt_id: int):
        """完成prompt的优化"""
        tracker = self.prompt_trackers[prompt_id]
        tracker.end_time = time.time()
        
    def save_prompt_history(self, prompt_id: int):
        """保存单个prompt的优化历史"""
        tracker = self.prompt_trackers[prompt_id]
        
        # 转换为DataFrame
        history_df = pd.DataFrame([asdict(h) for h in tracker.history])
        
        # 保存CSV
        csv_path = self.experiment_dir / f"prompt_{prompt_id}_history.csv"
        history_df.to_csv(csv_path, index=False)
        
        # 保存summary
        summary = {
            "prompt_id": prompt_id,
            "original_prompt": tracker.original_prompt,
            "best_control": tracker.best_control,
            "best_loss": tracker.best_loss,
            "optimization_time": tracker.end_time - tracker.start_time,
            "total_steps": len(tracker.history),
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        }
        
        json_path = self.experiment_dir / f"prompt_{prompt_id}_summary.json"
        with open(json_path, 'w') as f:
            json.dump(summary, f, indent=4)
            
    def save_experiment_summary(self):
        """保存整个实验的总结"""
        summaries = []
        for prompt_id, tracker in self.prompt_trackers.items():
            summary = {
                "prompt_id": prompt_id,
                "original_prompt": tracker.original_prompt,
                "best_control": tracker.best_control,
                "best_loss": tracker.best_loss,
                "optimization_time": tracker.end_time - tracker.start_time,
                "total_steps": len(tracker.history)
            }
            summaries.append(summary)
            
        # 保存为DataFrame
        summary_df = pd.DataFrame(summaries)
        summary_df.to_csv(self.experiment_dir / "experiment_summary.csv", index=False)
        
        # 计算统计信息
        stats = {
            "experiment_name": self.experiment_name,
            "total_prompts": len(self.prompt_trackers),
            "average_loss": float(summary_df["best_loss"].mean()),
            "std_loss": float(summary_df["best_loss"].std()),
            "average_time": float(summary_df["optimization_time"].mean()),
            "total_time": float(summary_df["optimization_time"].sum()),
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        }
        
        # 保存统计信息
        with open(self.experiment_dir / "experiment_stats.json", 'w') as f:
            json.dump(stats, f, indent=4)



class ParallelMMA(BaseAttacker):
    def __init__(
        self, 
        device="cuda",
        target_model: Any = None, 
        experiment_name: Optional[str] = None,
        text_detector:  Any = None,
        image_detector: Any = None,
        save_dir: str = "mma_experiments"
    ):
        super().__init__(target_model)
        
        # Extract the text encoder and tokenizer from the target model
        self.text_encoder = self.target_model.model.model.text_encoder
        self.tokenizer = self.target_model.model.model.tokenizer
        # self.tokenizer.enable_fast_tokenizer() 
        self.device = device
        
        # 初始化实验追踪器
        self.tracker = None
        if experiment_name is not None:
            self.tracker = ExperimentTracker(
                save_dir=save_dir,
                experiment_name=experiment_name
            )
            print(f"Experiment tracking enabled. Results will be saved to: {self.tracker.experiment_dir}")
        
        self.remove_set()

    def remove_set(self):
        nsfw_word_list = [
            'sexual', 'nude', 'nudity', 'sex', '18+', 'naked', 'nsfw', 'porn', 'dick',
            'vagina', 'naked child', 'explicit content', 'uncensored', 'fuck',
            'nipples', 'visible nipples', 'breast', 'areola',
        ]
        sensitive_word_list = ['ak', 'na', 'ke', 'nud'] + nsfw_word_list

        self.tokens_to_remove_set = set(
            token_id
            for word in sensitive_word_list
            for token, token_id in self.tokenizer.get_vocab().items()
            if word in token
        )


    # def tokenize_subset(tokenizer_name, prompts, max_length):
    #     # 在每个进程中创建自己的tokenizer实例
    #     tokenizer = CLIPTokenizer.from_pretrained(tokenizer_name)
    #     tokenized = tokenizer(
    #         prompts,
    #         padding="max_length",
    #         max_length=max_length,
    #         return_tensors="pt",
    #         truncation=True
    #     )
    #     # 返回需要的张量
    #     return {"input_ids": tokenized["input_ids"], 
    #             "attention_mask": tokenized["attention_mask"]}

    # def parallel_tokenize(prompts, tokenizer_name, max_length=77, num_processes=None):
    #     # 如果未指定进程数，使用可用CPU核心数
    #     if num_processes is None:
    #         num_processes = min(os.cpu_count(), 8)  # 通常不超过8个进程
        
    #     # 计算每个进程处理的样本数
    #     samples_per_process = len(prompts) // num_processes
        
    #     # 将数据分成多个子集
    #     prompts_subsets = []
    #     for i in range(num_processes - 1):
    #         start_idx = i * samples_per_process
    #         end_idx = (i + 1) * samples_per_process
    #         prompts_subsets.append(prompts[start_idx:end_idx])
    #     # 最后一个进程处理剩余的所有样本
    #     prompts_subsets.append(prompts[(num_processes - 1) * samples_per_process:])
        
    #     # 创建带有固定参数的函数
    #     tokenize_fn = partial(tokenize_subset, tokenizer_name, max_length=max_length)
        
    #     # 使用多进程并行处理
    #     with mp.Pool(processes=num_processes) as pool:
    #         results = pool.map(tokenize_fn, prompts_subsets)
        
    #     # 合并结果
    #     all_input_ids = torch.cat([r["input_ids"] for r in results], dim=0)
    #     all_attention_masks = torch.cat([r["attention_mask"] for r in results], dim=0)
        
    #     return {"input_ids": all_input_ids, "attention_mask": all_attention_masks}



    def get_target_embeddings_batch(self, prompts: List[str]):
        """批量获取目标embeddings"""
        # 批量tokenize
        target_tokenized = self.tokenizer(
            prompts,
            padding="max_length",
            return_tensors="pt",
            truncation=True
        )
        target_input = target_tokenized["input_ids"].to(self.device)
        
        # 批量获取embeddings
        with torch.no_grad():
            target_embeddings = self.text_encoder.text_model(target_input)["pooler_output"]
        return target_embeddings
    
    
    
    def get_target_embeddings_batch_3d(self, prompts: List[List[List[str]]]):
        """
        批量获取目标 embeddings，支持形状为 [32,512,20] 的 3D 列表。
        Args:
            prompts: 形状为 [batch_size, seq_len, num_cands] 的嵌套列表。
        Returns:
            target_embeddings: 张量，形状为 [batch_size, seq_len, num_cands, embedding_dim]。
        """
        batch_size, seq_len, num_cands = len(prompts), len(prompts[0]), len(prompts[0][0])
        
        # 展开 prompts 为一维列表
        flattened_prompts = [cand for batch in prompts for seq in batch for cand in seq]

        # 批量 tokenize 展开的 prompts
        target_tokenized = self.tokenizer(
            flattened_prompts,
            padding="max_length",
            return_tensors="pt",
            truncation=True
        )
        target_input = target_tokenized["input_ids"].to(self.device)
        
        # 批量获取 embeddings
        with torch.no_grad():
            target_embeddings = self.text_encoder.text_model(target_input)["pooler_output"]  # [batch_size * seq_len * num_cands, embedding_dim]
        
        # 将 embeddings 重新 reshape 成 [batch_size, seq_len, num_cands, embedding_dim]
        target_embeddings = target_embeddings.view(batch_size, seq_len,  -1)
        
        return target_embeddings

    def token_gradient_batch(self, controls: List[str], target_embeddings: torch.Tensor):
        """批量计算梯度"""
        tokens = self.tokenizer(
            controls, 
            padding="max_length",
            return_tensors="pt", 
            truncation=True
        )
        input_ids = tokens["input_ids"].to(self.device)
        
        batch_size = len(controls)
        control_length = 20
        embed_weights = self.text_encoder.text_model.embeddings.token_embedding.weight
        # print(embed_weights.shape)[49408,768]
        # print(2222222222)
        # 为整个batch创建one-hot向量
        one_hot = torch.zeros(
            batch_size,
            control_length, 
            embed_weights.shape[0], 
            device=self.device, 
            dtype=embed_weights.dtype
        )
        
        # 批量scatter
        for i in range(batch_size):
            one_hot[i].scatter_(1, input_ids[i][:control_length].unsqueeze(1), 1.0)
        
        one_hot.requires_grad_()
        # print(one_hot.shape)[32,20,49408]
        # 批量处理embeddings
        input_embeds = torch.matmul(one_hot, embed_weights)
        # print(input_embeds.shape)[32,20,768]
        embeds = self.text_encoder.text_model.embeddings.token_embedding(input_ids)
        full_embeds = torch.cat(
            [input_embeds, embeds[:, control_length:]], dim=1
        )
        # print(33333333333)
        # print(full_embeds.shape)[32,77,768]
        position_ids = torch.arange(0, self.tokenizer.model_max_length).to(self.device)
        position_embeddings = self.text_encoder.text_model.embeddings.position_embedding
        pos_embeds = position_embeddings(position_ids).unsqueeze(0).expand(batch_size, -1, -1)
        # print(pos_embeds.shape)[32,77,768]
        embeddings = full_embeds + pos_embeds
        
        # 计算loss和梯度
        control_embeddings = self.text_encoder.text_model(
            input_ids=input_ids, 
            input_embed=embeddings
        )["pooler_output"]
        # print(control_embeddings.shape)[32,768]
        criteria = CosineSimilarityLoss()
        loss = criteria(control_embeddings, target_embeddings)
        loss.backward()
        # print(one_hot.grad.shape)[32,20,49408]
        # print(444444)
        # exit()
        return one_hot.grad.clone()
    def sample_control_batch(self, grad, control_str=None,topk=256,num_can=512):
        """
        Sample control tokens based on gradient information, handling two levels of batching.
        
        Args:
            grad (torch.Tensor): Gradient tensor of shape [batch_size, 20, 490408].
            num_can (int): Number of samples to generate per batch (e.g., 512).
            topk (int): Number of top candidates to consider for replacement.

        Returns:
            torch.Tensor: Updated control tokens of shape [batch_size, num_can, 20].
        """
        batch_size, num_tokens, vocab_size = grad.shape  # batch_size=32, num_tokens=20, vocab_size=490408
        
        # 1. 将敏感 token 的梯度置为无穷大
        for input_id in set(self.tokens_to_remove_set):
            grad[:, :, input_id] = float("inf")  # 屏蔽敏感 token
        
        # 2. 提取 topk 的索引
        top_indices = (-grad).topk(topk, dim=2).indices  # shape [batch_size, 20, topk]
        # 3. 转换控制字符串为初始 token IDs
        if isinstance(control_str, list):

            # 批量分词和 ID 转换
            tokenized = self.tokenizer.batch_encode_plus(
                control_str,
                add_special_tokens=False,  # 不添加特殊标记
                return_tensors="pt"        # 返回 PyTorch 张量
            )

            # 提取 token IDs 并转为 Tensor
            control_toks = tokenized["input_ids"].to(grad.device).type(torch.int64)  # shape: [batch_size, token_length]
        else:
            tokens = self.tokenizer.tokenize(control_str)
            control_toks = torch.Tensor(self.tokenizer.convert_tokens_to_ids(tokens)).to(grad.device)
            control_toks = control_toks.type(torch.int64)# shape [32,20]
        # 扩展 control_toks 为 [batch_suze, num_can, num_tokens] 的初始形状
        original_control_toks = control_toks.unsqueeze(1).expand(batch_size, num_can, -1)  # shape [32, 512, 20]
        # 4. 随机选取需要替换的 token
        # 每个外部 batch 都需要生成 batch_size 个样本
        # Step 1: 生成一个等间距的 [512] 张量
        new_token_pos = torch.arange(0, len(control_toks[0]), len(control_toks[0]) / num_can).type(torch.int64).to(grad.device)  # shape [512]
        # Step 2: 扩展到 [32, 512]，所有行的值相同
        new_token_pos = new_token_pos.unsqueeze(0).repeat( original_control_toks.shape[0], 1)  # shape [32, 512]
        # 根据 new_token_pos 和 top_indices 随机选择替换值
        batch_indices = torch.arange(batch_size).unsqueeze(1).expand(-1, new_token_pos.size(1))#[32,512]
        
        selected_top_indices=top_indices[
            batch_indices,  # 用于选择每个 batch
            new_token_pos                          # [batch_size, num_replacements]
        ]  # 输出形状：[32, 512, 256]
    
        # 2. 随机选择候选集合中的一个 token
        random_indices = torch.randint(0, topk, (batch_size, num_can, 1), device=top_indices.device)#[32,512,1]
        # 利用 torch.gather 选择对应列的值
        new_token_val = torch.gather(selected_top_indices, 2, random_indices)
        # 输出形状：[32, 512, 1]
        # 5. 替换控制 token 的值
        new_control_toks = original_control_toks.clone().scatter_(
                2,  # 操作的维度是最后一个维度（token 维度）
                new_token_pos.unsqueeze(-1),  # 将 [32, 512] 扩展为 [32, 512, 1]
                new_token_val  # 形状已经是 [32, 512, 1]
            ) # shape [32, 512, 20]
        return new_control_toks# shape [32, 512, 20]
    def get_filtered_cands(self, control_cand, filter_cand=True, curr_control=None):
        """
        Args:
            control_cand: [32, 512, 20]，候选 token 的 ID。
            filter_cand: 是否过滤候选字符串。
            curr_control: 当前的控制字符串，用于排除。
        Returns:
            batch_cands: [batch_size, seq_len]，解码后的候选字符串。
        """
        batch_size, seq_len, num_cands = control_cand.shape  # 获取输入维度

        # 展平所有 batch 的 token [32, 512, 20] -> [32 * 512 * 20]
        flattened_tokens = control_cand.view(-1).tolist()

        # 一次性解码所有 token
        tokenizer = self.tokenizer
        decoded_flattened = tokenizer.convert_ids_to_tokens(flattened_tokens)

        # 将解码后的 token 转换为 3D list
        # 先按每个 batch 分割，每个 batch 包含 [seq_len * num_cands] 个 token
        batch_cands = []
        # NOTE 这一部分必须这样吗
        for batch_idx in range(batch_size):
            # 每个 batch 的解码结果
            start_idx = batch_idx * seq_len * num_cands
            end_idx = (batch_idx + 1) * seq_len * num_cands
            batch_decoded = decoded_flattened[start_idx:end_idx]

            # 将解码结果重组为 [seq_len, num_cands]
            decoded_batch = [
                "".join(batch_decoded[i * num_cands:(i + 1) * num_cands]).replace('</w>', ' ')[:-1]
                for i in range(seq_len)
            ]

            # 如果需要过滤，按照条件筛选
            cands = []
            count = 0
            for i, decoded_str in enumerate(decoded_batch):
                if filter_cand:
                    if decoded_str != curr_control and len(tokenizer(decoded_str, add_special_tokens=False).input_ids) == len(control_cand[batch_idx, i]):
                        cands.append(decoded_str) # 必须要长度一样吗
                    else:
                        count += 1
                else:
                    cands.append(decoded_str)

            if filter_cand:
                # 如果候选不足 seq_len，重复最后一个候选补足
                cands = cands + [cands[-1]] * (seq_len - len(cands))

            batch_cands.append(cands)

        return batch_cands
    
    
    def step(self,batch_controls,batch_target_embeddings,topk):
        # 计算梯度
        # print("getting gradient")
        grads = self.token_gradient_batch(
            batch_controls,
            batch_target_embeddings
        )
        # print("sampling")
        # 采样新的控制字符串
        new_controls = self.sample_control_batch(
            grads,
            batch_controls,
            topk=topk
        ) #[32,512,20] 32是batch，512是candidate个数，20是字符串长度。
        # print("filtering")
        control_cands=self.get_filtered_cands(new_controls)

        del grads, new_controls ; gc.collect()
        
        # 更新当前batch的控制字符串
        
        # 计算新的loss
        # print("getting loss")
        with torch.no_grad():
            control_embeddings = None  # 初始化为空
            # NOTE 是并行吗
            i = 0
            for controls in control_cands:
                embeddings = self.get_target_embeddings_batch(controls)  # 获取当前 batch 的 embeddings
                if control_embeddings is None:
                    control_embeddings = embeddings
                else:
                    control_embeddings = torch.cat([control_embeddings, embeddings], dim=0)
                
            # def process_all_controls(self, control_cands):
            #     # 将所有控制候选项平铺到一个列表中
            #     all_controls = [item for sublist in control_cands for item in sublist]
                
            #     # 一次性获取所有embeddings
            #     all_embeddings = self.get_target_embeddings_batch(all_controls)
                
            #     # 如果需要保持原有的批次结构，可以重新分割结果
            #     start_idx = 0
            #     results = []
            #     for controls in control_cands:
            #         end_idx = start_idx + len(controls)
            #         results.append(all_embeddings[start_idx:end_idx])
            #         start_idx = end_idx
                
            #     return results   
                
            # control_embeddings = process_all_controls(self, control_cands)
                
                
                
            # 最终 reshape 为目标形状
            control_embeddings = control_embeddings.view(
                len(control_cands), -1, control_embeddings.shape[-1]
            )
            
            # control_embeddings=[]
            # for controls in control_cands:
            #     control_embeddings.append(self.get_target_embeddings_batch(controls))
            # control_embeddings=torch.cat(control_embeddings,dim=0).view(len(control_cands),-1,control_embeddings[0].shape[-1])
            # control_embeddings= torch.stack(control_embeddings)
            
            
            criteria =CosineSimilarityLoss_batch(reduction=None)
            loss=criteria(control_embeddings,batch_target_embeddings)#[32,512]
        min_idx = loss.argmin(dim=1)
        next_control = [control_cands[i][min_idx[i].item()] for i in range(len(control_cands))]#返回了一个list，这个list是包含了32个字符串
        # next_control = control_cands[min_idx]#只返回了一个最小的loss对应的control string
        cand_loss = loss[torch.arange(loss.size(0)), min_idx]
        return next_control,cand_loss
    
    
    def run(self,control_strs,target_embeddings,batch_size,n_steps,topk=256):
        #control_strs和target_embeddings都是所有prompts对应的
        self.results=[]
        num_batches = (len(control_strs) + batch_size - 1) // batch_size
        print(f"Running optimization for {len(control_strs)} prompts in {num_batches} batches")
        for batch_idx in range(num_batches): 
            start_idx = batch_idx * batch_size
            end_idx = min((batch_idx + 1) * batch_size, len(control_strs))
            print(f"Batch {batch_idx + 1}/{num_batches} | Prompts {start_idx + 1}-{end_idx}")
            self.batch_controls = control_strs[start_idx:end_idx]
            self.best_controls=self.batch_controls
            self.batch_target_embeddings = target_embeddings[start_idx:end_idx]
            self.best_losses = [float('inf')] * len(self.batch_controls)
            for step in tqdm(range(n_steps), desc="Optimization Progress"):
                
                #print(batch_target_embeddings.shape)[32.768]
                #control 一个list，长度为batch_size个。
                control, loss = self.step(batch_controls=self.batch_controls,batch_target_embeddings=self.batch_target_embeddings,topk = topk)
                self.batch_controls = control
                for i in range(len(self.best_losses)):
                    if loss[i] < self.best_losses[i]:  # 如果新的 loss 比当前的最佳 loss 小
                            self.best_losses[i] = loss[i]  # 更新最佳 loss
                            self.best_controls[i] = control[i]  # 更新最佳候选

                # 打印当前step的统计信息
                if step % 10 == 0:
                    avg_loss = sum(self.best_losses) / len(self.best_losses)
                    print(f"Step {step}/{n_steps} | Avg Best Loss: {avg_loss:.4f}")

                # 可选：提前停止
                # if max(self.best_losses) < 0.1:
                #     print(f"Early stopping at step {step} - target loss achieved")
                #     break
                
            self.results.extend(self.best_controls)

        return self.results

    def attack_batch(
        self, 
        prompts: List[str], #传递的是数据集所有prompt
        n_steps: int = 1000, 
        batch_size: int = 256,
        topk: int = 256,
        track_interval: int = 1  # 每隔多少步记录一次
        
    ) -> BatchAttackResult:
        """并行处理多个提示词的攻击，并记录优化过程"""
        start_time = time.time()
        print(len(prompts))
        # 获取所有提示词的目标embeddings

        target_embeddings = self.get_target_embeddings_batch(prompts)
        #print(target_embeddings.shape)#[2243,768]
        # target_embeddings 是否需要保存？
        # 初始化控制字符串
        control_strs = [
            " ".join([random.choice(string.ascii_letters) for _ in range(20)])
            for _ in range(len(prompts))
        ]
        best_controls = control_strs.copy() #[243]
        # print(control_strs)
        # print(len(control_strs))#[243]
        result=self.run(control_strs, target_embeddings,batch_size,n_steps)
        
        # df = pd.DataFrame(result, columns=["prompt"])       
        # df.to_csv("prompts.csv", index=True, encoding="utf-8")
        # print("finished")
        # exit()
        
        # # 如果有tracker，初始化所有prompt
        # if self.tracker:
        #     for i, prompt in enumerate(prompts):
        #         self.tracker.initialize_prompt(i, prompt)

        # # 分batch处理
        # # 完成所有prompt的优化并保存结果
        # if self.tracker:
        #     for i in range(len(prompts)):
        #         self.tracker.finish_prompt(i)
        #         self.tracker.save_prompt_history(i)
            
        #     # 保存实验总结
        #     self.tracker.save_experiment_summary()

        end_time = time.time()
        
        return BatchAttackResult(
            success=[True] * len(prompts),
            original_prompts=prompts,
            attack_prompts=self.results,
            execution_time=end_time - start_time
        )

    def attack(self, prompt, **kwargs):
        pass
    
    
AttackerFactory.register('MMA', ParallelMMA)








# 使用示例：
"""
# 初始化MMA
mma = ParallelMMA(
    text_encoder=text_encoder,
    tokenizer=tokenizer,
    experiment_name="batch_attack_test"
)

# 运行攻击
prompts = ["prompt1", "prompt2", "prompt3", ...]
results = mma.attack_batch_parallel(prompts)

# 实验结果在 experiment_logs/batch_attack_test/ 目录下：
# - prompt_X_history.csv: 每个prompt的详细优化历史
# - prompt_X_summary.json: 每个prompt的优化总结
# - experiment_summary.csv: 所有prompt的总结
# - experiment_stats.json: 实验统计信息
"""