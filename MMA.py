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

from typing import Optional
import torch.nn.functional as F


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
    success: bool
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



class ParallelMMA:
    def __init__(
        self, 
        text_encoder, 
        tokenizer, 
        device="cuda",
        experiment_name: Optional[str] = None,
        save_dir: str = "mma_experiments"
    ):
        self.text_encoder = text_encoder
        self.tokenizer = tokenizer
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

        # 批量处理embeddings
        input_embeds = torch.matmul(one_hot, embed_weights)
        embeds = self.text_encoder.text_model.embeddings.token_embedding(input_ids)
        full_embeds = torch.cat(
            [input_embeds, embeds[:, control_length:]], dim=1
        )

        position_ids = torch.arange(0, self.tokenizer.model_max_length).to(self.device)
        position_embeddings = self.text_encoder.text_model.embeddings.position_embedding
        pos_embeds = position_embeddings(position_ids).unsqueeze(0).expand(batch_size, -1, -1)

        embeddings = full_embeds + pos_embeds
        
        # 计算loss和梯度
        control_embeddings = self.text_encoder.text_model(
            input_ids=input_ids, 
            input_embed=embeddings
        )["pooler_output"]

        criteria = CosineSimilarityLoss(reduction='none')
        loss = criteria(control_embeddings, target_embeddings).mean()
        loss.backward()

        return one_hot.grad.clone()

    def sample_control_batch(self, grads: torch.Tensor, controls: List[str], topk: int = 256) -> List[str]:
        """批量采样新的控制字符串"""
        batch_size = len(controls)
        
        # 处理禁用词
        for input_id in self.tokens_to_remove_set:
            grads[..., input_id] = np.inf
            
        # 获取topk索引
        top_indices = (-grads).topk(topk, dim=-1).indices
        
        # 批量处理token
        control_tokens = [
            self.tokenizer.tokenize(control)
            for control in controls
        ]
        
        control_ids = [
            torch.tensor(
                self.tokenizer.convert_tokens_to_ids(tokens),
                device=self.device
            )
            for tokens in control_tokens
        ]
        
        # 生成新的控制字符串
        new_controls = []
        for i in range(batch_size):
            # 随机选择要修改的位置
            pos = random.randint(0, len(control_tokens[i]) - 1)
            # 随机选择新的token
            new_token_idx = random.randint(0, topk - 1)
            new_token = top_indices[i, pos, new_token_idx].item()
            
            # 更新token
            temp_ids = control_ids[i].clone()
            temp_ids[pos] = new_token
            
            # 转换回字符串
            new_control = self.tokenizer.decode(temp_ids)
            new_controls.append(new_control)
            
        return new_controls



    def attack_batch_parallel(
        self, 
        prompts: List[str], 
        n_steps: int = 1000, 
        batch_size: int = 32,
        topk: int = 256,
        track_interval: int = 1  # 每隔多少步记录一次
    ) -> BatchAttackResult:
        """并行处理多个提示词的攻击，并记录优化过程"""
        start_time = time.time()
        
        # 获取所有提示词的目标embeddings
        target_embeddings = self.get_target_embeddings_batch(prompts)
        # target_embeddings 是否需要保存？
        # 初始化控制字符串
        control_strs = [
            " ".join([random.choice(string.ascii_letters) for _ in range(20)])
            for _ in range(len(prompts))
        ]
        best_controls = control_strs.copy()
        best_losses = [float('inf')] * len(prompts)

        # 如果有tracker，初始化所有prompt
        if self.tracker:
            for i, prompt in enumerate(prompts):
                self.tracker.initialize_prompt(i, prompt)

        # 分batch处理
        num_batches = (len(prompts) + batch_size - 1) // batch_size
        
        for step in tqdm(range(n_steps), desc="Optimization Progress"):
            for batch_idx in range(num_batches):
                start_idx = batch_idx * batch_size
                end_idx = min((batch_idx + 1) * batch_size, len(prompts))
                
                batch_controls = control_strs[start_idx:end_idx]
                batch_target_embeddings = target_embeddings[start_idx:end_idx]
                
                # 计算梯度
                grads = self.token_gradient_batch(
                    batch_controls,
                    batch_target_embeddings
                )
                
                # 采样新的控制字符串
                new_controls = self.sample_control_batch(
                    grads,
                    batch_controls,
                    topk=topk
                )
                
                # 更新当前batch的控制字符串
                control_strs[start_idx:end_idx] = new_controls
                
                # 计算新的loss
                with torch.no_grad():
                    new_embeddings = self.get_target_embeddings_batch(new_controls)
                    losses = F.cosine_similarity(
                        new_embeddings,
                        batch_target_embeddings
                    )
                    
                    # 更新最佳结果并记录
                    for i, loss in enumerate(losses):
                        idx = start_idx + i
                        loss_val = loss.item()
                        is_best = loss_val < best_losses[idx]
                        
                        # 记录优化历史（如果需要）
                        if self.tracker and step % track_interval == 0:
                            self.tracker.update_prompt(
                                prompt_id=idx,
                                control=new_controls[i],
                                loss=loss_val,
                                step=step,
                                is_best=is_best
                            )
                        
                        # 更新最佳结果
                        if is_best:
                            best_losses[idx] = loss_val
                            best_controls[idx] = new_controls[i]

            # 打印当前step的统计信息
            if step % 10 == 0:
                avg_loss = sum(best_losses) / len(best_losses)
                print(f"Step {step}/{n_steps} | Avg Best Loss: {avg_loss:.4f}")

            # 可选：提前停止
            if max(best_losses) < 0.1:
                print(f"Early stopping at step {step} - target loss achieved")
                break

        # 完成所有prompt的优化并保存结果
        if self.tracker:
            for i in range(len(prompts)):
                self.tracker.finish_prompt(i)
                self.tracker.save_prompt_history(i)
            
            # 保存实验总结
            self.tracker.save_experiment_summary()

        end_time = time.time()
        
        return BatchAttackResult(
            success=True,
            original_prompts=prompts,
            attack_prompts=best_controls,
            execution_time=end_time - start_time
        )

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