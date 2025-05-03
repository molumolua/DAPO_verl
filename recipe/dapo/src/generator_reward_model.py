# Copyright 2024 PRIME team and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import copy
import logging
import os
import warnings

import torch
import torch.distributed
from torch.distributed.device_mesh import init_device_mesh
import verl.utils.torch_functional as verl_F
from omegaconf import DictConfig, open_dict
from verl import DataProto
from verl.single_controller.base import Worker
from verl.single_controller.base.decorator import register, Dispatch
from verl.utils import hf_tokenizer
from verl.utils.debug import log_gpu_memory_usage
from verl.utils.fs import copy_local_path_from_hdfs
from verl.utils.fsdp_utils import get_fsdp_wrap_policy, init_fn, get_init_weight_context_manager
from verl.utils.fsdp_utils import offload_fsdp_optimizer, offload_fsdp_model_to_cpu, load_fsdp_optimizer, \
    load_fsdp_model_to_gpu
from verl.utils.seqlen_balancing import rearrange_micro_batches, get_reverse_idx
from verl.utils.import_utils import import_external_libs
from verl.utils.model import compute_position_id_with_mask
from verl.utils.flops_counter import FlopsCounter
from verl.utils.checkpoint.fsdp_checkpoint_manager import FSDPCheckpointManager
from verl.workers.sharding_manager.fsdp_ulysses import FSDPUlyssesShardingManager

from codetiming import Timer
from verl.workers.fsdp_workers import create_device_mesh, get_sharding_strategy
import re
from verl.workers.fsdp_workers import RewardModelWorker
logger = logging.getLogger(__file__)
logger.setLevel(os.getenv('VERL_PPO_LOGGING_LEVEL', 'WARN'))


class GeneratorRewardModelWorker(RewardModelWorker):

    def __init__(self, config):
        super().__init__(config=config)
        self._boxed_pattern = re.compile(r"boxed\s*{\s*([0-9]+(?:\.[0-9]+)?)\s*}", flags=re.I)
        self.rm_template = (
            "### 问题:\n{instruction}\n\n"
            "### 回答:\n{response}\n\n"
            "### 请给出 0~10 的分数，用 boxed{{score}} 的格式回复:"
        )

        # 生成参数（确定性、最短输出）
        self.rm_gen_cfg = dict(
            max_new_tokens=8,
            do_sample=False,
            temperature=0.0,
            eos_token_id=self.tokenizer.eos_token_id,
            pad_token_id=self.tokenizer.eos_token_id
        )
    def _build_rm_batch(self, micro_batch):
        """
        把 actor 的 prompt + response 解码成文本，
        再按照 rm_template 重新编码成 reward_model 的输入。
        """
        # 1. 从 micro_batch 解码出原始文本
        input_ids = micro_batch["input_ids"]
        texts = self.tokenizer.batch_decode(
            input_ids, skip_special_tokens=True, clean_up_tokenization_spaces=True
        )
        processed_texts = []
        for txt in texts:
            if "<|assistant|>" in txt:
                instr, resp = txt.split("<|assistant|>", 1)
            else:                      # 最坏情况全部视为回答
                instr, resp = "", txt
            processed_texts.append(
                self.rm_template.format(instruction=instr.strip(),
                                        response=resp.strip())
            )

        # 3. 用 reward_model 的 tokenizer 再编码
        rm_enc = self.tokenizer(
            processed_texts,
            padding=True,
            truncation=True,
            max_length=self.config.rm_max_length,
            return_tensors="pt"
        ).to(torch.cuda.current_device())

        # micro_batch 里后续仍用得到 batch_size
        rm_enc["batch_size"] = len(texts)
        return rm_enc

    def _forward_micro_batch(self, micro_batch):
        """
        1. 把 actor 的输入/输出 → 评分 prompt（_build_rm_batch）
        2. 调 reward_model.generate 得到带 boxed{score} 的文本
        3. regex 提取数字 → tensor 形状 (bs,)
        """
        # 1) 构造 reward_model 输入
        rm_mb = self._build_rm_batch(micro_batch)
        input_ids      = rm_mb["input_ids"]
        attention_mask = rm_mb["attention_mask"]

        # 2) 生成
        with torch.no_grad():
            gen_out = self.reward_module.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                **self.rm_gen_cfg
            )                              # (bs, seq_len + new)

        # 3) 解码
        texts = self.tokenizer.batch_decode(
            gen_out, skip_special_tokens=True, clean_up_tokenization_spaces=True
        )

        # 4) regex 抽取 boxed{score}
        scores = []
        for txt in texts:
            m = self._boxed_pattern.search(txt)
            if m:
                score_val = float(m.group(1))
            else:                # 若解析失败，可给默认分或抛异常
                score_val = 0.0
                # 或者 raise ValueError(f"未找到 boxed{{}}: {txt}")
            scores.append(score_val)

        rm_score = torch.tensor(scores, dtype=torch.float32, device=input_ids.device)  # (bs,)

        return rm_score

    # @register(dispatch_mode=Dispatch.DP_COMPUTE_PROTO)
    # def compute_rm_score(self, data: DataProto):
    #     data = data.to('cuda')
    #     self.reward_module.eval()
    #     self.ref_module.eval()
    #     micro_batch_size = data.meta_info['micro_batch_size']
    #     select_keys = ['responses', 'input_ids', 'attention_mask', 'position_ids', 'acc']
    #     batch = data.select(batch_keys=select_keys).batch
    #     use_dynamic_bsz = data.meta_info['use_dynamic_bsz']
    #     prompt_length = data.batch['input_ids'].shape[-1] - data.batch['responses'].shape[-1]

    #     if self._is_offload_param:
    #         load_fsdp_model_to_gpu(self.reward_module)
    #         load_fsdp_model_to_gpu(self.ref_module)
    #     micro_batch_size = self.config.micro_batch_size_per_gpu
    #     data.meta_info['micro_batch_size'] = micro_batch_size
    #     data.meta_info['max_token_len'] = self.config.forward_max_token_len_per_gpu
    #     data.meta_info['use_dynamic_bsz'] = self.config.use_dynamic_bsz
    #     # perform forward computation
    #     rm_scores_lst = []
    #     with self.ulysses_sharding_manager:
    #         data = self.ulysses_sharding_manager.preprocess_data(data=data)
    #         if use_dynamic_bsz:
    #             # split using dynamic bsz
    #             max_token_len = data.meta_info['max_token_len'] * self.ulysses_sequence_parallel_size
    #             micro_batches, indices = rearrange_micro_batches(batch=batch, max_token_len=max_token_len)
    #         else:
    #             micro_batches = batch.split(micro_batch_size)

    #         for micro_batch in micro_batches:
    #             with torch.no_grad():
    #                 rm_score = self._forward_micro_batch(micro_batch, prompt_length)
    #             rm_scores_lst.append(rm_score)
    #         # 这里让score和DAPO的compute_score 返回值一致
    #         rm_scores = torch.concat(rm_scores_lst, dim=0)
    #         token_level_scores = self._expand_to_token_level(data, rm_scores)
    #         # Note that this is only the scores, may not be the final rewards used to train RL
    #         output = DataProto.from_dict(tensors={'rm_scores': token_level_scores})
    #         output = self.ulysses_sharding_manager.postprocess_data(data=output)

    #     self.reward_module._handle.reshard(True)

    #     output = output.to('cpu')
    #     return output

    @register(dispatch_mode=Dispatch.DP_COMPUTE_PROTO)
    def compute_rm_score(self, data: DataProto):
        import itertools
        from verl.utils.seqlen_balancing import rearrange_micro_batches, get_reverse_idx
        # Support all hardwares
        data = data.to(torch.cuda.current_device())
        self.reward_module.eval()
        if self._do_switch_chat_template:
            rm_data = self._switch_chat_template(data)
        else:
            rm_input_ids = data.batch['input_ids']
            rm_attention_mask = data.batch['attention_mask']
            rm_position_ids = data.batch['position_ids']
            rm_inputs = {
                'input_ids': rm_input_ids,
                'attention_mask': rm_attention_mask,
                'position_ids': rm_position_ids
            }
            rm_data = DataProto.from_dict(rm_inputs)

        # Support all hardwares
        rm_data.batch = rm_data.batch.to(torch.cuda.current_device())

        # perform forward computation
        with self.ulysses_sharding_manager:
            rm_data = self.ulysses_sharding_manager.preprocess_data(data=rm_data)
            data = self.ulysses_sharding_manager.preprocess_data(data=data)

            use_dynamic_bsz = self.config.use_dynamic_bsz
            if use_dynamic_bsz:
                max_token_len = self.config.forward_max_token_len_per_gpu * self.ulysses_sequence_parallel_size
                micro_batches, indices = rearrange_micro_batches(batch=rm_data.batch, max_token_len=max_token_len)
            else:
                micro_batches = rm_data.batch.split(self.config.micro_batch_size_per_gpu)
            output = []
            for micro_batch in micro_batches:
                rm_score = self._forward_micro_batch(micro_batch)
                output.append(rm_score)
            scores = torch.cat(output, dim=0)  # (batch_size)

            if use_dynamic_bsz:
                indices = list(itertools.chain.from_iterable(indices))
                assert len(indices) == scores.size(0), f"{len(indices)} vs. {scores.size()}"
                revert_indices = torch.tensor(get_reverse_idx(indices), dtype=torch.long)
                scores = scores[revert_indices]

            token_level_scores = self._expand_to_token_level(data, scores)
            # Note that this is only the scores, may not be the final rewards used to train RL
            

        # https://pytorch.org/docs/stable/notes/fsdp.html#fsdp-notes
        # unshard the root FSDP module
        self.reward_module._handle.reshard(True)

        return {
                "reward_tensor": token_level_scores,
                "reward_extra_info": reward_extra_info,
        }