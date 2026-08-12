import torch
import torch.nn.functional as nnf
from torch import nn
from transformers import AutoModelForCausalLM
from brickanything_train.miche.encode import load_model
from brickanything_train.models.shape_opt import ShapeOPTConfig


def disable_dropout(model: torch.nn.Module):
    """Disable dropout in a model (following the reference DPO implementation)."""
    for module in model.modules():
        if isinstance(module, torch.nn.Dropout):
            module.p = 0


class SingleGPT(nn.Module):
    def __init__(self, args):
        super().__init__()

        self.args = args
        self.point_encoder = load_model()
        self.cond_length = 257
        self.cond_dim = 768

        self.n_discrete_size = args.n_discrete_size
        self.pad_id = -1

        self.max_length = int(args.max_seq_len + 3 + self.cond_length)

        # Root still uses discrete xyz; remaining slots are tree connectivity tokens.
        vocab_size = (
            self.n_discrete_size   # root x,y,z
            + args.hw_vocab_size   # h/w
            + args.f_vocab_size    # f
            + args.m_vocab_size    # m
            + 4                    # BOS, EOS, PAD, EOP
        )

        print(f"vocab_size: {vocab_size}")
        
        self.config = ShapeOPTConfig.from_pretrained(
            args.llm,
            n_positions=self.max_length,
            max_position_embeddings=self.max_length,
            vocab_size = vocab_size,
        )

        self.bos_token_id = 0
        self.eos_token_id = 1
        self.pad_token_id = 2

        self.config.bos_token_id = self.bos_token_id
        self.config.eos_token_id = self.eos_token_id
        self.config.pad_token_id = self.pad_token_id
        
        self.config.cond_length = self.cond_length
        self.config.word_embed_proj_dim = self.config.hidden_size

        self.transformer = AutoModelForCausalLM.from_config(
            config=self.config, 
            attn_implementation="flash_attention_2",
            torch_dtype=torch.float32
        )

        self.cond_head_proj = nn.Linear(self.cond_dim, self.config.word_embed_proj_dim)
        self.cond_proj = nn.Linear(self.cond_dim * 2, self.config.word_embed_proj_dim)

        self.train()

    def train(self, mode: bool = True):
        super().train(mode)
        if hasattr(self,"point_encoder"):
            self.point_encoder.eval()
            for param in self.point_encoder.parameters():
                param.requires_grad = False

    def forward(self, data_dict: dict, is_eval: bool = False) -> dict:
        return self.train_one_step(data_dict)

    def pad_id_and_attn(self, input_ids, attention_mask, face_ids = None): # same
        # Avoid in-place mutation of caller tensors (used multiple times in DPO loop).
        input_ids = input_ids.clone()
        attention_mask = attention_mask.clone()
        # reserve one space for `bos`, the pad_id will be replaced to `bos`
        place_holder = torch.ones_like(input_ids[:, [0]])   # batch x 1
        # prepare input_ids and attention_mask for transformers
        input_ids[attention_mask.bool()] += 3 # 0 - num_tokens to 3 - num_tokens + 3, total: 0 - num_tokens + 3, num: numtokens + 4
        input_ids[~attention_mask.bool()] = self.pad_token_id # in transformers pad token id is only used for init nn.embedding which we won't use
        input_ids = torch.cat(
            (place_holder * self.bos_token_id, input_ids, place_holder * self.pad_token_id),
            dim=1
        )
        input_ids[torch.arange(0, input_ids.shape[0]), attention_mask.sum(dim=1).long()+1] = self.eos_token_id


        attention_mask = torch.cat(
            (place_holder, place_holder, attention_mask, ),
            dim=1
        )
        # length
        return input_ids, attention_mask

    def process_point_feature(self, point_feature):
        batch_size = point_feature.shape[0]
        encode_feature = torch.zeros(batch_size, self.cond_length, self.config.word_embed_proj_dim,
                                    device=self.cond_head_proj.weight.device, dtype=self.cond_head_proj.weight.dtype)
        encode_feature[:, 0] = self.cond_head_proj(point_feature[:, 0])
        shape_latents = self.point_encoder.to_shape_latents(point_feature[:, 1:])
        encode_feature[:, 1:] = self.cond_proj(torch.cat([point_feature[:, 1:], shape_latents], dim=-1))

        return encode_feature

    def _build_model_inputs(self, pc_normal: torch.Tensor, sequence: torch.Tensor):
        point_feature = self.point_encoder.encode_latents(pc_normal)
        attention_mask = sequence != self.pad_id
        sequence_max_length = attention_mask.sum(dim=1).max()
        sequence = sequence[:, :sequence_max_length]
        attention_mask = attention_mask[:, :sequence_max_length]
        input_ids, attention_mask = self.pad_id_and_attn(sequence, attention_mask)

        pad_attention_mask = torch.ones(
            (attention_mask.shape[0], self.cond_length),
            device=attention_mask.device,
            dtype=attention_mask.dtype,
        )
        attention_mask = torch.concatenate((pad_attention_mask, attention_mask), dim=1)
        processed_point_feature = self.process_point_feature(point_feature=point_feature)
        return processed_point_feature, input_ids, attention_mask

    def train_one_step(self, data_dict: dict) -> dict:
        assert "sequence" in data_dict
        processed_point_feature, input_ids, attention_mask = self._build_model_inputs(
            pc_normal=data_dict["pc_normal"],
            sequence=data_dict["sequence"],
        )

        output = self.transformer(
            inputs_embeds = processed_point_feature,
            input_ids=input_ids,
            attention_mask=attention_mask,
            use_cache=False,
        )
        # compute loss with shift token right
        logit = output.logits[:, self.cond_length-1:-1]  # batch x ntoken x vocab
        label = input_ids[:, 0:]  # batch x ntoken
        #print(f"label: {label}")
        masks = attention_mask[:, self.cond_length-1:-1]  # batch x ntoken
        # also predict bos token
        loss_per_token = nnf.cross_entropy(
            logit.permute(0, 2, 1),  # batch x vocab x ntoken
            label,
            reduction='none'
        )  # batch x ntoken
        final_loss = torch.sum(loss_per_token * masks) / (torch.sum(masks) + 1e-8)
        #print(f"final_loss: {final_loss}")
        data_dict['loss'] = final_loss

        return data_dict

    def sequence_logprobs(self, pc_normal: torch.Tensor, sequence: torch.Tensor):
        processed_point_feature, input_ids, attention_mask = self._build_model_inputs(
            pc_normal=pc_normal,
            sequence=sequence,
        )
        output = self.transformer(
            inputs_embeds=processed_point_feature,
            input_ids=input_ids,
            attention_mask=attention_mask,
            use_cache=False,
        )
        logits = output.logits[:, self.cond_length - 1 : -1]
        labels = input_ids[:, 0:]
        masks = attention_mask[:, self.cond_length - 1 : -1].float()

        log_probs = torch.log_softmax(logits, dim=-1)
        token_logps = torch.gather(log_probs, dim=-1, index=labels.unsqueeze(-1)).squeeze(-1)
        token_logps = token_logps * masks
        seq_logps = token_logps.sum(dim=1)
        return seq_logps, token_logps, masks

