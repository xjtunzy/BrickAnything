"""
Shape OPT for training; aligned with BrickAnything.models.shape_opt_dynamic (DynamicCache / new OPT decoder API).
"""

from transformers import AutoModelForCausalLM, AutoConfig, OPTConfig
from transformers.models.opt.modeling_opt import OPTForCausalLM, OPTModel, OPTDecoder, OPTLearnedPositionalEmbedding, OPTDecoderLayer
from typing import List, Optional, Tuple, Union
from transformers.modeling_outputs import CausalLMOutputWithPast
import torch
from torch import nn
from torch.nn import CrossEntropyLoss
from transformers.modeling_outputs import BaseModelOutputWithPast
from transformers.cache_utils import Cache, DynamicCache


class ShapeOPTConfig(OPTConfig):
    model_type = "shape_opt"


class ShapeOPT(OPTForCausalLM):
    config_class = ShapeOPTConfig

    def __init__(self, config: ShapeOPTConfig):
        super(OPTForCausalLM, self).__init__(config)
        self.model = ShapeOPTModel(config)
        self.lm_head = nn.Linear(config.word_embed_proj_dim, config.vocab_size, bias=False)
        self.post_init()

    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        head_mask: Optional[torch.Tensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,  # 兼容旧 tuple，也可传 Cache
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        position_ids: Optional[torch.LongTensor] = None,
        cache_position: Optional[torch.Tensor] = None,
        **kwargs,
    ) -> Union[Tuple, CausalLMOutputWithPast]:
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        # decoder outputs: dec_features, past_key_values, hidden_states, attentions（见 BaseModelOutputWithPast）
        outputs = self.model.decoder(
            input_ids=input_ids,
            attention_mask=attention_mask,
            head_mask=head_mask,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            position_ids=position_ids,
            cache_position=cache_position,
            **kwargs,
        )

        logits = self.lm_head(outputs[0]).contiguous()
        loss = None
        if labels is not None:
            labels = labels.to(logits.device)
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            loss_fct = CrossEntropyLoss()
            loss = loss_fct(shift_logits.view(-1, self.config.vocab_size), shift_labels.view(-1))

        if not return_dict:
            output = (logits,) + outputs[1:]
            return (loss,) + output if loss is not None else output

        return CausalLMOutputWithPast(
            loss=loss,
            logits=logits,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )


class ShapeOPTModel(OPTModel):
    config_class = ShapeOPTConfig

    def __init__(self, config: ShapeOPTConfig):
        super(OPTModel, self).__init__(config)
        self.decoder = ShapeOPTDecoder(config)
        self.post_init()


class ShapeOPTDecoder(OPTDecoder):
    config_class = ShapeOPTConfig

    def __init__(self, config: ShapeOPTConfig):
        super(OPTDecoder, self).__init__(config)
        self.config = config
        self.dropout = config.dropout
        self.layerdrop = config.layerdrop
        self.padding_idx = config.pad_token_id
        self.vocab_size = config.vocab_size
        assert config.word_embed_proj_dim == config.hidden_size
        self.embed_tokens = nn.Embedding(config.vocab_size, config.word_embed_proj_dim, self.padding_idx)
        self.hidden_size = config.hidden_size
        self.word_embed_proj_dim = config.word_embed_proj_dim

        self.embed_positions = OPTLearnedPositionalEmbedding(config.max_position_embeddings, config.hidden_size)

        self.cond_length = config.cond_length
        self.cond_embed = nn.Embedding(2, config.word_embed_proj_dim)
        if config.do_layer_norm_before and not config._remove_final_layer_norm:
            self.final_layer_norm = nn.LayerNorm(
                config.hidden_size, elementwise_affine=config.layer_norm_elementwise_affine
            )
        else:
            self.final_layer_norm = None

        self.layers = nn.ModuleList([OPTDecoderLayer(config, layer_idx=i) for i in range(config.num_hidden_layers)])
        self._use_flash_attention_2 = config._attn_implementation == "flash_attention_2"

        self.gradient_checkpointing = False
        self.post_init()

    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        head_mask: Optional[torch.Tensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,  # 兼容旧 tuple，也可传 Cache
        inputs_embeds: Optional[torch.FloatTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        position_ids: Optional[torch.LongTensor] = None,
        cache_position: Optional[torch.Tensor] = None,
    ) -> Union[Tuple, BaseModelOutputWithPast]:
        """
        基于 transformers 4.51.x 的 OPTDecoder.forward 改写：
        - 上半部分保持 ShapeOPT 的 cond_embed 行为
        - 下半部分使用 DynamicCache + _update_causal_mask 等新逻辑
        """
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        use_cache = use_cache if use_cache is not None else self.config.use_cache
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        # 梯度检查点与 use_cache 不兼容（与官方 OPTDecoder 一致）
        if self.gradient_checkpointing and self.training and use_cache:
            from transformers.utils import logging

            logger = logging.get_logger(__name__)
            logger.warning_once(
                "`use_cache=True` is incompatible with gradient checkpointing. Setting `use_cache=False`."
            )
            use_cache = False

        if input_ids is None and inputs_embeds is None:
            raise ValueError("You have to specify either input_ids or inputs_embeds (or both).")

        # ========= inputs_embeds + cond_embed =========
        if input_ids is not None and inputs_embeds is not None:
            # Only use in Training
            embeds_from_id = self.embed_tokens(input_ids)

            inputs_embeds_length = inputs_embeds.shape[1]
            inputs_embeds = torch.cat([inputs_embeds, embeds_from_id], dim=1)
            total_length = inputs_embeds.shape[1]

            cond_embed_query = torch.ones(
                (inputs_embeds.shape[0], total_length),
                device=inputs_embeds.device,
                dtype=inputs_embeds.dtype,
            ).long()
            cond_embed_query[:, :inputs_embeds_length] = 0
            inputs_embeds = inputs_embeds + self.cond_embed(cond_embed_query)

        elif input_ids is not None:
            # 只给 ids：视为“普通 token 序列”，cond_embed 全 1
            assert not self.training
            input_ids = input_ids.view(-1, input_ids.shape[-1])
            inputs_embeds = self.embed_tokens(input_ids)

            cond_embed_query = torch.ones(
                (inputs_embeds.shape[0], inputs_embeds.shape[1]),
                device=inputs_embeds.device,
                dtype=inputs_embeds.dtype,
            ).long()
            inputs_embeds = inputs_embeds + self.cond_embed(cond_embed_query)

        else:
            # 只给 embeds：视为“条件部分”，cond_embed 全 0 或前缀 0
            assert not self.training

            total_length = inputs_embeds.shape[1]
            if total_length == self.cond_length:
                cond_embed_query = torch.zeros(
                    (inputs_embeds.shape[0], total_length),
                    device=inputs_embeds.device,
                    dtype=inputs_embeds.dtype,
                ).long()
                inputs_embeds = inputs_embeds + self.cond_embed(cond_embed_query)
            else:
                cond_embed_query = torch.ones(
                    (inputs_embeds.shape[0], total_length),
                    device=inputs_embeds.device,
                    dtype=inputs_embeds.dtype,
                ).long()
                cond_embed_query[:, : self.cond_length] = 0
                inputs_embeds = inputs_embeds + self.cond_embed(cond_embed_query)

        # ========= DynamicCache / 旧 tuple 兼容 =========
        return_legacy_cache = False
        if use_cache and not isinstance(past_key_values, Cache):
            return_legacy_cache = True
            past_key_values = DynamicCache.from_legacy_cache(past_key_values)
            if past_key_values is None:
                from transformers.utils import logging

                logger = logging.get_logger(__name__)
                logger.warning_once(
                    "Passing a tuple of `past_key_values` is deprecated and will be removed in a future version. "
                    "You should pass an instance of `DynamicCache` instead, e.g. "
                    "`past_key_values=DynamicCache.from_legacy_cache(past_key_values)`."
                )

        past_seen_tokens = past_key_values.get_seq_length() if past_key_values is not None else 0
        if cache_position is None:
            cache_position = torch.arange(
                past_seen_tokens, past_seen_tokens + inputs_embeds.shape[1], device=inputs_embeds.device
            )

        if attention_mask is None:
            seq_length = past_seen_tokens + inputs_embeds.shape[1]
            attention_mask = torch.ones(inputs_embeds.shape[0], seq_length, device=inputs_embeds.device)

        # ========= causal mask（用父类的 _update_causal_mask） =========
        causal_mask = self._update_causal_mask(
            attention_mask, inputs_embeds, cache_position, past_key_values, output_attentions
        )

        # ========= 位置编码 & project_in =========
        if position_ids is None:
            position_ids = torch.cumsum(attention_mask, dim=1)
            position_ids = (position_ids * attention_mask - 1).long()
            position_ids = position_ids[:, past_seen_tokens:]

        pos_embeds = self.embed_positions(attention_mask, past_seen_tokens, position_ids=position_ids)

        if getattr(self, "project_in", None) is not None:
            inputs_embeds = self.project_in(inputs_embeds)

        hidden_states = inputs_embeds + pos_embeds.to(inputs_embeds.device)

        # decoder layers
        all_hidden_states = () if output_hidden_states else None
        all_self_attns = () if output_attentions else None
        next_decoder_cache = None

        # head_mask 校验，保持与官方 OPTDecoder 一致
        for attn_mask, mask_name in zip([head_mask], ["head_mask"]):
            if attn_mask is not None:
                if attn_mask.size()[0] != len(self.layers):
                    raise ValueError(
                        f"The `{mask_name}` should be specified for {len(self.layers)} layers, but it is for"
                        f" {head_mask.size()[0]}."
                    )

        for idx, decoder_layer in enumerate(self.layers):
            if output_hidden_states:
                all_hidden_states += (hidden_states,)

            if self.training:
                dropout_probability = torch.rand([])
                if dropout_probability < self.layerdrop:
                    continue

            layer_head_mask = head_mask[idx] if head_mask is not None else None

            if self.gradient_checkpointing and self.training:
                layer_outputs = self._gradient_checkpointing_func(
                    decoder_layer.__call__,
                    hidden_states,
                    causal_mask,
                    layer_head_mask,
                    None,
                    output_attentions,
                    use_cache,
                    position_ids,
                    cache_position,
                )
            else:
                layer_outputs = decoder_layer(
                    hidden_states,
                    attention_mask=causal_mask,
                    position_ids=position_ids,
                    layer_head_mask=layer_head_mask,
                    past_key_value=past_key_values,
                    output_attentions=output_attentions,
                    use_cache=use_cache,
                    cache_position=cache_position,
                )

            hidden_states = layer_outputs[0]

            if use_cache:
                next_decoder_cache = layer_outputs[2 if output_attentions else 1]

            if output_attentions:
                all_self_attns += (layer_outputs[1],)

        if self.final_layer_norm is not None:
            hidden_states = self.final_layer_norm(hidden_states)

        if output_hidden_states:
            all_hidden_states += (hidden_states,)

        next_cache = next_decoder_cache if use_cache else None
        if return_legacy_cache and next_cache is not None:
            next_cache = next_cache.to_legacy_cache()

        if not return_dict:
            return tuple(v for v in [hidden_states, next_cache, all_hidden_states, all_self_attns] if v is not None)

        return BaseModelOutputWithPast(
            last_hidden_state=hidden_states,
            past_key_values=next_cache,
            hidden_states=all_hidden_states,
            attentions=all_self_attns,
        )


AutoConfig.register("shape_opt", ShapeOPTConfig)
AutoModelForCausalLM.register(ShapeOPTConfig, ShapeOPT)
