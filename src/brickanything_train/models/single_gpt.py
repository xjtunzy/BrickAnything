import torch
import torch.nn.functional as nnf
from torch import nn
from transformers import AutoModelForCausalLM
from brickanything_train.miche.encode import load_model
from brickanything_train.models.shape_opt import ShapeOPTConfig
from transformers import LogitsProcessor,LogitsProcessorList
from einops import repeat, reduce, rearrange, pack, unpack

brick_library = {
  "2": {
    "height": 2,
    "width": 4,
    "mass": 0.00216,
    "inventory": 100000,
    "partID": "3001.DAT"
  },
  "3": {
    "height": 2,
    "width": 6,
    "mass": 0.00323,
    "inventory": 100000,
    "partID": "2456.DAT"
  },
  "4": {
    "height": 1,
    "width": 8,
    "mass": 0.00303,
    "inventory": 100000,
    "partID": "3008.DAT"
  },
  "5": {
    "height": 1,
    "width": 4,
    "mass": 0.00157,
    "inventory": 100000,
    "partID": "3010.DAT"
  },
  "6": {
    "height": 1,
    "width": 6,
    "mass": 0.00228,
    "inventory": 100000,
    "partID": "3009.DAT"
  },
  "9": {
    "height": 1,
    "width": 2,
    "mass": 0.00081,
    "inventory": 100000,
    "partID": "3004.DAT"
  },
  "10": {
    "height": 1,
    "width": 1,
    "mass": 0.00043,
    "inventory": 100000,
    "partID": "3005.DAT"
  },
  "12": {
    "height": 2,
    "width": 2,
    "mass": 0.00115,
    "inventory": 100000,
    "partID": "3003.DAT"
  }
}

class Mask_invalid_token(LogitsProcessor):
    def __init__(self):
        super().__init__()
        self.n_discrete_size = 20
        self.offset = 3
        self.allowed_sizes = set()
        for item in brick_library.values():
            h, w = item['height'], item['width']
            self.allowed_sizes.add((h, w))
            self.allowed_sizes.add((w, h)) # 如果支持 90 度旋转

        # 提取所有可能的合法长度 (h 或 w)
        self.valid_lengths = set(h for h, w in self.allowed_sizes) | set(w for h, w in self.allowed_sizes)
    
    def __call__(self, input_ids, scores):
        batch_size = input_ids.shape[0]
        cur_len = input_ids.shape[1]
        device = input_ids.device
        #print(f"cur_len: {cur_len}")
        if cur_len == 0: return scores
        data_pos = (cur_len - 1) % 5
        for i in range(batch_size):
            
            seq = input_ids[i,1:] - self.offset
            #print(f"len seq: {len(seq)}\t data_pos: {data_pos}")
            #caculate voxel occupy
            occupancy = torch.zeros((self.n_discrete_size,self.n_discrete_size,self.n_discrete_size),device = device)
            num_complete = len(seq) // 5
            for b in range(num_complete):
                px, py, pz, px1, py1 = seq[b*5 : b*5+5]
                occupancy[px : px1 + 1, py : py1 + 1, pz] = 1
            if data_pos == 0:
                #predict x
                pass
            elif data_pos == 1:
                #predict y
                curr_x = seq[-1]
                yz_slice = occupancy[curr_x, :, :]
                for y_cand in range(self.n_discrete_size):
                    if yz_slice[y_cand,:].all():
                        scores[i,y_cand+self.offset]=float('-inf')
            elif data_pos == 2:
                #predict z
                curr_x = seq[-2]
                curr_y = seq[-1]
                for cand_z in range(self.n_discrete_size):
                    if occupancy[curr_x, curr_y, cand_z] == 1:
                        scores[i, cand_z + self.offset] = float('-inf')
            elif data_pos == 3:
                #predict x1
                curr_x = seq[-3]
                curr_y = seq[-2]
                curr_z = seq[-1]
                original_row = scores[i].clone()
                scores[i, self.offset : self.offset + self.n_discrete_size] = float('-inf')
                for h_len in self.valid_lengths:
                    cand_x1 = curr_x + h_len - 1
                    if cand_x1 >= self.n_discrete_size:
                        continue
                    if not occupancy[curr_x : cand_x1 + 1, curr_y, curr_z].any():
                        scores[i, cand_x1 + self.offset] = original_row[cand_x1 + self.offset]
                #print(f"x1 score: {scores}")
            elif data_pos == 4:
                #predict y2
                curr_x = seq[-4]
                curr_y = seq[-3]
                curr_z = seq[-2]
                curr_x1 = seq[-1]
                h_fixed = (curr_x1 - curr_x + 1).item()
                row_mask = torch.full_like(scores[i], float('-inf'))
                valid_ws = [w for (h_part, w) in self.allowed_sizes if h_part == h_fixed]
                valid_indices = []
                for w_len in valid_ws:
                    cand_y1 = curr_y + w_len - 1
                    if cand_y1 >= self.n_discrete_size:
                        continue
                    if not occupancy[curr_x : curr_x1 + 1, curr_y : cand_y1 + 1, curr_z].any():
                        valid_indices.append(cand_y1 + self.offset)
                if valid_indices:
                    valid_indices_tensor = torch.tensor(valid_indices, dtype=torch.long, device=scores.device)
                    row_mask[valid_indices_tensor] = 0
                scores[i] += row_mask
        #print(f"input_ids: {input_ids}")
        #print(f"scores: {scores}")
        return scores


class SingleGPT(nn.Module):
    def __init__(self, args):
        super().__init__()

        self.args = args
        self.point_encoder = load_model()
        self.cond_length = 257
        self.cond_dim = 768

        self.n_discrete_size = args.n_discrete_size
        self.max_seq_ratio = self.args.max_seq_ratio
        self.brick_per_token = 5
        self.pad_id = -1
        self.max_vertices = args.max_vertices

        self.max_length = int(args.n_max_bricks * self.brick_per_token * self.max_seq_ratio + 3 + self.cond_length)
        self.gen_max_length = int(args.gen_n_max_bricks * self.brick_per_token * self.max_seq_ratio + 3 + self.cond_length)

        self.coor_continuous_range = (-0.5, 0.5)

        vocab_size = self.n_discrete_size + 3 # 4 for bos, eos, pad
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
        #self.config._attn_implementation ="flash_attention_2"
        self.config.n_discrete_size = self.n_discrete_size
        self.config.face_per_token = self.brick_per_token
        self.config.cond_length = self.cond_length
        self.config.max_vertices = args.max_vertices
        self.config.word_embed_proj_dim = self.config.hidden_size

        self.transformer = AutoModelForCausalLM.from_config(
            config=self.config, 
            attn_implementation="flash_attention_2",
            torch_dtype=torch.float32
        )

        self.cond_head_proj = nn.Linear(self.cond_dim, self.config.word_embed_proj_dim)
        self.cond_proj = nn.Linear(self.cond_dim * 2, self.config.word_embed_proj_dim)

        self.train()

    def loop_detokenize(self, input_ids):
        batch_size = input_ids.shape[0]
        pad_id = -1
        valid_tokens = [row[row != pad_id] for row in input_ids]
        bricks = []
        for i in range(batch_size):
            tmp_bricks = []
            valid_seq = valid_tokens[i].tolist()
            #valid_seq = valid_seq[:-4]
            #print(f"valid_seq: {valid_seq}")
            assert len(valid_seq)%5==0,f"len of seq is error:{len(valid_seq)}"
            for i in range(0,len(valid_seq)//5):
                idx1 = 5*i
                idx2 = 5*i+1
                idx3 = 5*i+2
                idx4 = 5*i+3
                idx5 = 5*i+4
                x = valid_seq[idx1]
                y = valid_seq[idx2]
                z = valid_seq[idx3]
                h = valid_seq[idx4]-x+1
                w = valid_seq[idx5]-y+1
                #匹配砖块型号
                tmp_bricks.append([h,w,x,y,z])
            bricks.append(tmp_bricks)
        return bricks

    def train(self, mode: bool = True):
        super().train(mode)
        if hasattr(self,"point_encoder"):
            self.point_encoder.eval()
            for param in self.point_encoder.parameters():
                param.requires_grad = False

    def forward(self, data_dict: dict, is_eval: bool = False) -> dict:
        if not is_eval:
            return self.train_one_step(data_dict)
        else:
            return self.generate(data_dict)

    def pad_id_and_attn(self, input_ids, attention_mask, face_ids = None): # same
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
        encode_feature = torch.zeros(self.args.batchsize_per_gpu, self.cond_length, self.config.word_embed_proj_dim,
                                    device=self.cond_head_proj.weight.device, dtype=self.cond_head_proj.weight.dtype)
        encode_feature[:, 0] = self.cond_head_proj(point_feature[:, 0])
        shape_latents = self.point_encoder.to_shape_latents(point_feature[:, 1:])
        encode_feature[:, 1:] = self.cond_proj(torch.cat([point_feature[:, 1:], shape_latents], dim=-1))

        return encode_feature

    def train_one_step(self, data_dict: dict) -> dict:
        point_feature = self.point_encoder.encode_latents(data_dict["pc_normal"])
        with torch.no_grad():
            assert "sequence" in data_dict
            input_ids = data_dict['sequence']
            attention_mask = input_ids != self.pad_id
            sequence_max_length = attention_mask.sum(dim=1).max()
            input_ids = input_ids[:, :sequence_max_length]
            attention_mask = attention_mask[:, :sequence_max_length]
            input_ids, attention_mask = self.pad_id_and_attn(input_ids, attention_mask)
        #print(f"inputs_ids: {input_ids}")
        # add cond_length to attention mask
        pad_attention_mask = torch.ones((attention_mask.shape[0], self.cond_length), device=attention_mask.device, dtype=attention_mask.dtype)
        attention_mask = torch.concatenate((pad_attention_mask, attention_mask), dim=1)

        processed_point_feature = self.process_point_feature(point_feature=point_feature)

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

    @torch.no_grad()
    def generate(self, data_dict) -> dict:
        self.train(False)
        data = data_dict["pc_normal"].to("cuda")
        data_name = data_dict['model_name']
        #print(f"data_name: {data_name}")
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            point_feature = self.point_encoder.encode_latents(data)
            processed_point_feature = self.process_point_feature(point_feature)
            generate_length = self.gen_max_length - self.cond_length
            net_device = next(self.parameters()).device
            outputs = torch.ones(self.args.batchsize_per_gpu, generate_length).long().to(net_device) * self.eos_token_id
            mask_processors = LogitsProcessorList([Mask_invalid_token()])
            # batch x ntokens
            if self.args.num_beams is not None and "pc_normal" in data_dict:
                results = self.transformer.generate(
                    inputs_embeds=processed_point_feature,
                    max_new_tokens=generate_length,  # all faces plus two
                    num_beams=self.args.num_beams,
                    bos_token_id=self.bos_token_id,
                    eos_token_id=self.eos_token_id,
                    pad_token_id=self.pad_token_id,
                    logits_processor=mask_processors,
                )
            else:
                results = self.transformer.generate(
                    inputs_embeds = processed_point_feature,
                    max_new_tokens = generate_length, # all faces plus two
                    do_sample=True,
                    top_k=50,
                    top_p=0.95,
                    bos_token_id = self.bos_token_id,
                    eos_token_id = self.eos_token_id,
                    pad_token_id = self.pad_token_id,
                    logits_processor=mask_processors,
                )
        assert results.shape[1] <= generate_length # B x ID  bos is not included since it's predicted
        outputs[:, :results.shape[1]] = results
        # batch x ntokens ====> batch x ntokens x D
        outputs = outputs[:, 1: -1] # eos and bos removed

        outputs[outputs == self.bos_token_id] = self.pad_id
        outputs[outputs == self.eos_token_id] = self.pad_id
        outputs[outputs == self.pad_token_id] = self.pad_id

        outputs[outputs != self.pad_id] -= 3
        gen_brick = self.loop_detokenize(outputs)


        return gen_brick,data_name

