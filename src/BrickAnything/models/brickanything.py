import torch
from torch import nn
from transformers import AutoModelForCausalLM
from BrickAnything.miche.encode import load_model
from BrickAnything.models.shape_opt import ShapeOPTConfig
from einops import rearrange
from transformers import LogitsProcessor,LogitsProcessorList
from huggingface_hub import PyTorchModelHubMixin

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

class BrickAnything(nn.Module):

    def __init__(self, config={}):
        super().__init__()
        self.config = config
        self.point_encoder = load_model(ckpt_path=None)
        self.n_discrete_size = 20
        self.max_seq_ratio = 0.70
        self.face_per_token = 5
        self.cond_length = 257
        self.cond_dim = 768
        self.pad_id = -1
        self.n_max_triangles = 1600
        self.max_length = int(self.n_max_triangles * self.face_per_token * self.max_seq_ratio + 3 + self.cond_length) # add 1

        self.coor_continuous_range = (-0.5, 0.5)

        self.config = ShapeOPTConfig.from_pretrained(
            "facebook/opt-125m",
            n_positions=self.max_length,
            max_position_embeddings=self.max_length,
            vocab_size=self.n_discrete_size + 3,
            _attn_implementation="flash_attention_2"
        )

        self.bos_token_id = 0
        self.eos_token_id = 1
        self.pad_token_id = 2

        self.config.bos_token_id = self.bos_token_id
        self.config.eos_token_id = self.eos_token_id
        self.config.pad_token_id = self.pad_token_id
        self.config._attn_implementation="flash_attention_2"
        self.config.n_discrete_size = self.n_discrete_size
        self.config.face_per_token = self.face_per_token
        self.config.cond_length = self.cond_length

        if self.config.word_embed_proj_dim != self.config.hidden_size:
            self.config.word_embed_proj_dim = self.config.hidden_size
        self.transformer = AutoModelForCausalLM.from_config(
            config=self.config, 
            attn_implementation="flash_attention_2",
        )
        #self.transformer.to_bettertransformer()
        self.transformer = self.transformer.to(dtype=torch.float16)
        self.cond_head_proj = nn.Linear(self.cond_dim, self.config.word_embed_proj_dim).to(dtype=torch.float16)
        self.cond_proj = nn.Linear(self.cond_dim * 2, self.config.word_embed_proj_dim).to(dtype=torch.float16)

        self.eval()

    
    def process_point_feature(self, point_feature):
        encode_feature = torch.zeros(point_feature.shape[0], self.cond_length, self.config.word_embed_proj_dim,
                                    device=self.cond_head_proj.weight.device, dtype=self.cond_head_proj.weight.dtype)
        encode_feature[:, 0] = self.cond_head_proj(point_feature[:, 0])
        shape_latents = self.point_encoder.to_shape_latents(point_feature[:, 1:])
        encode_feature[:, 1:] = self.cond_proj(torch.cat([point_feature[:, 1:], shape_latents], dim=-1))

        return encode_feature
    
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
                tmp_bricks.append([h,w,x,y,z])
            bricks.append(tmp_bricks)
            
        #print(f"valid_tokens: {valid_tokens}")
        return bricks
    
    @torch.no_grad()
    def forward(self, pc_normal, sampling=False) -> dict:
        batch_size = pc_normal.shape[0]
        point_feature = self.point_encoder.encode_latents(pc_normal)
        processed_point_feature = self.process_point_feature(point_feature)
        generate_length = self.max_length - self.cond_length
        net_device = next(self.parameters()).device
        outputs = torch.ones(batch_size, generate_length).long().to(net_device) * self.eos_token_id
        mask_processors = LogitsProcessorList([Mask_invalid_token()])
        # batch x ntokens
        if not sampling:
            results = self.transformer.generate(
                inputs_embeds=processed_point_feature,
                max_new_tokens=generate_length,  # all faces plus two
                num_beams=1,
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
        print(results)
        #logits = results.logits
        #print(f"logits: {logits}")
        outputs[:, :results.shape[1]] = results
        # batch x ntokens ====> batch x ntokens x D
        outputs = outputs[:, 1: -1] # eos and bos removed
        outputs[outputs == self.bos_token_id] = self.pad_id
        outputs[outputs == self.eos_token_id] = self.pad_id
        outputs[outputs == self.pad_token_id] = self.pad_id

        outputs[outputs != self.pad_id] -= 3
        gen_brick = self.loop_detokenize(outputs)


        return gen_brick