from transformers import BartForConditionalGeneration
import torch
import torch.nn as nn
from torch.nn.init import xavier_uniform_
import torch.nn.functional as F
from KG import DBpedia, KGModel

def init_params(model):
    for name, param in model.named_parameters():
        if param.data.dim() > 1:
            xavier_uniform_(param.data)
        else:
            pass

def shift_tokens_right(input_ids: torch.Tensor, pad_token_id: int, decoder_start_token_id: int):
    """
    Shift input ids one token to the right.
    """
    shifted_input_ids = input_ids.new_zeros(input_ids.shape)
    shifted_input_ids[:, 1:] = input_ids[:, :-1].clone()
    shifted_input_ids[:, 0] = decoder_start_token_id

    assert pad_token_id is not None, "self.model.config.pad_token_id has to be defined."
    # replace possible -100 values in labels by `pad_token_id`
    shifted_input_ids.masked_fill_(shifted_input_ids == -100, pad_token_id)

    return shifted_input_ids

class MLP(nn.Module):
    def __init__(self, input_size, output_size, hidden_size):
        super().__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, output_size)
    
    def forward(self, din):
        dout = F.relu(self.fc1(din))
        dout = F.relu(self.fc2(dout))
        return dout


class UniMind(nn.Module):
    def __init__(self, args, config, item_num):
        super().__init__()
        revision = "cac8228"
        self.bart = BartForConditionalGeneration.from_pretrained(args.model_name_or_path, from_tf=bool('.ckpt' in args.model_name_or_path),
                    config=config, cache_dir=args.cache_dir, revision=revision)
        d_model = config.d_model
        self.mlp = MLP(d_model, d_model//4, d_model//2)
        self.classifier = nn.Linear(d_model//4, item_num)
        #self.criterion = F.binary_cross_entropy_with_logits
        self.criterion = F.cross_entropy
        self.item_num = item_num
        self.config = config


        # construct kg
        kg = DBpedia(dataset_dir='/projects/prjs1158/KG/redail/MESE_kg/DATA/nltk', debug=False)  
        kg_info = kg.get_entity_kg_info()
        self.entity_encoder = KGModel(config.d_model, 
            n_entity=kg_info['num_entities'], num_relations=kg_info['num_relations'], num_bases=8,
            edge_index=kg_info['edge_index'], edge_type=kg_info['edge_type'],
        ).to(args.device)

        # self.entity_emb = self.entity_encoder.get_entity_embeds()
        # 保存entity_encoder而不是静态的embeddings
        # self.entity_encoder = entity_encoder
        self.entity_proj = nn.Linear(d_model, d_model)  # 注意这里的输入维度变化
        
        init_params(self.classifier)
        init_params(self.mlp)

    def forward(self, input_ids, attention_mask, hist_ids=None, labels=None, item_ids=None, entity_ids=None, entity_attn=None):
        # 处理输入嵌入
        entity_emb = self.entity_proj(self.entity_encoder.get_entity_embeds()[entity_ids])
        input_embeds = self.bart.model.encoder.embed_tokens(input_ids)
        input_embeds = torch.cat([input_embeds, entity_emb], dim=1)
        

        attention_mask = torch.cat([attention_mask, entity_attn], dim=1)
        # print(f'BART config: {self.config}')

        # print(f'input emb shape: {input_embeds.shape}')
        # print(f'attention_mask shaoe: {attention_mask.shape}')

        # print(f'input_ids: {input_ids}')
        if labels is not None:
            outputs = self.bart(
                inputs_embeds=input_embeds,
                attention_mask=attention_mask,
                labels=labels,
                output_hidden_states=True
            )
        else:
            batch_size = input_ids.shape[0]
            decoder_input_ids = torch.full(
                (batch_size, 1),  # 仅提供一个起始 token
                self.bart.config.decoder_start_token_id,
                dtype=torch.long,
                device=input_ids.device
            )
            outputs = self.bart(
                inputs_embeds=input_embeds,
                attention_mask=attention_mask,
                labels=labels,
                decoder_input_ids=decoder_input_ids,
                output_hidden_states=True
            )
        
        encode_state = outputs.decoder_hidden_states[-1][:,0,:].squeeze(1)
        hidden = self.mlp(encode_state)
        res = self.classifier(hidden)
        
        if self.training:
            loss_g = outputs[0]
            loss_r = self.criterion(res, item_ids.squeeze(1), ignore_index=self.item_num-1)
            return loss_g + loss_r, res, loss_g
        return res
    
    def generate(self, input_ids=None, attention_mask=None, entity_ids=None, entity_attn=None, **kwargs):
        # 处理输入嵌入
        if entity_ids is not None:
            entity_emb = self.entity_proj(self.entity_encoder.get_entity_embeds()[entity_ids])
            input_embeds = self.bart.model.encoder.embed_tokens(input_ids)
            input_embeds = torch.cat([input_embeds, entity_emb], dim=1)
        
        # 同样需要扩展attention_mask
        # entity_attention = torch.ones(
        #     attention_mask.shape[0], 
        #     entity_emb.shape[1], 
        #     device=attention_mask.device, 
        #     dtype=attention_mask.dtype
        # )
            attention_mask = torch.cat([attention_mask, entity_attn], dim=1)
        else:
            input_embeds = self.bart.model.encoder.embed_tokens(input_ids)
        outputs = self.bart.generate(
            inputs_embeds=input_embeds,
            attention_mask=attention_mask,
            **kwargs
        )
        return outputs
