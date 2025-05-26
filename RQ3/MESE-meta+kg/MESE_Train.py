#!/usr/bin/env python
# coding: utf-8

# In[1]:

from rouge_score import rouge_scorer
import torch
from torch.utils.data import DataLoader, Dataset

from transformers import GPT2Config, GPT2Tokenizer, BertModel, BertTokenizer, DistilBertModel, DistilBertTokenizer
from transformers import AdamW, get_linear_schedule_with_warmup
import torch.nn.functional as F
import torch.nn as nn

from utils.InductiveAttentionModels import GPT2InductiveAttentionHeadModel
from utils.SequenceCrossEntropyLoss import SequenceCrossEntropyLoss
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
import numpy as np
import math
import random
import time
import tqdm
import os
import string
import json
from KG import DBpedia, KGModel
from nltk.translate import bleu_score
from nltk.translate.bleu_score import sentence_bleu

from annoy import AnnoyIndex
from rouge import Rouge  # Add this import at the top of the file


class FusionGate(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.gate = nn.Linear(dim * 2, dim)
        self.linear1 = nn.Linear(dim * 2, dim)
        
    def forward(self, meta_emb, entity_emb):
        combined = torch.cat([meta_emb, entity_emb], dim=-1)
        gate = torch.sigmoid(self.gate(combined))
        fused = self.linear1(combined)
        return gate * meta_emb + (1 - gate) * fused
    
# In[2]:
def intra_distinct_metrics(outs):
    """
    Calculate intra-distinct metrics for each sentence
    Args:
        outs: list of sentences, where each sentence is a list of words
    Returns:
        dis1, dis2, dis3, dis4: intra-distinct scores for 1-4 grams
    """
    # 存储每个句子的 ratio
    ratios1 = []
    ratios2 = []
    ratios3 = []
    ratios4 = []

    for sen in outs:
        # 对每个句子计算 distinct ratio
        
        # unigrams
        unigram_set = set()
        unigram_total = 0
        for word in sen:
            unigram_total += 1
            unigram_set.add(word)
        if unigram_total > 0:
            ratios1.append(len(unigram_set) / unigram_total)
        
        # bigrams
        bigram_set = set()
        bigram_total = 0
        if len(sen) >= 2:
            for start in range(len(sen) - 1):
                bg = str(sen[start]) + ' ' + str(sen[start + 1])
                bigram_total += 1
                bigram_set.add(bg)
            ratios2.append(len(bigram_set) / bigram_total)
        
        # trigrams
        trigram_set = set()
        trigram_total = 0
        if len(sen) >= 3:
            for start in range(len(sen) - 2):
                trg = str(sen[start]) + ' ' + str(sen[start + 1]) + ' ' + str(sen[start + 2])
                trigram_total += 1
                trigram_set.add(trg)
            ratios3.append(len(trigram_set) / trigram_total)
        
        # quagrams
        quagram_set = set()
        quagram_total = 0
        if len(sen) >= 4:
            for start in range(len(sen) - 3):
                quag = str(sen[start]) + ' ' + str(sen[start + 1]) + ' ' + str(sen[start + 2]) + ' ' + str(sen[start + 3])
                quagram_total += 1
                quagram_set.add(quag)
            ratios4.append(len(quagram_set) / quagram_total)

    # 计算所有句子的平均 ratio
    dis1 = sum(ratios1) / len(outs) if ratios1 else 0
    dis2 = sum(ratios2) / len(outs) if ratios2 else 0
    dis3 = sum(ratios3) / len(outs) if ratios3 else 0
    dis4 = sum(ratios4) / len(outs) if ratios4 else 0

    return dis1, dis2, dis3, dis4

def inter_distinct_metrics(outs):
    """
    Calculate inter-diversity metrics between sentences
    Args:
        outs: list of sentences, where each sentence is a list of words
    Returns:
        dis1, dis2, dis3, dis4: inter-diversity scores for 1-4 grams
    """
    # 收集所有n-grams
    all_unigrams = []
    all_bigrams = []
    all_trigrams = []
    all_quagrams = []

    for sen in outs:
        # unigrams
        all_unigrams.extend(sen)
        
        # bigrams
        for start in range(len(sen) - 1):
            bg = str(sen[start]) + ' ' + str(sen[start + 1])
            all_bigrams.append(bg)
            
        # trigrams
        for start in range(len(sen) - 2):
            trg = str(sen[start]) + ' ' + str(sen[start + 1]) + ' ' + str(sen[start + 2])
            all_trigrams.append(trg)
            
        # quagrams
        for start in range(len(sen) - 3):
            quag = str(sen[start]) + ' ' + str(sen[start + 1]) + ' ' + str(sen[start + 2]) + ' ' + str(sen[start + 3])
            all_quagrams.append(quag)

    # 计算distinct比率
    dis1 = len(set(all_unigrams)) / max(len(all_unigrams), 1)  # 避免除零
    dis2 = len(set(all_bigrams)) / max(len(all_bigrams), 1)
    dis3 = len(set(all_trigrams)) / max(len(all_trigrams), 1)
    dis4 = len(set(all_quagrams)) / max(len(all_quagrams), 1)

    return dis1, dis2, dis3, dis4
    


    
    
def bleu_calc_one(ref, hyp):
    for i in range(len(ref)):
        ref[i] = ref[i].lower()
    for i in range(len(hyp)):
        hyp[i] = hyp[i].lower()
    smoother = SmoothingFunction().method1
    bleu1 = sentence_bleu([ref], hyp, weights=(1, 0, 0, 0), smoothing_function=smoother)
    bleu2 = sentence_bleu([ref], hyp, weights=(0.5, 0.5, 0, 0), smoothing_function=smoother)
    bleu3 = sentence_bleu([ref], hyp, weights=(0.33, 0.33, 0.33, 0), smoothing_function=smoother)
    bleu4 = sentence_bleu([ref], hyp, weights=(0.25, 0.25, 0.25, 0.25), smoothing_function=smoother)
    return bleu1, bleu2, bleu3, bleu4

def rouge_calc_one(ref, hyp):
    """计算单个句子的ROUGE分数
    Args:
        ref: 参考句子的词列表
        hyp: 生成句子的词列表
    Returns:
        rouge-1,2,l的f1分数
    """
    # 预处理:转小写,去标点
    ref = [word.lower() for word in ref]
    hyp = [word.lower() for word in hyp]
    
    # 截断过长的句子以避免递归深度问题
    max_len = 100  # 设置最大长度
    ref = ref[:max_len]
    hyp = hyp[:max_len]
    
    # 转换为字符串
    ref = ' '.join(ref)
    hyp = ' '.join(hyp)
    
    # 初始化Rouge评估器
    rouge = Rouge()
    
    try:
        # 计算ROUGE分数
        scores = rouge.get_scores(hyp, ref)[0]
        
        rouge1 = scores['rouge-1']['f']
        rouge2 = scores['rouge-2']['f'] 
        rougel = scores['rouge-l']['f']
        
        return rouge1, rouge2, rougel
        
    except (ValueError, RecursionError): # 处理空字符串和递归错误
        return 0, 0, 0

def rouge_calc_all(originals, generated):
    """计算所有句子的平均ROUGE分数
    Args:
        originals: 参考句子列表的列表
        generated: 生成句子列表的列表
    Returns:
        平均rouge-1,2,l分数
    """
    rouge1_total = 0
    rouge2_total = 0  
    rougel_total = 0
    total = 0
    
    for o, g in zip(originals, generated):
        # 预处理句子
        ref = [i.translate(str.maketrans('', '', string.punctuation)) for i in o]
        hyp = [i.translate(str.maketrans('', '', string.punctuation)) for i in g]
        
        # 跳过无效句子
        if len(hyp) == 0 or len(ref) == 0:
            continue
            
        try:
            rouge1, rouge2, rougel = rouge_calc_one(ref, hyp)
            rouge1_total += rouge1
            rouge2_total += rouge2
            rougel_total += rougel
            total += 1
        except Exception as e:
            print(f"Error calculating ROUGE scores: {e}")
            continue
        
    if total == 0:
        return 0, 0, 0
        
    return (
        rouge1_total / total,
        rouge2_total / total,
        rougel_total / total
    )

def bleu_calc_all(originals, generated):
    bleu1_total, bleu2_total, bleu3_total, bleu4_total = 0, 0, 0, 0
    total = 0
    for o, g in zip(originals, generated):
        r = [ i.translate(str.maketrans('', '', string.punctuation)) for i in o][1:]
        h = [ i.translate(str.maketrans('', '', string.punctuation)) for i in g][1:]
        # if '[MOVIE_ID]' in r: continue
#         if len(g) >= 500: continue
        # if len(g) >= 100: continue
        bleu1, bleu2, bleu3, bleu4 = bleu_calc_one(r, h)
        bleu1_total += bleu1; bleu2_total += bleu2; bleu3_total += bleu3; bleu4_total += bleu4;
        total += 1
    return bleu1_total / total, bleu2_total / total, bleu3_total / total, bleu4_total / total


# In[13]:


def replace_placeholder(sentence, movie_titles):
    sen = sentence
    for title in movie_titles:
        sen = sen.replace("[MOVIE_ID]", title, 1)
    return sen
        


bert_tokenizer = DistilBertTokenizer.from_pretrained("distilbert-base-uncased")
bert_model_recall = DistilBertModel.from_pretrained('distilbert-base-uncased')
bert_model_rerank = DistilBertModel.from_pretrained('distilbert-base-uncased')
gpt_tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
gpt2_model = GPT2InductiveAttentionHeadModel.from_pretrained('gpt2')

# REC_TOKEN = "R"
# REC_END_TOKEN = "E"
REC_TOKEN = "[REC]"
REC_END_TOKEN = "[REC_END]"
SEP_TOKEN = "[SEP]"
PLACEHOLDER_TOKEN = "[MOVIE_ID]"
gpt_tokenizer.add_tokens([REC_TOKEN, REC_END_TOKEN, SEP_TOKEN, PLACEHOLDER_TOKEN])
gpt2_model.resize_token_embeddings(len(gpt_tokenizer)) 
original_token_emb_size = gpt2_model.get_input_embeddings().weight.shape[0]


# In[3]:


class MovieRecDataset(Dataset):
    def __init__(self, data, json_data, bert_tok, gpt2_tok):
        self.data = data
        self.json_data = json_data
        # print(f'data')
        # print(self.data)
        self.bert_tok = bert_tok
        self.gpt2_tok = gpt2_tok
        self.turn_ending = torch.tensor([[628, 198]]) # end of turn, '\n\n\n'

        # load entity2id
        entity2id_path = "./DATA/nltk/entity2id.json"
        entity2id_data = json.load(open(entity2id_path, 'r', encoding='utf-8'))
        self.entity2id = entity2id_data
        
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, index):
        dialogue_data = self.data[index]
        
        dialogue = dialogue_data[1]

        entity_data = self.json_data[index]['dialog']
        
        
        dialogue_tokens = []
        
        for cnt, (utterance, gt_ind) in enumerate(dialogue):
            entity = entity_data[cnt]['entity']
            movie = entity_data[cnt]['movies']
            overall_entity = entity + movie
            filter_overall_entity = [self.entity2id[entity] for entity in overall_entity if entity in self.entity2id]
            utt_tokens = self.gpt2_tok(utterance, return_tensors="pt")['input_ids']
            utt_tokens = torch.cat([utt_tokens, self.turn_ending], dim=1)
            dialogue_tokens.append((utt_tokens, gt_ind, filter_overall_entity))
            # dialogue_tokens.append( ( torch.cat( [utt_tokens, self.turn_ending], dim=1), gt_ind) )
            
        role_ids = None
        previous_role_ids = None
        if role_ids == None:
            role_ids = [ 0 if item[0] == 'B' else 1 for item, _ in dialogue]
            previous_role_ids = role_ids
        else:
            role_ids = [ 0 if item[0] == 'B' else 1 for item, _ in dialogue]
            if not np.array_equal(role_ids, previous_role_ids):
                raise Exception("Role ids dont match between languages")
            previous_role_ids = role_ids
        
        return role_ids, dialogue_tokens
    
    def collate(self, unpacked_data):
        return unpacked_data
    


# In[4]:


train_path = "/projects/0/prjs1158/KG/redail/MESE/DATA/train_data_processed"
valid_path = "/projects/0/prjs1158/KG/redail/MESE/DATA/valid_data_processed"
test_path = "/projects/0/prjs1158/KG/redail/MESE/DATA/test_data_processed"

train_json = "./DATA/nltk/train_data.json"
valid_json = "./DATA/nltk/valid_data.json"
test_json = "./DATA/nltk/test_data.json"

train_json_data = json.load(open(train_json, 'r', encoding='utf-8'))
valid_json_data = json.load(open(valid_json, 'r', encoding='utf-8'))
test_json_data = json.load(open(test_json, 'r', encoding='utf-8'))

items_db_path = "/projects/0/prjs1158/KG/redail/MESE/DATA/movie_db"


# In[5]:


train_dataset = MovieRecDataset(torch.load(train_path), train_json_data, bert_tokenizer, gpt_tokenizer)
valid_dataset = MovieRecDataset(torch.load(valid_path), valid_json_data, bert_tokenizer, gpt_tokenizer)
test_dataset = MovieRecDataset(torch.load(test_path), test_json_data, bert_tokenizer, gpt_tokenizer)
train_dataloader = DataLoader(dataset=train_dataset, shuffle=False, batch_size=1, collate_fn=train_dataset.collate)
valid_dataloader = DataLoader(dataset=valid_dataset, shuffle=False, batch_size=1, collate_fn=valid_dataset.collate)
test_dataloader = DataLoader(dataset=test_dataset, shuffle=False, batch_size=1, collate_fn=test_dataset.collate)


# In[6]:


items_db = torch.load(items_db_path)
# print(items_db)

def sample_ids_from_db(item_db,
                       gt_id, # ground truth id
                       num_samples, # num samples to return
                       include_gt # if we want gt_id to be included
                      ):
    ids_2_sample_from = list(item_db.keys())
    ids_2_sample_from.remove(gt_id)
    if include_gt:
        results = random.sample(ids_2_sample_from, num_samples-1)
        results.append(gt_id)
    else:
        results = random.sample(ids_2_sample_from, num_samples)
    return results


# In[7]:


class UniversalCRSModel(torch.nn.Module):
    def __init__(self, 
                 language_model, # backbone of Pretrained LM such as GPT2
                 encoder, # backbone of item encoder such as bert
                 recall_encoder,
                 lm_tokenizer, # language model tokenizer
                 encoder_tokenizer, # item encoder tokenizer
                 device, # Cuda device
                 items_db, # {id:info}, information of all items to be recommended
                 annoy_base_recall=None, # annoy index base of encoded recall embeddings of items
                 annoy_base_rerank=None, # annoy index base of encoded rerank embeddings of items, for validation and inference only
                 recall_item_dim=768, # dimension of each item to be stored in annoy base
                 lm_trim_offset=100, # offset to trim language model wte inputs length = (1024-lm_trim_offset)
                 rec_token_str="[REC]", # special token indicating recommendation and used for recall
                 rec_end_token_str="[REC_END]", # special token indicating recommendation ended, conditional generation starts
                 sep_token_str="[SEP]",
                 placeholder_token_str="[MOVIE_ID]"
                ):
        super(UniversalCRSModel, self).__init__()
        
        #models and tokenizers
        self.language_model = language_model
        self.item_encoder = encoder
        self.recall_encoder = recall_encoder
        self.lm_tokenizer = lm_tokenizer
        self.encoder_tokenizer = encoder_tokenizer
        self.device = device
        
        # item db and annoy index base
        self.items_db = items_db
        self.annoy_base_recall = annoy_base_recall
        self.annoy_base_rerank = annoy_base_rerank
        
        # hyperparameters
        self.recall_item_dim = recall_item_dim
        self.lm_trim_offset = lm_trim_offset
        
        #constants
        self.REC_TOKEN_STR = rec_token_str
        self.REC_END_TOKEN_STR = rec_end_token_str
        self.SEP_TOKEN_STR = sep_token_str
        self.PLACEHOLDER_TOKEN_STR = placeholder_token_str
        self.fusion_gate = FusionGate(self.language_model.config.n_embd)
        
        # map language model hidden states to a vector to query annoy-item-base for recall
        self.recall_lm_query_mapper = torch.nn.Linear(self.language_model.config.n_embd, self.recall_item_dim) # default [768,768]
        # map output of self.item_encoder to vectors to be stored in annoy-item-base 
        self.recall_item_wte_mapper = torch.nn.Linear(self.recall_encoder.config.hidden_size, self.recall_item_dim) # default [768,768]
        # map output of self.item_encoder to a wte of self.language_model
        self.rerank_item_wte_mapper = torch.nn.Linear(self.item_encoder.config.hidden_size, self.language_model.config.n_embd) # default [768,768]
        # map language model hidden states of item wte to a one digit logit for softmax computation
        self.rerank_logits_mapper = torch.nn.Linear(self.language_model.config.n_embd, 1) # default [768,1]
    
        # add KG
        # self.kg_info = DBpedia(dataset_dir='./DATA/nltk', debug=False).get_entity_kg_info()
        self.kg = DBpedia(dataset_dir='./DATA/nltk', debug=False)  
        self.kg_info = self.kg.get_entity_kg_info()
        self.entity_encoder = KGModel(self.language_model.config.n_embd, 
            n_entity=self.kg_info['num_entities'], num_relations=self.kg_info['num_relations'], num_bases=8,
            edge_index=self.kg_info['edge_index'], edge_type=self.kg_info['edge_type'],
        ).to(device)
        with torch.no_grad():  # 使用no_grad来缓存实体嵌入
            self.entity_emb = self.entity_encoder.get_entity_embeds()
        self.entity_mapper = torch.nn.Linear(
            self.language_model.config.n_embd*2,  # entity维度
            self.language_model.config.n_embd  # 映射到GPT2维度
        )

    def get_sep_token_wtes(self):
        sep_token_input_ids = self.lm_tokenizer(self.SEP_TOKEN_STR, return_tensors="pt")["input_ids"].to(self.device)
        return self.language_model.transformer.wte(sep_token_input_ids) # [1, 1, self.language_model.config.n_embd]

    def get_rec_token_wtes(self):
        rec_token_input_ids = self.lm_tokenizer(self.REC_TOKEN_STR, return_tensors="pt")["input_ids"].to(self.device)
        return self.language_model.transformer.wte(rec_token_input_ids) # [1, 1, self.language_model.config.n_embd]
    
    def get_rec_end_token_wtes(self):
        rec_end_token_input_ids = self.lm_tokenizer(self.REC_END_TOKEN_STR, return_tensors="pt")["input_ids"].to(self.device)
        return self.language_model.transformer.wte(rec_end_token_input_ids) # [1, 1, self.language_model.config.n_embd]
    
    def get_movie_title(self, m_id):
        title = self.items_db[m_id]
        title = title.split('[SEP]')[0].strip()
        return title
    
    # compute BERT encoded item hidden representation
    # output can be passed to self.recall_item_wte_mapper or self.rerank_item_wte_mapper
    def compute_encoded_embeddings_for_items(self, 
                                             encoder_to_use,
                                             item_ids, # an array of ids, single id should be passed as [id]
                                             items_db_to_use, # item databse to use
                                             entity_list # overall entity
                                            ):
        chunk_ids = item_ids
        chunk_infos = [items_db_to_use[key] for key in chunk_ids ]
        # print(chunk_infos)
        chunk_tokens = self.encoder_tokenizer(chunk_infos, padding=True, truncation=True, return_tensors="pt")
        chunk_input_ids = chunk_tokens['input_ids'].to(self.device)
        chunk_attention_mask = chunk_tokens['attention_mask'].to(self.device)
        chunk_hiddens = encoder_to_use(input_ids=chunk_input_ids, attention_mask=chunk_attention_mask).last_hidden_state

        # average of non-padding tokens
        expanded_mask_size = list(chunk_attention_mask.size())
        expanded_mask_size.append(encoder_to_use.config.hidden_size)
        expanded_mask = chunk_attention_mask.unsqueeze(-1).expand(expanded_mask_size)
        chunk_masked = torch.mul(chunk_hiddens, expanded_mask) # [num_example, len, 768]
        chunk_pooled = torch.sum(chunk_masked, dim=1) / torch.sum(chunk_attention_mask, dim=1).unsqueeze(-1)
        
        if entity_list is not None:
            if len(entity_list) != 0:
                # single_entity_emb = torch.tensor([self.entity_emb[int(entity_id)] for entity_id in entity_list]).to(self.device)
                single_entity_emb = torch.stack([self.entity_emb[int(entity_id)] for entity_id in entity_list])
                average_entity_emb = torch.mean(single_entity_emb, dim=0)
                normalized_entity_emb = F.normalize(average_entity_emb, p=2, dim=-1)
                batch_entity_emb = normalized_entity_emb
            else:
                batch_entity_emb = torch.zeros(self.language_model.config.n_embd).to(self.device)
            # print(f"batch_entity_emb shape: {batch_entity_emb.unsqueeze(0).shape}")
            # print(f"chunk_pooled shape: {chunk_pooled.shape}")
            chunk_pooled = batch_entity_emb.unsqueeze(0)
        # batch_entity_emb = torch.cat(batch_entity_emb, dim=0)

        # concat entity emb and chunk_pooled
        # combined_embeddings = torch.cat([chunk_pooled, batch_entity_emb], dim=-1)
        # print(f"batch_entity_emb shape: {batch_entity_emb.shape}")
        # print(f"chunk_pooled shape: {chunk_pooled.shape}")
        
        # map to original dimension
        # chunk_pooled = self.combined_mapper(combined_embeddings)  # [batch_size, gpt2_dim]


        # [len(item_ids), encoder_to_use.config.hidden_size], del chunk_hiddens to free up GPU memory
        return chunk_pooled, chunk_hiddens
    
    # annoy_base_constructor, constructs
    def annoy_base_constructor(self, items_db=None, distance_type='angular', chunk_size=50, n_trees=10):
        items_db_to_use = self.items_db if items_db == None else items_db
        all_item_ids = list(items_db_to_use.keys())
        
        total_pooled = []
        # break into chunks/batches for model concurrency
        num_chunks = math.ceil(len(all_item_ids) / chunk_size)
        for i in range(num_chunks):
            chunk_ids = all_item_ids[i*chunk_size: (i+1)*chunk_size]
            chunk_pooled, chunk_hiddens = self.compute_encoded_embeddings_for_items(self.recall_encoder, chunk_ids, items_db_to_use, None)
            chunk_pooled = chunk_pooled.cpu().detach().numpy()
            del chunk_hiddens
            total_pooled.append(chunk_pooled)
        total_pooled = np.concatenate(total_pooled, axis=0)
        
        pooled_tensor = torch.tensor(total_pooled).to(self.device)
        
        #build recall annoy index
        annoy_base_recall = AnnoyIndex(self.recall_item_wte_mapper.out_features, distance_type)
        pooled_recall = self.recall_item_wte_mapper(pooled_tensor) # [len(items_db_to_use), self.recall_item_wte_mapper.out_features]
        pooled_recall = pooled_recall.cpu().detach().numpy()
        for i, vector in zip(all_item_ids, pooled_recall):
            annoy_base_recall.add_item(i, vector)
        annoy_base_recall.build(n_trees)
        
        total_pooled = []
        # break into chunks/batches for model concurrency
        num_chunks = math.ceil(len(all_item_ids) / chunk_size)
        for i in range(num_chunks):
            chunk_ids = all_item_ids[i*chunk_size: (i+1)*chunk_size]
            chunk_pooled, chunk_hiddens = self.compute_encoded_embeddings_for_items(self.item_encoder, chunk_ids, items_db_to_use, None)
            chunk_pooled = chunk_pooled.cpu().detach().numpy()
            del chunk_hiddens
            total_pooled.append(chunk_pooled)
        total_pooled = np.concatenate(total_pooled, axis=0)
        
        pooled_tensor = torch.tensor(total_pooled).to(self.device)
        
        #build rerank annoy index, for validation and inference only
        annoy_base_rerank = AnnoyIndex(self.rerank_item_wte_mapper.out_features, distance_type)
        pooled_rerank = self.rerank_item_wte_mapper(pooled_tensor) # [len(items_db_to_use), self.recall_item_wte_mapper.out_features]
        pooled_rerank = pooled_rerank.cpu().detach().numpy()
        for i, vector in zip(all_item_ids, pooled_rerank):
            annoy_base_rerank.add_item(i, vector)
        annoy_base_rerank.build(n_trees)
        
        del pooled_tensor
        
        self.annoy_base_recall = annoy_base_recall
        self.annoy_base_rerank = annoy_base_rerank
    
    def annoy_loader(self, path, annoy_type, distance_type="angular"):
        if annoy_type == "recall":
            annoy_base = AnnoyIndex(self.recall_item_wte_mapper.out_features, distance_type)
            annoy_base.load(path)
            return annoy_base
        elif annoy_type == "rerank":
            annoy_base = AnnoyIndex(self.rerank_item_wte_mapper.out_features, distance_type)
            annoy_base.load(path)
            return annoy_base
        else:
            return None
    
    def lm_expand_wtes_with_items_annoy_base(self):
        all_item_ids = list(self.items_db.keys())
        total_pooled = []
        for i in all_item_ids:
            total_pooled.append(self.annoy_base_rerank.get_item_vector(i))
        total_pooled = np.asarray(total_pooled) # [len(all_item_ids), 768]
        pooled_tensor = torch.tensor(total_pooled, dtype=torch.float).to(self.device)
        
        old_embeddings = self.language_model.get_input_embeddings()
        item_id_2_lm_token_id = {}
        for k in all_item_ids:
            item_id_2_lm_token_id[k] = len(item_id_2_lm_token_id) + old_embeddings.weight.shape[0]
        new_embeddings = torch.cat([old_embeddings.weight, pooled_tensor], 0)
        new_embeddings = torch.nn.Embedding.from_pretrained(new_embeddings)
        self.language_model.set_input_embeddings(new_embeddings)
        self.language_model.to(device)
        return item_id_2_lm_token_id
    
    def lm_restore_wtes(self, original_token_emb_size):
        old_embeddings = self.language_model.get_input_embeddings()
        new_embeddings = torch.nn.Embedding(original_token_emb_size, old_embeddings.weight.size()[1])
        new_embeddings.to(self.device, dtype=old_embeddings.weight.dtype)
        new_embeddings.weight.data[:original_token_emb_size, :] = old_embeddings.weight.data[:original_token_emb_size, :]
        self.language_model.set_input_embeddings(new_embeddings)
        self.language_model.to(self.device)
        assert(self.language_model.get_input_embeddings().weight.shape[0] == original_token_emb_size)
    
    def trim_lm_wtes(self, wtes):
        trimmed_wtes = wtes
        if trimmed_wtes.shape[1] > self.language_model.config.n_positions:
            trimmed_wtes = trimmed_wtes[:,-self.language_model.config.n_positions + self.lm_trim_offset:,:]
        return trimmed_wtes # [batch, self.language_model.config.n_positions - self.lm_trim_offset, self.language_model.config.n_embd]
    
    def trim_positional_ids(self, p_ids, num_items_wtes):
        trimmed_ids = p_ids
        if trimmed_ids.shape[1] > self.language_model.config.n_positions:
            past_ids = trimmed_ids[:,:self.language_model.config.n_positions - self.lm_trim_offset - num_items_wtes]
#             past_ids = trimmed_ids[:, self.lm_trim_offset + num_items_wtes:self.language_model.config.n_positions]
            item_ids = trimmed_ids[:,-num_items_wtes:]
            trimmed_ids = torch.cat((past_ids, item_ids), dim=1)
        return trimmed_ids # [batch, self.language_model.config.n_positions - self.lm_trim_offset]
    
    def compute_inductive_attention_mask(self, length_language, length_rerank_items_wtes):
        total_length = length_language + length_rerank_items_wtes
        language_mask_to_add = torch.zeros((length_language, total_length), dtype=torch.float, device=self.device)
        items_mask_to_add = torch.ones((length_rerank_items_wtes, total_length), dtype=torch.float, device=self.device)
        combined_mask_to_add = torch.cat((language_mask_to_add, items_mask_to_add), dim=0)
        return combined_mask_to_add #[total_length, total_length]
    
    def forward_pure_language_turn(self, 
                                   past_wtes, # past word token embeddings, [1, len, 768]
                                   current_tokens # tokens of current turn conversation, [1, len]
                                  ):
        train_logits, train_targets = None, None
        current_wtes = self.language_model.transformer.wte(current_tokens)
        
        if past_wtes == None:
            lm_outputs = self.language_model(inputs_embeds=current_wtes)
            train_logits = lm_outputs.logits[:, :-1, :]
            train_targets = current_tokens[:,1:]
        else:
            all_wtes = torch.cat((past_wtes, current_wtes), dim=1)
            all_wtes = self.trim_lm_wtes(all_wtes)
            lm_outputs = self.language_model(inputs_embeds=all_wtes)
            train_logits = lm_outputs.logits[:, -current_wtes.shape[1]:-1, :] # skip the last one
            train_targets = current_tokens[:,1:]
        
        # torch.Size([batch, len_cur, lm_vocab]), torch.Size([batch, len_cur]), torch.Size([batch, len_past+len_cur, lm_emb(768)])
        return train_logits, train_targets
        
    def forward_recall(self, 
                       past_wtes, # past word token embeddings, [1, len, 768]
                       current_tokens, # tokens of current turn conversation, [1, len]
                       gt_item_id, # id, ex. 0
                       num_samples, # num examples to sample for training, including groud truth id
                       overall_entity # overall entity
                      ):
        # recall step 1. construct LM sequence output
        # LM input composition: [past_wtes, REC_wtes, gt_item_wte, (gt_item_info_wtes), REC_END_wtes, current_wtes ]
        # print(f"overall_entity: {overall_entity}")
        
        REC_wtes = self.get_rec_token_wtes() # [1, 1, self.language_model.config.n_embd]
        gt_item_wte, _ = self.compute_encoded_embeddings_for_items(self.recall_encoder, [gt_item_id], self.items_db, overall_entity)
        gt_item_wte = self.rerank_item_wte_mapper(gt_item_wte) # [1, self.rerank_item_wte_mapper.out_features]
        
        REC_END_wtes = self.get_rec_end_token_wtes() # [1, 1, self.language_model.config.n_embd]
        current_wtes = self.language_model.transformer.wte(current_tokens) #[1, current_tokens.shape[1], self.language_model.config.n_embd]
        
        REC_wtes_len = REC_wtes.shape[1] # 1 by default
        gt_item_wte_len = gt_item_wte.shape[0] # 1 by default
        REC_END_wtes_len = REC_END_wtes.shape[1] # 1 by default
        current_wtes_len = current_wtes.shape[1]
        
        lm_wte_inputs = torch.cat(
            (past_wtes, # [batch (1), len, self.language_model.config.n_embd]
             REC_wtes,
             gt_item_wte.unsqueeze(0), # reshape to [1,1,self.rerank_item_wte_mapper.out_features]
             REC_END_wtes,
             current_wtes # [batch (1), len, self.language_model.config.n_embd]
            ),
            dim=1
        )
        lm_wte_inputs = self.trim_lm_wtes(lm_wte_inputs) # trim for len > self.language_model.config.n_positions
        
        # recall step 2. get gpt output logits and hidden states
        lm_outputs = self.language_model(inputs_embeds=lm_wte_inputs, output_hidden_states=True)
        
        # recall step 3. pull logits (recall, rec_token and language logits of current turn) and compute
        
        # recall logit(s)
        rec_token_start_index = -current_wtes_len-REC_END_wtes_len-gt_item_wte_len-REC_wtes_len
        rec_token_end_index = -current_wtes_len-REC_END_wtes_len-gt_item_wte_len
        # [batch (1), REC_wtes_len, self.language_model.config.n_embd]
        rec_token_hidden = lm_outputs.hidden_states[-1][:, rec_token_start_index:rec_token_end_index, :]
        # [batch (1), self.recall_lm_query_mapper.out_features]
        rec_query_vector = self.recall_lm_query_mapper(rec_token_hidden).squeeze(1)
        
        # sample num_samples item ids to train recall with "recommendation as classification"
        sampled_item_ids = sample_ids_from_db(self.items_db, gt_item_id, num_samples, include_gt=True)
        gt_item_id_index = sampled_item_ids.index(gt_item_id)
        
        # [num_samples, self.item_encoder.config.hidden_size]
        encoded_items_embeddings, _ = self.compute_encoded_embeddings_for_items(self.recall_encoder, sampled_item_ids, self.items_db, None)
        # to compute dot product with rec_query_vector
        items_key_vectors = self.recall_item_wte_mapper(encoded_items_embeddings) # [num_samples, self.recall_item_wte_mapper.out_features]
        expanded_rec_query_vector = rec_query_vector.expand(items_key_vectors.shape[0], rec_query_vector.shape[1]) # [num_samples, self.recall_item_wte_mapper.out_features]
        recall_logits = torch.sum(expanded_rec_query_vector * items_key_vectors, dim=1) # torch.size([num_samples])
        
        # REC_TOKEN prediction and future sentence prediction
        # hidden rep of the token that's right before REC_TOKEN
        token_before_REC_logits = lm_outputs.logits[:, rec_token_start_index-1:rec_token_end_index-1, :]
        REC_targets = self.lm_tokenizer(self.REC_TOKEN_STR, return_tensors="pt")['input_ids'].to(self.device) # [1, 1]
        
        #language logits and targets
        current_language_logits = lm_outputs.logits[:, -current_wtes_len:-1, :]
        current_language_targets = current_tokens[:,1:]
        
        # REC token and language, their logits and targets
        # [batch, current_wtes_len+REC_wtes_len, lm_vocab]
        all_wte_logits = torch.cat((token_before_REC_logits, current_language_logits), dim=1)
        # [current_wtes_len+REC_wtes_len, lm_vocab]
        all_wte_targets = torch.cat((REC_targets, current_language_targets), dim=1)
        
        # torch.size([num_samples]), id, [batch, current_wtes_len+REC_wtes_len, lm_vocab], [current_wtes_len+REC_wtes_len, lm_vocab]
        return recall_logits, gt_item_id_index, all_wte_logits, all_wte_targets
        
    def forward_rerank(self,
                       past_wtes, # past word token embeddings, [1, len, 768]
                       gt_item_id, # tokens of current turn conversation, [1, len]
                       num_samples, # num examples to sample for training, including groud truth id
                       rerank_items_chunk_size=10, # batch size for encoder GPU computation
                        # overall entity
                      ):    
        # REC wte
        REC_wtes = self.get_rec_token_wtes() # [batch (1), 1, self.language_model.config.n_embd]
        # print(f'Overall entity: {overall_entity}')
        
        #  items wtes to compute rerank loss
        # sample rerank examples
        sampled_item_ids = sample_ids_from_db(self.items_db, gt_item_id, num_samples, include_gt=True)
        gt_item_id_index = sampled_item_ids.index(gt_item_id)
        # compute item wtes by batch
        num_chunks = math.ceil(len(sampled_item_ids) / rerank_items_chunk_size)
        total_wtes = []
        for i in range(num_chunks):
            chunk_ids = sampled_item_ids[i*rerank_items_chunk_size: (i+1)*rerank_items_chunk_size]
            chunk_pooled, _ = self.compute_encoded_embeddings_for_items(self.item_encoder, chunk_ids, self.items_db, None) # [rerank_items_chunk_size, self.item_encoder.config.hidden_size]
            chunk_wtes = self.rerank_item_wte_mapper(chunk_pooled)
            total_wtes.append(chunk_wtes)
        total_wtes = torch.cat(total_wtes, dim=0) # [num_samples, self.language_model.config.n_embd]
        
        past_wtes_len = past_wtes.shape[1]
        REC_wtes_len = REC_wtes.shape[1] # 1 by default
        total_wtes_len = total_wtes.shape[0]
        
        # compute positional ids, all rerank item wte should have the same positional encoding id 0
        position_ids = torch.arange(0, past_wtes_len + REC_wtes_len, dtype=torch.long, device=self.device)
        items_position_ids = torch.zeros(total_wtes.shape[0], dtype=torch.long, device=device)
#         items_position_ids = torch.tensor([1023] * total_wtes.shape[0], dtype=torch.long, device=device)
        combined_position_ids = torch.cat((position_ids, items_position_ids), dim=0)
        combined_position_ids = combined_position_ids.unsqueeze(0) # [1, past_wtes_len+REC_wtes_len+total_wtes_len]
        
        # compute concatenated lm wtes
        lm_wte_inputs = torch.cat(
            (past_wtes, # [batch (1), len, self.language_model.config.n_embd]
             REC_wtes, # [batch (1), 1, self.language_model.config.n_embd]
             total_wtes.unsqueeze(0), # [1, num_samples, self.language_model.config.n_embd]
            ),
            dim=1
        ) # [1, past_len + REC_wtes_len + num_samples, self.language_model.config.n_embd]

        # trim sequence to smaller length (len < self.language_model.config.n_positions-self.lm_trim_offset)
        combined_position_ids_trimmed = self.trim_positional_ids(combined_position_ids, total_wtes_len) # [1, len]
        lm_wte_inputs_trimmed = self.trim_lm_wtes(lm_wte_inputs) # [1, len, self.language_model.config.n_embd]
        assert(combined_position_ids.shape[1] == lm_wte_inputs.shape[1])
        
        # compute inductive attention mask
        #     Order of recommended items shouldn't affect their score, thus every item 
        # should have full attention over the entire sequence: they should know each other and the entire
        # conversation history
        inductive_attention_mask = self.compute_inductive_attention_mask(
            lm_wte_inputs_trimmed.shape[1]-total_wtes.shape[0], 
            total_wtes.shape[0]
        )
        rerank_lm_outputs = self.language_model(inputs_embeds=lm_wte_inputs_trimmed,
                  inductive_attention_mask=inductive_attention_mask,
                  position_ids=combined_position_ids_trimmed,
                  output_hidden_states=True)
        
        rerank_lm_hidden = rerank_lm_outputs.hidden_states[-1][:, -total_wtes.shape[0]:, :]
        rerank_logits = self.rerank_logits_mapper(rerank_lm_hidden).squeeze() # torch.Size([num_samples])
        
        return rerank_logits, gt_item_id_index
    
    def validation_perform_recall(self, past_wtes, topk):
        REC_wtes = self.get_rec_token_wtes()
        lm_wte_inputs = torch.cat(
            (past_wtes, # [batch (1), len, self.language_model.config.n_embd]
             REC_wtes # [1, 1, self.language_model.config.n_embd]
            ),
            dim=1
        )
        lm_wte_inputs = self.trim_lm_wtes(lm_wte_inputs) # trim for len > self.language_model.config.n_positions
        lm_outputs = self.language_model(inputs_embeds=lm_wte_inputs, output_hidden_states=True)
        
        rec_token_hidden = lm_outputs.hidden_states[-1][:, -1, :]
        # [batch (1), self.recall_lm_query_mapper.out_features]
        rec_query_vector = self.recall_lm_query_mapper(rec_token_hidden).squeeze(0) # [768]
        rec_query_vector = rec_query_vector.cpu().detach().numpy()
        recall_results = self.annoy_base_recall.get_nns_by_vector(rec_query_vector, topk)
        return recall_results
    
    def validation_perform_rerank(self, past_wtes, recalled_ids):
        REC_wtes = self.get_rec_token_wtes()
        
        total_wtes = [ self.annoy_base_rerank.get_item_vector(r_id) for r_id in recalled_ids]
        total_wtes = [ torch.tensor(wte).reshape(-1, self.language_model.config.n_embd).to(self.device) for wte in total_wtes]
        total_wtes = torch.cat(total_wtes, dim=0) # [len(recalled_ids), 768]
        
        past_wtes_len = past_wtes.shape[1]
        REC_wtes_len = REC_wtes.shape[1] # 1 by default
        total_wtes_len = total_wtes.shape[0]
        
        # compute positional ids, all rerank item wte should have the same positional encoding id 0
        position_ids = torch.arange(0, past_wtes_len + REC_wtes_len, dtype=torch.long, device=self.device)
        items_position_ids = torch.zeros(total_wtes.shape[0], dtype=torch.long, device=device)
        combined_position_ids = torch.cat((position_ids, items_position_ids), dim=0)
        combined_position_ids = combined_position_ids.unsqueeze(0) # [1, past_wtes_len+REC_wtes_len+total_wtes_len]
        
        # compute concatenated lm wtes
        lm_wte_inputs = torch.cat(
            (past_wtes, # [batch (1), len, self.language_model.config.n_embd]
             REC_wtes, # [batch (1), 1, self.language_model.config.n_embd]
             total_wtes.unsqueeze(0), # [1, num_samples, self.language_model.config.n_embd]
            ),
            dim=1
        ) # [1, past_len + REC_wtes_len + num_samples, self.language_model.config.n_embd]

        # trim sequence to smaller length (len < self.language_model.config.n_positions-self.lm_trim_offset)
        combined_position_ids_trimmed = self.trim_positional_ids(combined_position_ids, total_wtes_len) # [1, len]
        lm_wte_inputs_trimmed = self.trim_lm_wtes(lm_wte_inputs) # [1, len, self.language_model.config.n_embd]
        assert(combined_position_ids.shape[1] == lm_wte_inputs.shape[1])
        
        inductive_attention_mask = self.compute_inductive_attention_mask(
            lm_wte_inputs_trimmed.shape[1]-total_wtes.shape[0], 
            total_wtes.shape[0]
        )
        rerank_lm_outputs = self.language_model(inputs_embeds=lm_wte_inputs_trimmed,
                  inductive_attention_mask=inductive_attention_mask,
                  position_ids=combined_position_ids_trimmed,
                  output_hidden_states=True)
        
        rerank_lm_hidden = rerank_lm_outputs.hidden_states[-1][:, -total_wtes.shape[0]:, :]
        rerank_logits = self.rerank_logits_mapper(rerank_lm_hidden).squeeze()
        
        return rerank_logits
    

# In[8]:

device = torch.device(0)
model = UniversalCRSModel(
    gpt2_model, 
    bert_model_recall, 
    bert_model_rerank, 
    gpt_tokenizer, 
    bert_tokenizer, 
    device, 
    items_db, 
    rec_token_str=REC_TOKEN, 
    rec_end_token_str=REC_END_TOKEN
)
# model.load_state_dict(torch.load())

model.to(device)
pass


# In[9]:


start = time.time()
model.annoy_base_constructor()
end = time.time()
print(end-start)
# model.annoy_base_recall.save('/local-scratch1/data/by2299/INITIAL_RECALL_ANNOY_BASE_REDIAL_TRAIN_BERT_DISTIL_MULTIPLE.ann')
# model.annoy_base_rerank.save('/local-scratch1/data/by2299/INITIAL_RERANK_ANNOY_BASE_REDIAL_TRAIN_BERT_DISTIL_MULTIPLE.ann')


# In[10]:


# parameters
batch_size = 1
num_epochs = 20
num_gradients_accumulation = 1
num_train_optimization_steps = len(train_dataset) * num_epochs // batch_size // num_gradients_accumulation

num_samples_recall_train = 100
num_samples_rerank_train = 150
rerank_encoder_chunk_size = int(num_samples_rerank_train / 15)
validation_recall_size = 500

temperature = 1.2

language_loss_train_coeff = 0.15
language_loss_train_coeff_beginnging_turn = 1.0
recall_loss_train_coeff = 0.8
rerank_loss_train_coeff = 1.0

# loss
criterion_language = SequenceCrossEntropyLoss()
criterion_recall = torch.nn.CrossEntropyLoss()
rerank_class_weights = torch.FloatTensor([1] * (num_samples_rerank_train-1) + [30]).to(model.device)
criterion_rerank_train = torch.nn.CrossEntropyLoss(weight=rerank_class_weights)

# optimizer and scheduler
param_optimizer = list(model.language_model.named_parameters()) +     list(model.recall_encoder.named_parameters()) +     list(model.item_encoder.named_parameters()) +     list(model.recall_lm_query_mapper.named_parameters()) +     list(model.recall_item_wte_mapper.named_parameters()) +     list(model.rerank_item_wte_mapper.named_parameters()) +     list(model.rerank_logits_mapper.named_parameters()) +     list(model.entity_encoder.named_parameters()) +     list(model.entity_mapper.named_parameters()) + list(model.fusion_gate.named_parameters())

no_decay = ['bias', 'ln', 'LayerNorm.weight']
optimizer_grouped_parameters = [
    {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)], 'weight_decay': 0.001},
    {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
    ]

optimizer = AdamW(optimizer_grouped_parameters, 
                  lr=3e-5,
                  eps=1e-06)

scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=len(train_dataset) // num_gradients_accumulation , num_training_steps = num_train_optimization_steps)

update_count = 0
progress_bar = tqdm.tqdm
start = time.time()


# In[23]:


def past_wtes_constructor(past_list, model):
    past_wtes = []
    # print(f'past_list: {past_list}')
    for language_tokens, recommended_ids, overall_entity in past_list:
        # print(f'recommended_ids: {recommended_ids}')
        if language_tokens == None and (recommended_ids != None and recommended_ids != []): # rec turn
            # append REC, gt_item_wte, REC_END
            REC_wte = model.get_rec_token_wtes() # [1, 1, 768]
            gt_item_wte, _ = model.compute_encoded_embeddings_for_items(
                model.item_encoder,
                recommended_ids, 
                model.items_db,
                None
            ) # [1, 768]
            gt_item_wte = model.rerank_item_wte_mapper(gt_item_wte)
            
            REC_END_wte = model.get_rec_end_token_wtes() # [1, 1, 768]
            combined_wtes = torch.cat(
                (REC_wte,
                 gt_item_wte.unsqueeze(0), # [1, 1, 768]
                 REC_END_wte
                ), 
                dim=1
            ) # [1, 3, 768]
            past_wtes.append(combined_wtes)
        elif (recommended_ids == None or recommended_ids == []) and language_tokens != None: # language turn simply append wtes
            wtes = model.language_model.transformer.wte(language_tokens) # [1, len, 768]
            past_wtes.append(wtes)
        elif (recommended_ids != None and recommended_ids != []) and language_tokens != None: # user mentioned turn
            l_wtes = model.language_model.transformer.wte(language_tokens)
            
            SEP_wte = model.get_sep_token_wtes()
            
            gt_item_wte, _ = model.compute_encoded_embeddings_for_items(
                model.item_encoder,
                recommended_ids, 
                model.items_db,
                None
            ) # [1, 768]
            gt_item_wte = model.rerank_item_wte_mapper(gt_item_wte)
            SEP_wte = model.get_sep_token_wtes()
            combined_wtes = torch.cat(
                (l_wtes,
                 SEP_wte,
                 gt_item_wte.unsqueeze(0), # [1, 1, 768]
                 SEP_wte
                ), 
                dim=1
            )
            past_wtes.append(combined_wtes)
            
    
    past_wtes = torch.cat(past_wtes, dim=1)
    # don't trim since we already dealt with length in model functions
    return past_wtes

def train_one_iteration(batch, model):
    role_ids, dialogues = batch
    dialog_tensors = [torch.LongTensor(utterance).to(model.device) for utterance, _, _ in dialogues]
    
    past_list = []
    ppl_history = []
#     language_logits, language_targets = [], []
    for turn_num in range(len(role_ids)):
        current_tokens = dialog_tensors[turn_num]
        _, recommended_ids, overall_entity = dialogues[turn_num]
        # print(f'dialogues: {dialogues}')
        # print(f'initialized recommended_ids: {recommended_ids}')
        # if turn_num == 3:
        #     print(sd)
        
        if past_list == []:
            past_list.append((current_tokens, recommended_ids, overall_entity))
            continue
        
        if recommended_ids == None: # no rec
            if role_ids[turn_num] == 0: # user
                past_list.append((current_tokens, None, overall_entity))
            else: #system
                past_wtes = past_wtes_constructor(past_list, model)
                language_logits, language_targets = model.forward_pure_language_turn(past_wtes, current_tokens)
                
                # loss backward
                language_targets_mask = torch.ones_like(language_targets).float()
                loss_ppl = criterion_language(language_logits, language_targets, language_targets_mask, label_smoothing=0.02, reduce="batch")
                loss_ppl = language_loss_train_coeff * loss_ppl
                loss_ppl.backward()
                perplexity = np.exp(loss_ppl.item())
                ppl_history.append(perplexity)
                
                # append to past list
                past_list.append((current_tokens, None, overall_entity))
        else: # rec!
            
            if role_ids[turn_num] == 0: #user mentioned
                past_list.append((current_tokens, recommended_ids, overall_entity))
                continue
            for recommended_id in recommended_ids:
                #system recommend turn
                past_wtes = past_wtes_constructor(past_list, model)

                # recall
                recall_logits, recall_true_index, all_wte_logits, all_wte_targets = model.forward_recall(
                    past_wtes, 
                    current_tokens, 
                    recommended_id, 
                    num_samples_recall_train,
                    overall_entity
                )
                
                # recall items loss
                recall_targets = torch.LongTensor([recall_true_index]).to(model.device)
                loss_recall = criterion_recall(recall_logits.unsqueeze(0), recall_targets)

                # language loss in recall turn, REC_TOKEN, Language on conditional generation
                all_wte_targets_mask = torch.ones_like(all_wte_targets).float()
                loss_ppl = criterion_language(all_wte_logits, all_wte_targets, all_wte_targets_mask, label_smoothing=0.02, reduce="batch")
                perplexity = np.exp(loss_ppl.item())
                ppl_history.append(perplexity)

                # combined loss
                recall_total_loss = loss_recall * recall_loss_train_coeff + loss_ppl * language_loss_train_coeff
                recall_total_loss.backward()

                # rerank
                past_wtes = past_wtes_constructor(past_list, model)
                rerank_logits, rerank_true_index = model.forward_rerank(
                    past_wtes, 
                    recommended_id, 
                    num_samples_rerank_train,
                    rerank_encoder_chunk_size
                )
                
                rerank_logits /= temperature

                # rerank loss 
                rerank_targets = torch.LongTensor([rerank_true_index]).to(model.device)
                loss_rerank = criterion_rerank_train(rerank_logits.unsqueeze(0), rerank_targets)
                loss_rerank *= rerank_loss_train_coeff
                loss_rerank.backward()

            past_list.append((None, recommended_ids, overall_entity))
            past_list.append((current_tokens, None, overall_entity))
    return np.mean(ppl_history)


def calculate_ndcg(ranks, k):
    """
    计算NDCG@k
    ranks: 正确item的排名位置(从0开始)
    k: 截断位置
    """
    if ranks >= k:
        return 0
    return 1 / np.log2(ranks + 2)  # +2是因为rank从0开始,log2(1)=0

def validate_one_iteration(batch, model):
    role_ids, dialogues = batch
    dialog_tensors = [torch.LongTensor(utterance).to(model.device) for utterance, _, _ in dialogues]
    
    past_list = []
    ppl_history = []
    recall_loss_history = []
    rerank_loss_history = []
    total = 0
    recall_top100, recall_top300, recall_top500 = 0, 0, 0
    rerank_top1, rerank_top10, rerank_top50 = 0, 0, 0
    ndcg_10, ndcg_50 = 0, 0  # 新增NDCG指标
    
    for turn_num in range(len(role_ids)):
        current_tokens = dialog_tensors[turn_num]
        _, recommended_ids, overall_entity = dialogues[turn_num]
        
        if past_list == []:
            past_list.append((current_tokens, None, overall_entity))
            continue
        
        if recommended_ids == None: # no rec
            if role_ids[turn_num] == 0: # user
                past_list.append((current_tokens, None, overall_entity))
            else: #system
                past_wtes = past_wtes_constructor(past_list, model)
                language_logits, language_targets = model.forward_pure_language_turn(past_wtes, current_tokens)
                
                # loss backward
                language_targets_mask = torch.ones_like(language_targets).float()
                loss_ppl = criterion_language(language_logits, language_targets, language_targets_mask, label_smoothing=-1, reduce="sentence")
                perplexity = np.exp(loss_ppl.item())
                ppl_history.append(perplexity)
                del loss_ppl
                
                # append to past list
                past_list.append((current_tokens, None, overall_entity))
        else: # rec!
            
            if role_ids[turn_num] == 0: #user mentioned
                past_list.append((current_tokens, recommended_ids, overall_entity))
                continue
            for recommended_id in recommended_ids:
                past_wtes = past_wtes_constructor(past_list, model)

                total += 1

                # recall
                recall_logits, recall_true_index, all_wte_logits, all_wte_targets = model.forward_recall(
                    past_wtes, 
                    current_tokens, 
                    recommended_id, 
                    num_samples_recall_train,
                    overall_entity
                )

                # recall items loss
                recall_targets = torch.LongTensor([recall_true_index]).to(model.device)
                loss_recall = criterion_recall(recall_logits.unsqueeze(0), recall_targets)
                recall_loss_history.append(loss_recall.item())
                del loss_recall; del recall_logits; del recall_targets

                # language loss in recall turn, REC_TOKEN, Language on conditional generation
                all_wte_targets_mask = torch.ones_like(all_wte_targets).float()
                loss_ppl = criterion_language(all_wte_logits, all_wte_targets, all_wte_targets_mask, label_smoothing=-1, reduce="sentence")
                perplexity = np.exp(loss_ppl.item())
                ppl_history.append(perplexity)
                del loss_ppl; del all_wte_logits; del all_wte_targets

                recalled_ids = model.validation_perform_recall(past_wtes, validation_recall_size)

                if recommended_id in recalled_ids[:500]:
                    recall_top500 += 1
                if recommended_id in recalled_ids[:400]:
                    recall_top300 += 1
                if recommended_id in recalled_ids[:300]:
                    recall_top100 += 1

                if recommended_id not in recalled_ids:
                    continue # no need to compute rerank since recall is unsuccessful

                # rerank
                past_wtes = past_wtes_constructor(past_list, model)
                rerank_true_index = recalled_ids.index(recommended_id)
                rerank_logits = model.validation_perform_rerank(past_wtes, recalled_ids)
                
                # 计算NDCG
                reranks = np.argsort(rerank_logits.cpu().detach().numpy())
                if rerank_true_index in reranks[-50:]:
                    rerank_top50 += 1
                if rerank_true_index in reranks[-10:]:
                    rerank_top10 += 1
                if rerank_true_index in reranks[-1:]:
                    rerank_top1 += 1
                rank_position = len(reranks) - 1 - np.where(reranks == rerank_true_index)[0][0]
                ndcg_10 += calculate_ndcg(rank_position, 10)
                ndcg_50 += calculate_ndcg(rank_position, 50)
                
                rerank_targets = torch.LongTensor([rerank_true_index]).to(model.device)
                rerank_loss_val = torch.nn.CrossEntropyLoss()
                loss_rerank = rerank_loss_val(rerank_logits.unsqueeze(0), rerank_targets)
                rerank_loss_history.append(loss_rerank.item())
                del loss_rerank; del rerank_logits; del rerank_targets
            
            past_list.append((None, recommended_ids, overall_entity))
            past_list.append((current_tokens, None, overall_entity))
    return ppl_history, recall_loss_history, rerank_loss_history,             total, recall_top100, recall_top300, recall_top500,             rerank_top1, rerank_top10, rerank_top50,             ndcg_10, ndcg_50


# In[24]:


def distinct_metrics(outs):
    # outputs is a list which contains several sentences, each sentence contains several words
    unigram_count = 0
    bigram_count = 0
    trigram_count=0
    quagram_count=0
    unigram_set = set()
    bigram_set = set()
    trigram_set=set()
    quagram_set=set()
    for sen in outs:
        for word in sen:
            unigram_count += 1
            unigram_set.add(word)
        for start in range(len(sen) - 1):
            bg = str(sen[start]) + ' ' + str(sen[start + 1])
            bigram_count += 1
            bigram_set.add(bg)
        for start in range(len(sen)-2):
            trg=str(sen[start]) + ' ' + str(sen[start + 1]) + ' ' + str(sen[start + 2])
            trigram_count+=1
            trigram_set.add(trg)
        for start in range(len(sen)-3):
            quag=str(sen[start]) + ' ' + str(sen[start + 1]) + ' ' + str(sen[start + 2]) + ' ' + str(sen[start + 3])
            quagram_count+=1
            quagram_set.add(quag)
    dis1 = len(unigram_set) / len(outs)#unigram_count
    dis2 = len(bigram_set) / len(outs)#bigram_count
    dis3 = len(trigram_set)/len(outs)#trigram_count
    dis4 = len(quagram_set)/len(outs)#quagram_count
    return dis1, dis2, dis3, dis4


# In[25]:
def validate_language_metrics_batch(batch, model, item_id_2_lm_token_id):
    role_ids, dialogues = batch
    dialog_tensors = [torch.LongTensor(utterance).to(model.device) for utterance, _, _ in dialogues]
    
#     past_list = []
    past_tokens = None
    original_sentences = []
    tokenized_sentences = []
    integration_total, integration_cnt = 0, 0
    valid_gen_selected_cnt = 0; total_gen_cnt = 0; response_with_items = 0; original_response_with_items = 0
    
    for turn_num in range(len(role_ids)):
        dial_turn_inputs = dialog_tensors[turn_num]
        _, recommended_ids, overall_entity = dialogues[turn_num]
        
        item_ids = []; item_titles = []
        if recommended_ids != None:
            for r_id in recommended_ids:
                item_ids.append(item_id_2_lm_token_id[r_id])
                title = model.items_db[r_id]
                title = title.split('[SEP]')[0].strip()
                item_titles.append(title)
            item_ids = torch.tensor([item_ids]).to(device)
        
#         if turn_num == 0:
#             past_tokens = dial_turn_inputs
        if role_ids[turn_num] == 0:
            if turn_num == 0:
                past_tokens = dial_turn_inputs
            elif turn_num != 0:
                past_tokens = torch.cat((past_tokens, dial_turn_inputs), dim=1)
        else:
            if turn_num != 0:
                if item_ids != []:
                    rec_start_token = model.lm_tokenizer(model.REC_TOKEN_STR, return_tensors="pt")["input_ids"].to(model.device)
                    rec_end_token = model.lm_tokenizer(model.REC_END_TOKEN_STR, return_tensors="pt")["input_ids"].to(model.device)
                    past_tokens = torch.cat((past_tokens, rec_start_token, item_ids, rec_end_token), dim=1)
                else:
                    past_tokens = past_tokens
                
                total_len = past_tokens.shape[1]
                if total_len >= 1024: break

                original_sen = gpt_tokenizer.decode(dial_turn_inputs[0], skip_special_tokens=True)


                generated = model.language_model.generate(
                    input_ids= torch.cat((past_tokens, torch.tensor([[32, 25]]).to(device)), dim=1),
                    max_length=1024,
                    num_return_sequences=5,
                    do_sample=True,
                    num_beams=5,
                    top_k=50,
                    temperature=1.25,
                    eos_token_id=628,
                    pad_token_id=628,
                    output_scores=True,
                    return_dict_in_generate=True
#                 no_repeat_ngram_size=3,
#                         length_penalty=3.0

                )
                # check valid generations, equal num [MOVIE_ID] placeholders
                total_gen_cnt += 1
                valid_gens = []; valid_gens_scores = []
                final_gen = None
                if len(item_ids) == 0: # no rec items
                    for i in range(len(generated.sequences)):
                        gen_sen = gpt_tokenizer.decode(generated.sequences[i][past_tokens.shape[1]:], skip_special_tokens=True)
                        if gen_sen.count("[MOVIE_ID]") == 0:
                            valid_gens.append(gen_sen); valid_gens_scores.append(generated.sequences_scores[i].item())
                    if valid_gens == [] and valid_gens_scores == []: # no valid, pick with highest score
                        i = torch.argmax(generated.sequences_scores).item()
                        final_gen = gpt_tokenizer.decode(generated.sequences[i][past_tokens.shape[1]:], skip_special_tokens=True)
                    else: # yes valid
                        i = np.argmax(valid_gens_scores)
                        final_gen = valid_gens[i]
                        valid_gen_selected_cnt += 1
                else:
                    original_response_with_items += 1
                    for i in range(len(generated.sequences)):
                        gen_sen = gpt_tokenizer.decode(generated.sequences[i][past_tokens.shape[1]:], skip_special_tokens=True)
                        if gen_sen.count("[MOVIE_ID]") == original_sen.count("[MOVIE_ID]"):
                            valid_gens.append(gen_sen); valid_gens_scores.append(generated.sequences_scores[i].item())
                    if valid_gens == [] and valid_gens_scores == []: # no valid, pick with highest score
                        i = torch.argmax(generated.sequences_scores).item()
                        final_gen = gpt_tokenizer.decode(generated.sequences[i][past_tokens.shape[1]:], skip_special_tokens=True)
                        if "[MOVIE_ID]" in final_gen:
                            response_with_items += 1
                        final_gen = replace_placeholder(final_gen, item_titles)
                    else:
                        i = np.argmax(valid_gens_scores)
                        final_gen = valid_gens[i]
                        if "[MOVIE_ID]" in final_gen:
                            response_with_items += 1
                        final_gen = replace_placeholder(final_gen, item_titles)
                        valid_gen_selected_cnt += 1

    #             generated_sen =  gpt_tokenizer.decode(generated[0][past_tokens.shape[1]:], skip_special_tokens=True)
    #             print("Generated Rec: " + final_gen)
                tokenized_sen = final_gen.strip().split(' ')
                tokenized_sentences.append(tokenized_sen)
                original_sen = replace_placeholder(original_sen, item_titles).replace("\n\n\n", "")
    #             print("Original Rec: " + original_sen)
                original_sentences.append( original_sen.strip().split(' ') )
                if recommended_ids != None:
                    integration_total += 1                        
                    if "[MOVIE_ID]" in final_gen:
                        integration_cnt += 1

                if turn_num != 0:
                    past_tokens = torch.cat((past_tokens, dial_turn_inputs), dim=1)
            elif turn_num == 0:
                original_sen = gpt_tokenizer.decode(dial_turn_inputs[0], skip_special_tokens=True)


                generated = model.language_model.generate(
                    input_ids= torch.tensor([[32, 25]]).to(device),
                    max_length=1024,
                    num_return_sequences=5,
                    do_sample=True,
                    num_beams=5,
                    top_k=50,
                    temperature=1.25,
                    eos_token_id=628,
                    pad_token_id=628,
#                 no_repeat_ngram_size=3,
                    output_scores=True,
                    return_dict_in_generate=True
                )
                # check valid generations, equal num [MOVIE_ID] placeholders
                total_gen_cnt += 1
                valid_gens = []; valid_gens_scores = []
                final_gen = None
                if len(item_ids) == 0: # no rec items
                    for i in range(len(generated.sequences)):
                        gen_sen = gpt_tokenizer.decode(generated.sequences[i], skip_special_tokens=True)
                        if gen_sen.count("[MOVIE_ID]") == 0:
                            valid_gens.append(gen_sen); valid_gens_scores.append(generated.sequences_scores[i].item())
                    if valid_gens == [] and valid_gens_scores == []: # no valid, pick with highest score
                        i = torch.argmax(generated.sequences_scores).item()
                        final_gen = gpt_tokenizer.decode(generated.sequences[i], skip_special_tokens=True)
                    else: # yes valid
                        i = np.argmax(valid_gens_scores)
                        final_gen = valid_gens[i]
                        valid_gen_selected_cnt += 1
                else:
                    original_response_with_items += 1
                    for i in range(len(generated.sequences)):
                        gen_sen = gpt_tokenizer.decode(generated.sequences[i], skip_special_tokens=True)
                        if gen_sen.count("[MOVIE_ID]") == original_sen.count("[MOVIE_ID]"):
                            valid_gens.append(gen_sen); valid_gens_scores.append(generated.sequences_scores[i].item())
                    if valid_gens == [] and valid_gens_scores == []: # no valid, pick with highest score
                        i = torch.argmax(generated.sequences_scores).item()
                        final_gen = gpt_tokenizer.decode(generated.sequences[i], skip_special_tokens=True)
                        if "[MOVIE_ID]" in final_gen:
                            response_with_items += 1
                        final_gen = replace_placeholder(final_gen, item_titles)
                    else:
                        i = np.argmax(valid_gens_scores)
                        final_gen = valid_gens[i]
                        if "[MOVIE_ID]" in final_gen:
                            response_with_items += 1
                        final_gen = replace_placeholder(final_gen, item_titles)
                        valid_gen_selected_cnt += 1

    #             generated_sen =  gpt_tokenizer.decode(generated[0][past_tokens.shape[1]:], skip_special_tokens=True)
    #             print("Generated Rec: " + final_gen)
                tokenized_sen = final_gen.strip().split(' ')
                tokenized_sentences.append(tokenized_sen)
                original_sen = replace_placeholder(original_sen, item_titles).replace("\n\n\n", "")
    #             print("Original Rec: " + original_sen)
                original_sentences.append( original_sen.strip().split(' ') )
                if recommended_ids != None:
                    integration_total += 1                        
                    if "[MOVIE_ID]" in final_gen:
                        integration_cnt += 1
                
                if turn_num == 0:
                    past_tokens = dial_turn_inputs
            
    return original_sentences, tokenized_sentences, integration_cnt, integration_total, valid_gen_selected_cnt, total_gen_cnt, response_with_items, original_response_with_items


# def validate_language_metrics_batch(batch, model, item_id_2_lm_token_id):
#     role_ids, dialogues = batch
#     dialog_tensors = [torch.LongTensor(utterance).to(model.device) for utterance, _ in dialogues]
    
#     past_tokens = None
#     original_sentences = []  # 存储原始句子
#     tokenized_sentences = []  # 存储生成的句子
#     integration_total, integration_cnt = 0, 0
    
#     for turn_num in range(len(role_ids)):
#         dial_turn_inputs = dialog_tensors[turn_num]
#         _, recommended_ids = dialogues[turn_num]
        
#         item_ids = []; item_titles = []
#         if recommended_ids != None:
#             for r_id in recommended_ids:
#                 item_ids.append(item_id_2_lm_token_id[r_id])
#                 title = model.items_db[r_id]
#                 title = title.split('[SEP]')[0].strip()
#                 item_titles.append(title)
#             item_ids = torch.tensor([item_ids]).to(device)
        
#         if turn_num == 0:
#             past_tokens = dial_turn_inputs
#         if role_ids[turn_num] == 0:
#             if turn_num != 0:
#                 past_tokens = torch.cat((past_tokens, dial_turn_inputs), dim=1)
#         else:
#             if turn_num != 0:
#                 if item_ids != []:
#                     rec_start_token = model.lm_tokenizer(model.REC_TOKEN_STR, return_tensors="pt")["input_ids"].to(model.device)
#                     rec_end_token = model.lm_tokenizer(model.REC_END_TOKEN_STR, return_tensors="pt")["input_ids"].to(model.device)
#                     past_tokens = torch.cat((past_tokens, rec_start_token, item_ids, rec_end_token), dim=1)
#                 else:
#                     past_tokens = past_tokens
                
#             total_len = past_tokens.shape[1]
#             if total_len >= 1024: break
# #                 print("Original Rec: " + gpt_tokenizer.decode(dial_turn_inputs[0], skip_special_tokens=True))
#             generated = model.language_model.generate(
#                 input_ids= torch.cat((past_tokens, torch.tensor([[32, 25]]).to(device)), dim=1),
#                 max_length=1024,
#                 num_return_sequences=1,
#                 do_sample=True,
#                 num_beams=2,
#                 top_k=50,
#                 temperature=1.05,
#                 eos_token_id=628,
#                 pad_token_id=628,
# #                 no_repeat_ngram_size=3,
# #                         length_penalty=3.0

#             )
#             generated_sen =  gpt_tokenizer.decode(generated[0][past_tokens.shape[1]:], skip_special_tokens=True)
# #                 print("Generated Rec: " + generated_sen)
#             tokenized_sen = generated_sen.strip().split(' ')
#             tokenized_sentences.append(tokenized_sen)
#             if recommended_ids != None:
#                 integration_total += 1                        
#                 if "[MOVIE_ID]" in generated_sen:
#                     integration_cnt += 1
            
#             if turn_num != 0:
#                 past_tokens = torch.cat((past_tokens, dial_turn_inputs), dim=1)
            
#     return tokenized_sentences, integration_cnt, integration_total


# In[14]:

# output_file_path = "Outputs/CRS_Train.txt"
model_saved_path = "/projects/0/prjs1158/KG/redail/MESE/Output"


# In[26]:

early_stop_count = 0
best_ndcg_top10 = 0

for ep in range(num_epochs):

    #"Training"
    pbar = progress_bar(train_dataloader)
    model.train()
    for batch in pbar:
    # for batch in train_dataloader:
        # batch size of train_dataloader is 1
        avg_ppl = train_one_iteration(batch[0], model)
        update_count +=1
        if update_count % num_gradients_accumulation == num_gradients_accumulation - 1:
            
            # update for gradient accumulation
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()
            
            # speed measure
            end = time.time()
            speed = batch_size * num_gradients_accumulation / (end - start)
            start = end
            
            # show progress
            # pbar.set_postfix(ppl=avg_ppl, speed=speed)
            
    model.eval()
    model.annoy_base_constructor()
    
    pbar = progress_bar(valid_dataloader)
    ppls, recall_losses, rerank_losses = [],[],[]
    total_val, recall_top100_val, recall_top300_val, recall_top500_val,         rerank_top1_val, rerank_top10_val, rerank_top50_val = 0,0,0,0,0,0,0
    ndcg_top10_val, ndcg_top50_val = 0, 0
    for batch in pbar:
        ppl_history, recall_loss_history, rerank_loss_history,         total, recall_top100, recall_top300, recall_top500,         rerank_top1, rerank_top10, rerank_top50,             ndcg_10, ndcg_50 = validate_one_iteration(batch[0], model)
        ppls += ppl_history; recall_losses += recall_loss_history; rerank_losses += rerank_loss_history
        total_val += total; 
        recall_top100_val += recall_top100; recall_top300_val += recall_top300; recall_top500_val += recall_top500
        rerank_top1_val += rerank_top1; rerank_top10_val += rerank_top10; rerank_top50_val += rerank_top50
        ndcg_top10_val += ndcg_10; ndcg_top50_val += ndcg_50
        
    item_id_2_lm_token_id = model.lm_expand_wtes_with_items_annoy_base()
    pbar = progress_bar(test_dataloader)
    total_sentences = []
    integration_cnt, total_int_cnt = 0, 0
    for batch in pbar:
        original_sens, sentences, ic, tc, vc, tgc, rwi, group = validate_language_metrics_batch(batch[0], model, item_id_2_lm_token_id)
        for s in sentences:
            total_sentences.append(s)

        integration_cnt += ic; total_int_cnt += tc
    integration_ratio = integration_cnt / total_int_cnt
    dist1, dist2, dist3, dist4 = distinct_metrics(total_sentences)
    model.lm_restore_wtes(original_token_emb_size)

    # output_file = open(output_file_path, 'a')
    print([f"Epcoh {ep} ppl: {np.mean(ppls)}, recall_loss: {np.mean(recall_losses)}, rerank_loss: {np.mean(rerank_losses)}"])      
    print([f"recall top100: {recall_top100_val/total_val}, top300: {recall_top300_val/total_val}, top500: {recall_top500_val/total_val}"])
    print([f"rerank top1: {rerank_top1_val/total_val}, top10: {rerank_top10_val/total_val}, top50: {rerank_top50_val/total_val}"])
    print([f"NDCG@10: {ndcg_top10_val/total_val}, NDCG@50: {ndcg_top50_val/total_val}"])
    print([f"Integration Ratio: {integration_ratio}"])
    print([f"Dist1: {dist1}, Dist2: {dist2}, Dist3: {dist3}, Dist4: {dist4}"])
    

    eval_metric = ndcg_top10_val/total_val
    if eval_metric > best_ndcg_top10:
        best_ndcg_top10 = eval_metric
        early_stop_count = 0
        print(f'save model at epoch {ep}')
        torch.save(model.state_dict(), model_saved_path + "best.pt")
        # test

        print(f'start testing in epoch {ep}')
        model.eval()
        model.annoy_base_constructor()
        pbar = progress_bar(test_dataloader)
        ppls, recall_losses, rerank_losses = [],[],[]
        total_val, recall_top100_val, recall_top300_val, recall_top500_val,         rerank_top1_val, rerank_top10_val, rerank_top50_val = 0,0,0,0,0,0,0
        ndcg_top10_val, ndcg_top50_val = 0, 0
        for batch in pbar:
            ppl_history, recall_loss_history, rerank_loss_history,         total, recall_top100, recall_top300, recall_top500,         rerank_top1, rerank_top10, rerank_top50,             ndcg_10, ndcg_50 = validate_one_iteration(batch[0], model)
            ppls += ppl_history; recall_losses += recall_loss_history; rerank_losses += rerank_loss_history
            total_val += total; 
            recall_top100_val += recall_top100; recall_top300_val += recall_top300; recall_top500_val += recall_top500
            rerank_top1_val += rerank_top1; rerank_top10_val += rerank_top10; rerank_top50_val += rerank_top50
            ndcg_top10_val += ndcg_10; ndcg_top50_val += ndcg_50

        item_id_2_lm_token_id = model.lm_expand_wtes_with_items_annoy_base()
        pbar = progress_bar(test_dataloader)
        total_sentences_original = []  # 存储原始句子
        total_sentences_generated = []  # 存储生成的句子
        integration_cnt, total_int_cnt = 0, 0
        for batch in pbar:
            original_sens, sentences, ic, tc, vc, tgc, rwi, group = validate_language_metrics_batch(batch[0], model, item_id_2_lm_token_id)
            total_sentences_original.extend(original_sens)
            total_sentences_generated.extend(sentences)
            integration_cnt += ic
            total_int_cnt += tc

        # 计算各项指标
        integration_ratio = integration_cnt / total_int_cnt
        
        # 计算 intra-distinct 指标
        intra_dist1, intra_dist2, intra_dist3, intra_dist4 = intra_distinct_metrics(total_sentences_generated)
        
        # 计算 inter-distinct 指标
        inter_dist1, inter_dist2, inter_dist3, inter_dist4 = inter_distinct_metrics(total_sentences_generated)
        
        # 计算 BLEU 分数
        bleu1, bleu2, bleu3, bleu4 = bleu_calc_all(total_sentences_original, total_sentences_generated)
        
        # 计算 ROUGE 分数
        rouge1, rouge2, rougeL = rouge_calc_all(total_sentences_original, total_sentences_generated)
        
        model.lm_restore_wtes(original_token_emb_size)

        # 打印所有指标
        print([f"Epoch {ep} ppl: {np.mean(ppls)}, recall_loss: {np.mean(recall_losses)}, rerank_loss: {np.mean(rerank_losses)}"])      
        print([f"recall top100: {recall_top100_val/total_val}, top300: {recall_top300_val/total_val}, top500: {recall_top500_val/total_val}"])
        print([f"rerank top1: {rerank_top1_val/total_val}, top10: {rerank_top10_val/total_val}, top50: {rerank_top50_val/total_val}"])
        print([f"NDCG@10: {ndcg_top10_val/total_val}, NDCG@50: {ndcg_top50_val/total_val}"])
        print([f"Integration Ratio: {integration_ratio}"])
        print([f"Intra-Distinct@1: {intra_dist1}, Intra-Distinct@2: {intra_dist2}, Intra-Distinct@3: {intra_dist3}, Intra-Distinct@4: {intra_dist4}"])
        print([f"Inter-Distinct@1: {inter_dist1}, Inter-Distinct@2: {inter_dist2}, Inter-Distinct@3: {inter_dist3}, Inter-Distinct@4: {inter_dist4}"])
        print([f"BLEU@1: {bleu1}, BLEU@2: {bleu2}, BLEU@3: {bleu3}, BLEU@4: {bleu4}"])
        print([f"ROUGE@1: {rouge1}, ROUGE@2: {rouge2}, ROUGE@L: {rougeL}"])


    else:
        early_stop_count += 1

    if early_stop_count >= 4:
        print(f'early stop at epoch {ep}')
        break
    



# In[12]:




# In[14]:

# load the best model
# model
model.load_state_dict(torch.load(model_saved_path + "best.pt"))

model.eval()
model.annoy_base_constructor()


item_id_2_lm_token_id = model.lm_expand_wtes_with_items_annoy_base()
pbar = progress_bar(valid_dataloader)
total_sentences_original = []; total_sentences_generated = []
integration_cnt, total_int_cnt = 0, 0
valid_cnt, total_gen_cnt, response_with_items = 0, 0, 0
for batch in pbar:
    original_sens, sentences, ic, tc, vc, tgc, rwi, group = validate_language_metrics_batch(batch[0], model, item_id_2_lm_token_id)
    for s in original_sens:
        total_sentences_original.append(s)
    for s in sentences:
        total_sentences_generated.append(s)
    # total_sentences_original.append(original_sens)
    # total_sentences_generated.append(sentences)
    
    integration_cnt += ic; total_int_cnt += tc
    valid_cnt += vc; total_gen_cnt += tgc; response_with_items += rwi
integration_ratio = integration_cnt / total_int_cnt
valid_gen_ratio = valid_cnt / total_gen_cnt
model.lm_restore_wtes(original_token_emb_size)


# # In[15]:


# valid_cnt / total_gen_cnt, response_with_items / total_gen_cnt


# # In[17]:


# torch.save(total_sentences_generated, '../human_eval/mese2.pt')


# # In[22]:


# valid_cnt / total_gen_cnt, response_with_items / total_gen_cnt


# # In[23]:


intra_dist1, intra_dist2, intra_dist3, intra_dist4 = intra_distinct_metrics(total_sentences_generated)
# 计算 inter-distinct 指标
inter_dist1, inter_dist2, inter_dist3, inter_dist4 = inter_distinct_metrics(total_sentences_generated)
bleu1, bleu2, bleu3, bleu4 = bleu_calc_all(total_sentences_original, total_sentences_generated)
rouge1, rouge2, rougeL = rouge_calc_all(total_sentences_original, total_sentences_generated)
print(f'intra_distinct@1: {intra_dist1}, intra_distinct@2: {intra_dist2}, intra_distinct@3: {intra_dist3}, intra_distinct@4: {intra_dist4}')
print(f'inter_distinct@1: {inter_dist1}, inter_distinct@2: {inter_dist2}, inter_distinct@3: {inter_dist3}, inter_distinct@4: {inter_dist4}')
print(f'bleu@1: {bleu1}, bleu@2: {bleu2}, bleu@3: {bleu3}, bleu@4: {bleu4}')
print(f'rouge@1: {rouge1}, rouge@2: {rouge2}, rouge@L: {rougeL}')


# # In[ ]:

