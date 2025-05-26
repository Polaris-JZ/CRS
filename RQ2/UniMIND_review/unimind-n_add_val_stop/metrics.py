import re
import os
import json
import numpy as np
from nltk.translate.bleu_score import corpus_bleu, sentence_bleu
from collections import Counter
import torch
import torch.nn.functional as F
from sklearn import metrics
import jieba
import re
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from rouge import Rouge
import logging

def calculate_rouge(preds, refs):
    """
    Calculate ROUGE-1, ROUGE-2, and ROUGE-L scores using Rouge library
    Args:
        preds: list of predicted texts
        refs: list of reference texts
    Returns:
        [rouge1_f, rouge2_f, rougeL_f]: F1 scores for ROUGE-1, ROUGE-2, and ROUGE-L
    """
    rouge = Rouge()
    
    rouge1_f = []
    rouge2_f = []
    rougeL_f = []
    
    for pred, ref in zip(preds, refs):
        # Convert token lists back to strings if necessary
        if isinstance(pred, list):
            pred = ' '.join(pred)
        if isinstance(ref, list):
            ref = ' '.join(ref)
            
        # Handle empty strings to prevent Rouge errors
        if not pred.strip():
            pred = "empty"
        if not ref.strip():
            ref = "empty"
            
        try:
            scores = rouge.get_scores(pred, ref)[0]
            
            rouge1_f.append(scores['rouge-1']['f'])
            rouge2_f.append(scores['rouge-2']['f'])
            rougeL_f.append(scores['rouge-l']['f'])
        except Exception as e:
            logging.warning(f"Rouge calculation failed for: pred='{pred}', ref='{ref}'. Error: {e}")
            rouge1_f.append(0.0)
            rouge2_f.append(0.0)
            rougeL_f.append(0.0)
    
    # Calculate average scores
    rouge1_f = sum(rouge1_f) / len(rouge1_f)
    rouge2_f = sum(rouge2_f) / len(rouge2_f)
    rougeL_f = sum(rougeL_f) / len(rougeL_f)
    
    return [rouge1_f, rouge2_f, rougeL_f]

def recall_score(preds, refs):
    """
    Calculate Recall@K for K = 1, 10, 50
    Args:
        preds: list of predicted items
        refs: list of reference (ground truth) items
    Returns:
        [recall@1, recall@10, recall@50]
    """
    recall1, recall10, recall50 = [], [], []
    
    for pred, ref in zip(preds, refs):
        # Calculate number of relevant items in top-K
        rel_1 = len(set(pred[:1]) & set(ref))
        rel_10 = len(set(pred[:10]) & set(ref))
        rel_50 = len(set(pred[:50]) & set(ref))
        
        # Calculate recall@K
        r1 = rel_1 / len(ref) if len(ref) > 0 else 0
        r10 = rel_10 / len(ref) if len(ref) > 0 else 0
        r50 = rel_50 / len(ref) if len(ref) > 0 else 0
        
        recall1.append(r1)
        recall10.append(r10)
        recall50.append(r50)
    
    # Average over all samples
    recall1 = sum(recall1) / len(recall1)
    recall10 = sum(recall10) / len(recall10)
    recall50 = sum(recall50) / len(recall50)
    
    return [recall1, recall10, recall50]
    
def f1_score(preds, refs):
    f1s = []
    for pred_items, gold_items in zip(preds, refs):
        common = Counter(gold_items) & Counter(pred_items)
        num_same = sum(common.values())
        if num_same == 0:
            f1 = 0
        else:
            precision = 1.0 * num_same / len(pred_items)
            recall = 1.0 * num_same / len(gold_items)
            f1 = (2 * precision * recall) / (precision + recall)
        f1s.append(f1)
    return sum(f1s)/len(f1s)

def distinct(seqs):
    """Calculate distinct-1,2,3,4 scores"""
    intra_dist1, intra_dist2, intra_dist3, intra_dist4 = [], [], [], []
    unigrams_all = Counter()
    bigrams_all = Counter()
    trigrams_all = Counter()
    fourgrams_all = Counter()

    for seq in seqs:
        # Calculate n-grams
        unigrams = Counter(seq)
        bigrams = Counter(zip(seq, seq[1:]))
        trigrams = Counter(zip(seq, seq[1:], seq[2:]))
        fourgrams = Counter(zip(seq, seq[1:], seq[2:], seq[3:]))
        
        # Calculate intra-distinct scores
        intra_dist1.append((len(unigrams)+1e-12) / (len(seq)+1e-5))
        intra_dist2.append((len(bigrams)+1e-12) / (max(0, len(seq)-1)+1e-5))
        intra_dist3.append((len(trigrams)+1e-12) / (max(0, len(seq)-2)+1e-5))
        intra_dist4.append((len(fourgrams)+1e-12) / (max(0, len(seq)-3)+1e-5))

        # Update global counters
        unigrams_all.update(unigrams)
        bigrams_all.update(bigrams)
        trigrams_all.update(trigrams)
        fourgrams_all.update(fourgrams)

    # Calculate inter-distinct scores
    inter_dist1 = (len(unigrams_all)+1e-12) / (sum(unigrams_all.values())+1e-5)
    inter_dist2 = (len(bigrams_all)+1e-12) / (sum(bigrams_all.values())+1e-5)
    inter_dist3 = (len(trigrams_all)+1e-12) / (sum(trigrams_all.values())+1e-5)
    inter_dist4 = (len(fourgrams_all)+1e-12) / (sum(fourgrams_all.values())+1e-5)
    
    # Calculate average intra-distinct scores
    intra_dist1 = np.average(intra_dist1)
    intra_dist2 = np.average(intra_dist2)
    intra_dist3 = np.average(intra_dist3)
    intra_dist4 = np.average(intra_dist4)

    return (intra_dist1, intra_dist2, intra_dist3, intra_dist4, 
            inter_dist1, inter_dist2, inter_dist3, inter_dist4)

def perplexity(logits, targets, weight=None, padding_idx=None, device=None):
    """
    logits: (batch_size, max_len, vocab_size)
    targets: (batch_size, max_len)
    """
    batch_size = logits.size(0)
    if weight is None and padding_idx is not None:
        weight = torch.ones(logits.size(-1), device=device)
        weight[padding_idx] = 0
    nll = F.nll_loss(input=logits.view(-1, logits.size(-1)),
                     target=targets.contiguous().view(-1),
                     weight=weight,
                     reduction='none')
    nll = nll.view(batch_size, -1).sum(dim=1)
    if padding_idx is not None:
        word_cnt = targets.ne(padding_idx).float().sum()
        nll = nll / word_cnt
    ppl = nll.exp()
    return ppl

def know_f1_score(pred_pt, gold_pt):
    ps = []
    rs = []
    f1s = []
    for pred_labels, gold_labels in zip(pred_pt, gold_pt):
        if len(pred_labels) == 0:
            pred_labels.append('empty')
        if len(gold_labels) == 0:
            gold_labels.append('empty')
        tp = 0
        for t in pred_labels:
            if t in gold_labels:
                tp += 1
        r = tp / len(gold_labels)
        p = tp / len(pred_labels)
        try:
            f1 = 2 * p * r / (p + r)
        except ZeroDivisionError:
            f1 = 0
        ps.append(p)
        rs.append(r)
        f1s.append(f1)
    p = sum(ps) / len(ps)
    r = sum(rs) / len(rs)
    f1 = sum(f1s) / len(f1s)
    scores = [p, r, f1]

    return scores

'''
def know_hit_score(pred_pt, gold_pt):
    hits = []
    for pred_labels, gold_labels in zip(pred_pt, gold_pt):
        if len(gold_labels) == 0:
            continue
        if len(set(pred_labels)&set(gold_labels)) > 0:
            hits.append(1)
        else:
            hits.append(0)
    hits = sum(hits)/len(hits)
    return [hits]
'''
def know_hit_score(pred_pt, gold_pt):
    hits1 = []
    hits3 = []
    hits5 = []
    for pred_labels, gold_labels in zip(pred_pt, gold_pt):
        if len(gold_labels) == 0:
            continue
        if len(set(pred_labels[:1])&set(gold_labels)) > 0:
            hits1.append(1)
        else:
            hits1.append(0)
        if len(set(pred_labels[:3])&set(gold_labels)) > 0:
            hits3.append(1)
        else:
            hits3.append(0)
        if len(set(pred_labels[:5])&set(gold_labels)) > 0:
            hits5.append(1)
        else:
            hits5.append(0)
    hits1 = sum(hits1)/len(hits1)
    hits3 = sum(hits3)/len(hits3)
    hits5 = sum(hits5)/len(hits5)
    return [hits1, hits3, hits5]


def goal_f1_score(pred_pt, gold_pt, data_name):
    goal_dict = {}
    with open('../data/{}/goal2id.txt'.format(data_name),'r',encoding='utf-8') as infile:
        for line in infile:
            items = line.strip().lower().split('\t')
            goal_dict[items[0]] = items[1]

    def make_label(l, label_dict):
        length = len(label_dict)
        result = [0] * length
        for label in l:
            if label.strip().lower() == '':
                continue
            label = ''.join(label.strip().lower().split(' '))
            if label not in label_dict:
                continue
            result[int(label_dict[label])] = 1
        return result
    
    def get_metrics(y, y_pre, data_name):
        if data_name == 'durecdial':
            macro_f1 = metrics.f1_score(y, y_pre, average='macro')
            macro_precision = metrics.precision_score(y, y_pre, average='macro')
            macro_recall = metrics.recall_score(y, y_pre, average='macro')
        else:
            f1 = metrics.f1_score(y, y_pre, average=None).tolist()
            p = metrics.precision_score(y, y_pre, average=None).tolist()
            r = metrics.recall_score(y, y_pre, average=None).tolist()
            print(f1.count(0), p.count(0), r.count(0))
            macro_f1 = sum(f1)/(len(f1)-f1.count(0))
            macro_precision = sum(p)/(len(p)-p.count(0))
            macro_recall = sum(r)/(len(r)-r.count(0))
        return macro_precision, macro_recall, macro_f1
    if data_name == 'tgredial':
        reference = np.array([make_label(y, goal_dict) for y in gold_pt])
        candidate = np.array([make_label(y_pre, goal_dict) for y_pre in pred_pt])
    else:
        reference = gold_pt
        candidate = pred_pt
    all_scores = list(get_metrics(reference, candidate, data_name))

    return all_scores

def ndcg_score(preds, refs):
    ndcg10 = []
    ndcg50 = []
    for pred, ref in zip(preds, refs):
        #if 0 in ref:
        #    continue
        score10 = 0.0
        score50 = 0.0
        for rank, item in enumerate(pred):
            if item in ref:
                if rank < 10:
                    score10 += 1.0/np.log2(rank+2)
                if rank < 50:
                    score50 += 1.0/np.log2(rank+2)
        
        norm = 0.0
        for rank in range(len(ref)):
            norm += 1.0/np.log2(rank+2)
        ndcg10.append(score10/max(0.3,norm))
        ndcg50.append(score50/max(0.3,norm))
    ndcg10 = sum(ndcg10)/len(ndcg10)
    ndcg50 = sum(ndcg50)/len(ndcg50)
    return [ndcg10, ndcg50]

def mrr_score(preds, refs):
    mrr10 = []
    mrr50 = []
    for pred, ref in zip(preds, refs):
        #if 0 in ref:
        #    continue
        score10 = 0.0
        score50 = 0.0
        for rank, item in enumerate(pred):
            if item in ref:
                if rank < 10:
                    score10 = 1.0/ (rank+1.0)
                    score50 = 1.0/ (rank+1.0)
                    break
                if rank < 50:
                    score50 = 1.0/ (rank+1.0)
                    break
        mrr10.append(score10)
        mrr50.append(score50)
    mrr10 = sum(mrr10)/len(mrr10)
    mrr50 = sum(mrr50)/len(mrr50)
    return [mrr10, mrr50]

def bleu_cal(sen1, tar1):
    smoother = SmoothingFunction().method1
    bleu1 = sentence_bleu([tar1], sen1, weights=(1, 0, 0, 0), smoothing_function=smoother)
    bleu2 = sentence_bleu([tar1], sen1, weights=(0.5, 0.5, 0, 0), smoothing_function=smoother)
    bleu3 = sentence_bleu([tar1], sen1, weights=(0.33, 0.33, 0.33, 0), smoothing_function=smoother)
    bleu4 = sentence_bleu([tar1], sen1, weights=(0.25, 0.25, 0.25, 0.25), smoothing_function=smoother)
    bleu = sentence_bleu([tar1], sen1, smoothing_function=smoother)
    return bleu1, bleu2, bleu3, bleu4, bleu

def tgredial_bleu(tokenized_gen, tokenized_tar):
    bleu1_sum, bleu2_sum, bleu3_sum, bleu4_sum, bleu_sum, count = 0, 0, 0, 0, 0, 0  # 添加bleu3_sum和bleu4_sum
    for sen, tar in zip(tokenized_gen, tokenized_tar):
        bleu1, bleu2, bleu3, bleu4, bleu = bleu_cal(sen, tar)  # 修改接收的返回值
        bleu1_sum += bleu1
        bleu2_sum += bleu2
        bleu3_sum += bleu3  # 添加bleu3累加
        bleu4_sum += bleu4  # 添加bleu4累加
        bleu_sum += bleu
        count += 1

    return (bleu_sum / count, bleu1_sum / count, bleu2_sum / count, 
            bleu3_sum / count, bleu4_sum / count)  # 修改返回值

def calculate(raw_pred, raw_ref, data_name, task):
    print('Dataset: ', data_name, 'Task: ', task, '-----------------')
    if task in ['resp','direct']:
        # 预处理文本，同时保存原始版本和处理后版本
        processed_refs = []
        processed_preds = []
        if data_name == 'durecdial':
            processed_refs = [ref.split(' ') for ref in raw_ref]
            processed_preds = [pred.split(' ') for pred in raw_pred]
        else:
            # 处理中文文本
            for ref, pred in zip(raw_ref, raw_pred):
                # 处理参考文本
                new_ref = []
                ref = re.sub(r'《(.*)》', '<movie>', ''.join(ref.split(' ')))
                ref_split_by_movie = list(ref.split('<movie>'))
                for i, sen_split in enumerate(ref_split_by_movie):
                    for segment in jieba.cut(sen_split):
                        new_ref.append(segment)
                    if i != len(ref_split_by_movie) - 1:
                        new_ref.append('<movie>')
                processed_refs.append(new_ref)
                
                # 处理预测文本
                new_pred = []
                pred = re.sub(r'《(.*)》', '<movie>', ''.join(pred.split(' ')))
                pred_split_by_movie = list(pred.split('<movie>'))
                for i, sen_split in enumerate(pred_split_by_movie):
                    for segment in jieba.cut(sen_split):
                        new_pred.append(segment)
                    if i != len(pred_split_by_movie) - 1:
                        new_pred.append('<movie>')
                processed_preds.append(new_pred)
        
        # BLEU 计算使用处理后的文本
        if data_name == 'durecdial': 
            bleu_refs = [[ref] for ref in processed_refs]
            bleu_score = corpus_bleu(bleu_refs, processed_preds)
            bleu1 = corpus_bleu(bleu_refs, processed_preds, weights=(1, 0, 0, 0))
            bleu2 = corpus_bleu(bleu_refs, processed_preds, weights=(0.5, 0.5, 0, 0))
        else:
            bleu_score, bleu1, bleu2, bleu3, bleu4 = tgredial_bleu(processed_refs, processed_preds)
        
        # ROUGE 计算也使用处理后的文本，但需要转换回字符串
        processed_ref_strs = [' '.join(ref) for ref in processed_refs]
        processed_pred_strs = [' '.join(pred) for pred in processed_preds]
        rouge_scores = calculate_rouge(processed_pred_strs, processed_ref_strs)
        
        bleu_scores = [bleu_score, bleu1, bleu2, bleu3, bleu4]
        logging.info('Running BLEU for %s -----------------------------', data_name)
        logging.info('BLEU/BLEU-1/BLEU-2/BLEU-3/BLEU-4: %s', str(bleu_scores))
        
        # Add ROUGE calculation
        logging.info('Running ROUGE for %s -----------------------------', data_name)
        logging.info('ROUGE-1/ROUGE-2/ROUGE-L: %s', str(rouge_scores))
        
        dist_scores = list(distinct(processed_preds))
        logging.info('Running Dist for %s -----------------------------', data_name)
        logging.info('Intra-Dist1/2/3/4, Inter-Dist1/2/3/4: %s', str(dist_scores))

        f1_scores = [f1_score(processed_preds, processed_refs)]
        logging.info('Running F1 for %s -----------------------------', data_name)
        logging.info('F1: %s', str(f1_scores))

        auto_scores = bleu_scores + rouge_scores + dist_scores + f1_scores

    elif task == 'know':
        refs = [ref.split(' | ') for ref in raw_ref]
        preds = [pred.split(' | ') for pred in raw_pred]
        hit_scores = know_hit_score(preds, refs)
        f1_scores = know_f1_score(preds, refs)
        print('Running P/R/F1 for %s -----------------------------', data_name)
        print('P/R/F1/hits: %s %s', str(f1_scores), str(hit_scores))
        auto_scores = f1_scores + hit_scores
    elif task == 'goal':
        if data_name == 'durecdial':
            refs = raw_ref
            preds = raw_pred
        else:
            refs = [ref.split(' | ') for ref in raw_ref]
            preds = [pred.split(' | ') for pred in raw_pred]
        f1_scores = goal_f1_score(preds, refs, data_name)
        print('Running P/R/F1 for %s -----------------------------', data_name)
        print('P/R/F1: %s', str(f1_scores))
        auto_scores = f1_scores
    elif task == 'item':
        if type(raw_pred[0]) is not list: 
            preds = [eval(pred) for pred in raw_pred]
        else:
            preds = raw_pred.copy()
        if type(raw_ref[0]) is not list: 
            refs = [eval(ref) for ref in raw_ref]
        else:
            refs = raw_ref.copy()
        ndcg_scores = ndcg_score(preds, refs)
        mrr_scores = mrr_score(preds, refs)
        recall_scores = recall_score(preds, refs)  # 添加这行
        logging.info('Running NDCG and Recall and MRR for %s -----------------------------', data_name)
        logging.info('NDCG@10/NDCG@50/Recall@1/Recall@10/Recall@50/MRR@10/MRR@50: %s %s %s', 
                    str(ndcg_scores), str(recall_scores), str(mrr_scores))
        auto_scores = ndcg_scores + recall_scores + mrr_scores

    return auto_scores



if __name__ == '__main__':
    data = 'tgredial'
    task = 'item'
    pred = []
    with open('output/{}/{}/{}_{}.decoded'.format(data,task,data,task), 'r', encoding='utf-8') as infile:
        for line in infile:
            pred.append(line.lower().strip())
    ref = []
    with open('output/{}/{}/{}_{}.reference'.format(data,task,data,task), 'r', encoding='utf-8') as infile:
        for line in infile:
            ref.append(line.lower().strip())
    auto_metric = calculate(pred, ref, data, task)