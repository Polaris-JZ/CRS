# @Time   : 2020/11/30
# @Author : Xiaolei Wang
# @Email  : wxl1999@foxmail.com

# UPDATE:
# @Time   : 2020/12/18
# @Author : Xiaolei Wang
# @Email  : wxl1999@foxmail.com

import re
from collections import Counter

import math
import numpy as np
from nltk import ngrams
from nltk.translate.bleu_score import sentence_bleu
from sklearn.metrics.pairwise import cosine_similarity
from typing import List, Optional

from crslab.evaluator.metrics.base import AverageMetric, SumMetric
from rouge import Rouge

re_art = re.compile(r'\b(a|an|the)\b')
re_punc = re.compile(r'[!"#$%&()*+,-./:;<=>?@\[\]\\^`{|}~_\']')
re_space = re.compile(r'\s+')


class PPLMetric(AverageMetric):
    def value(self):
        return math.exp(super().value())


def normalize_answer(s):
    """
    Lower text and remove punctuation, articles and extra whitespace.
    """

    s = s.lower()
    s = re_punc.sub(' ', s)
    s = re_art.sub(' ', s)
    s = re_space.sub(' ', s)
    # s = ' '.join(s.split())
    return s


class ExactMatchMetric(AverageMetric):
    @staticmethod
    def compute(guess: str, answers: List[str]) -> 'ExactMatchMetric':
        if guess is None or answers is None:
            return None
        for a in answers:
            if guess == a:
                return ExactMatchMetric(1)
        return ExactMatchMetric(0)


class F1Metric(AverageMetric):
    """
    Helper class which computes token-level F1.
    """

    @staticmethod
    def _prec_recall_f1_score(pred_items, gold_items):
        """
        Compute precision, recall and f1 given a set of gold and prediction items.

        :param pred_items: iterable of predicted values
        :param gold_items: iterable of gold values

        :return: tuple (p, r, f1) for precision, recall, f1
        """
        common = Counter(gold_items) & Counter(pred_items)
        num_same = sum(common.values())
        if num_same == 0:
            return 0
        precision = 1.0 * num_same / len(pred_items)
        recall = 1.0 * num_same / len(gold_items)
        f1 = (2 * precision * recall) / (precision + recall)
        return f1

    @staticmethod
    def compute(guess: str, answers: List[str]) -> 'F1Metric':
        if guess is None or answers is None:
            return AverageMetric(0, 0)
        g_tokens = guess.split()
        scores = [
            F1Metric._prec_recall_f1_score(g_tokens, a.split())
            for a in answers
        ]
        return F1Metric(max(scores), 1)


class BleuMetric(AverageMetric):
    @staticmethod
    def compute(guess: str, answers: List[str], k: int) -> Optional['BleuMetric']:
        """
        Compute approximate BLEU score between guess and a set of answers.
        """


        if guess is None or answers is None:
            return BleuMetric(0.0)
            
        # Normalize texts
        guess = normalize_answer(guess)
        answers = [normalize_answer(a) for a in answers]
        
        # Handle empty strings
        if not guess.strip():
            return BleuMetric(0.0)
        
        # Remove empty references
        answers = [a for a in answers if a.strip()]
        if not answers:
            return BleuMetric(0.0)
            
        # For single n-gram evaluation
        if k == 1:
            weights = [1,0,0,0]
        elif k == 2:
            weights = [0.5,0.5,0,0]
        elif k == 3:
            weights = [0.33,0.33,0.33,0]
        elif k == 4:
            weights = [0.25,0.25,0.25,0.25]
        
        try:
            score = sentence_bleu(
                [a.split() for a in answers],
                guess.split(),
                weights=weights,
            )
        except Exception:
            score = 0.0
            
        return BleuMetric(score)


class DistMetric(SumMetric):
    """计算所有句子的 inter-distinct 分数"""
    
    @staticmethod
    def compute(sents: List[str], k: int) -> 'DistMetric':
        """
        计算 k-gram 的 unique 数量
        
        Args:
            sents: 句子列表
            k: n-gram 长度
            
        Returns:
            DistMetric: unique n-grams 数量
        """
        if not isinstance(sents, list):
            sents = [sents]
            
        # 收集所有 k-grams
        all_ngrams = []
        
        for sent in sents:
            tokens = sent.split()
            # 对每个句子生成 k-grams
            for i in range(len(tokens) - k + 1):
                ngram = ' '.join(tokens[i:i+k])
                all_ngrams.append(ngram)
                
        # 返回 unique n-grams 数量
        return DistMetric(len(set(all_ngrams)))


class IntraDistMetric(AverageMetric):
    @staticmethod
    def compute(sent: str, k: int) -> 'IntraDistMetric':
        """
        计算单个句子的 distinct ratio
        
        Args:
            sent: 输入句子
            k: n-gram 长度
            
        Returns:
            IntraDistMetric: 该句子的 distinct ratio
        """
        tokens = sent.split()
        if len(tokens) < k:
            return IntraDistMetric(0.0)
            
        # 生成该句子的所有 k-grams
        ngram_set = set()
        total_ngrams = 0
        
        for i in range(len(tokens) - k + 1):
            ngram = ' '.join(tokens[i:i+k])
            ngram_set.add(ngram)
            total_ngrams += 1
            
        # 返回该句子的 distinct ratio
        return IntraDistMetric(len(ngram_set) / max(total_ngrams, 1))
    

class EmbeddingAverage(AverageMetric):
    @staticmethod
    def _avg_embedding(embedding):
        return np.sum(embedding, axis=0) / (np.linalg.norm(np.sum(embedding, axis=0)) + 1e-12)

    @staticmethod
    def compute(hyp_embedding, ref_embeddings) -> 'EmbeddingAverage':
        hyp_avg_emb = EmbeddingAverage._avg_embedding(hyp_embedding).reshape(1, -1)
        ref_avg_embs = [EmbeddingAverage._avg_embedding(emb) for emb in ref_embeddings]
        ref_avg_embs = np.array(ref_avg_embs)
        return EmbeddingAverage(float(cosine_similarity(hyp_avg_emb, ref_avg_embs).max()))


class VectorExtrema(AverageMetric):
    @staticmethod
    def _extreme_embedding(embedding):
        max_emb = np.max(embedding, axis=0)
        min_emb = np.min(embedding, axis=0)
        extreme_emb = np.fromiter(
            map(lambda x, y: x if ((x > y or x < -y) and y > 0) or ((x < y or x > -y) and y < 0) else y, max_emb,
                min_emb), dtype=float)
        return extreme_emb

    @staticmethod
    def compute(hyp_embedding, ref_embeddings) -> 'VectorExtrema':
        hyp_ext_emb = VectorExtrema._extreme_embedding(hyp_embedding).reshape(1, -1)
        ref_ext_embs = [VectorExtrema._extreme_embedding(emb) for emb in ref_embeddings]
        ref_ext_embs = np.asarray(ref_ext_embs)
        return VectorExtrema(float(cosine_similarity(hyp_ext_emb, ref_ext_embs).max()))


class GreedyMatch(AverageMetric):
    @staticmethod
    def compute(hyp_embedding, ref_embeddings) -> 'GreedyMatch':
        hyp_emb = np.asarray(hyp_embedding)
        ref_embs = (np.asarray(ref_embedding) for ref_embedding in ref_embeddings)
        score_max = 0
        for ref_emb in ref_embs:
            sim_mat = cosine_similarity(hyp_emb, ref_emb)
            score_max = max(score_max, (sim_mat.max(axis=0).mean() + sim_mat.max(axis=1).mean()) / 2)
        return GreedyMatch(score_max)


class ROUGEMetric(AverageMetric):
    """
    使用 Rouge 库计算 ROUGE 分数
    支持 ROUGE-1, ROUGE-2, ROUGE-L 等多种变体
    """
    
    @staticmethod
    def compute(guess: str, answers: List[str], rouge_type: str = 'rouge1') -> float:
        """
        计算单个ROUGE分数
        
        Args:
            guess: 预测的文本
            answers: 参考答案列表
            rouge_type: ROUGE类型 ('rouge-1', 'rouge-2', 'rouge-l')
            
        Returns:
            float: ROUGE分数
        """
        if rouge_type == 'rouge1':
            rouge_type = 'rouge-1'
        elif rouge_type == 'rouge2':
            rouge_type = 'rouge-2'
        elif rouge_type == 'rougeL':
            rouge_type = 'rouge-l'
        if guess is None or answers is None:
            return ROUGEMetric(0.0)

        rouge = Rouge()
        max_score = 0.0
        
        normalized_guess = normalize_answer(guess)
        if not normalized_guess.strip():  # Handle empty string case
            return ROUGEMetric(0.0)
            
        for answer in answers:
            normalized_answer = normalize_answer(answer)
            if not normalized_answer.strip():  # Handle empty string case
                continue
                
            try:
                scores = rouge.get_scores(normalized_guess, normalized_answer)[0]
                # Convert rouge_type to match Rouge library format (e.g., 'rouge-1' -> 'rouge-1')
                rouge_key = rouge_type.lower()
                if rouge_key in scores:
                    max_score = max(max_score, scores[rouge_key]['f'])
            except ValueError:  # Handle potential errors from Rouge library
                continue
        
        return ROUGEMetric(max_score)

