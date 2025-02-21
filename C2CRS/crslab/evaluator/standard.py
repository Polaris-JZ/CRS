# @Time   : 2020/11/30
# @Author : Xiaolei Wang
# @Email  : wxl1999@foxmail.com

# UPDATE:
# @Time   : 2020/12/18
# @Author : Xiaolei Wang
# @Email  : wxl1999@foxmail.com

import os
import time
from collections import defaultdict

import fasttext
from loguru import logger
from nltk import ngrams
from torch.utils.tensorboard import SummaryWriter

from crslab.evaluator.base import BaseEvaluator
from crslab.evaluator.utils import nice_report
from .embeddings import resources
from .metrics import *
from ..config import EMBEDDING_PATH
from ..download import build


class StandardEvaluator(BaseEvaluator):
    """The evaluator for all kind of model(recommender, conversation, policy)
    
    Args:
        rec_metrics: the metrics to evaluate recommender model, including hit@K, ndcg@K and mrr@K
        dist_set: the set to record dist n-gram
        dist_cnt: the count of dist n-gram evaluation
        gen_metrics: the metrics to evaluate conversational model, including bleu, dist, embedding metrics, f1
        optim_metrics: the metrics to optimize in training
    """

    def __init__(self, opt, language):
        super(StandardEvaluator, self).__init__()
        self.log_prefix = opt['log_prefix']
        # rec
        self.rec_metrics = Metrics()
        # gen
        self.dist_set = defaultdict(set)
        self.sent_len_list = defaultdict(list)
        self.total_ngrams = defaultdict(int)
        self.dist_cnt = 0
        self.gen_metrics = Metrics()
        self.intra_ngrams = defaultdict(int)
        self._load_embedding(language)
        # optim
        self.optim_metrics = Metrics()

    def _load_embedding(self, language):
        resource = resources[language]
        dpath = os.path.join(EMBEDDING_PATH, language)
        build(dpath, resource['file'], resource['version'])

        model_file = os.path.join(dpath, f'cc.{language}.300.bin')
        self.ft = fasttext.load_model(model_file)
        logger.info(f'[{self.log_prefix}][Load {model_file} for embedding metric')

    def _get_sent_embedding(self, sent):
        return [self.ft[token] for token in sent.split()]

    def rec_evaluate(self, ranks, label):
        for k in [1, 10, 50]:
            if len(ranks) >= k:
                self.rec_metrics.add(f"hit@{k}", HitMetric.compute(ranks, label, k))
                self.rec_metrics.add(f"ndcg@{k}", NDCGMetric.compute(ranks, label, k))
                self.rec_metrics.add(f"mrr@{k}", MRRMetric.compute(ranks, label, k))

    def rec_evaluate_and_return_score(self, ranks, fully_rec_ranks, label, score_type='hit'):
        hits, ndcgs, mrrs = [], [], []
        for k in [1, 10, 50]:
            if len(ranks) >= k:
                hitk = HitMetric.compute(ranks, label, k)
                ndcgk = NDCGMetric.compute(ranks, label, k)
                mrrk = MRRMetric.compute(ranks, label, k)

                hits.append(hitk)
                ndcgs.append(ndcgk)
                mrrs.append(mrrk)

                self.rec_metrics.add(f"hit@{k}", hitk)
                self.rec_metrics.add(f"ndcg@{k}", ndcgk)
                self.rec_metrics.add(f"mrr@{k}", mrrk)

        assert len(hits) == 3
        if score_type == 'hit':
            score = (hits[0].value() + hits[-1].value()) / 2
        elif score_type == 'hit50':
            score = hits[-1].value()
        elif score_type == 'rank':
            score = 1 / (fully_rec_ranks.index(label) + 1)
        elif score_type == 'ndcg':
            score = (ndcgs[0].value() + ndcgs[-1].value()) / 2

        return score

    def gen_evaluate(self, hyp, refs, hyp_ListStr, refs_ListStr):
        if hyp:
            self.gen_metrics.add("f1", F1Metric.compute(hyp, refs))

            # BLEU and Distinct metrics
            hyp_token = hyp.split()
            for k in range(1, 5):
                self.gen_metrics.add(f"bleu@{k}", BleuMetric.compute(hyp, refs, k))
                
                # Inter-distinct
                n = k
                if len(hyp_token) >= n:
                    # 累积总的 n-grams 数量
                    self.total_ngrams[f"dist@{k}"] += len(hyp_token) - n + 1
                    # 收集 unique n-grams
                    for token in ngrams(hyp_token, n):
                        self.dist_set[f"dist@{k}"].add(token)
                
                # Intra-distinct - 计算该句子的 ratio
                intra_score = IntraDistMetric.compute(hyp, k)
                self.gen_metrics.add(f"intra_dist@{k}", intra_score)
                
            self.dist_cnt += 1

            # ROUGE metrics
            for rouge_type in ['rouge1', 'rouge2', 'rougeL']:
                score = ROUGEMetric.compute(hyp, refs, rouge_type)
                self.gen_metrics.add(rouge_type, score)
            one_gram_list = [token for token in ngrams(hyp_ListStr, 1)]
            self.sent_len_list['sent_len'].append(len(one_gram_list))

            # Embedding-based metrics
            hyp_emb = self._get_sent_embedding(hyp)
            ref_embs = [self._get_sent_embedding(ref) for ref in refs]
            self.gen_metrics.add('greedy', GreedyMatch.compute(hyp_emb, ref_embs))
            self.gen_metrics.add('average', EmbeddingAverage.compute(hyp_emb, ref_embs))
            self.gen_metrics.add('extreme', VectorExtrema.compute(hyp_emb, ref_embs))

    def report(self):
        # Inter-distinct
        for k, v in self.dist_set.items():
            self.gen_metrics.add(k, AverageMetric(len(v) / max(self.total_ngrams[k], 1)))
        for k, v in self.sent_len_list.items():
            self.gen_metrics.add(k, AverageMetric(sum(v) / len(v)))       
        # Intra-distinct - 不需要再除以 dist_cnt
        for k in range(1, 5):
            intra_key = f"intra_dist@{k}"
            if intra_key in self.gen_metrics._data:
                value = self.gen_metrics._data[intra_key].value()  # 已经是平均值了
                self.gen_metrics._data[intra_key] = AverageMetric(value)  # 不再除以 dist_cnt
        
        reports = [self.rec_metrics.report(), self.gen_metrics.report(), self.optim_metrics.report()]
        logger.info(f'[{self.log_prefix}]\n' + nice_report(aggregate_unnamed_reports(reports)))
    
    def reset_metrics(self):
        # rec
        self.rec_metrics.clear()
        # conv
        self.gen_metrics.clear()
        self.dist_cnt = 0
        self.dist_set.clear()
        self.total_ngrams.clear()
        # optim
        self.optim_metrics.clear()