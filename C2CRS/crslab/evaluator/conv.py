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
from crslab.evaluator.metrics.gen import IntraDistMetric


class ConvEvaluator(BaseEvaluator):
    """The evaluator specially for conversational model
    
    Args:
        dist_set: the set to record dist n-gram
        dist_cnt: the count of dist n-gram evaluation
        gen_metrics: the metrics to evaluate conversational model, including bleu, dist, embedding metrics, f1
        optim_metrics: the metrics to optimize in training

    """

    def __init__(self):
        super(ConvEvaluator, self).__init__()
        self.dist_set = defaultdict(set)
        self.total_ngrams = defaultdict(int)  # 用于 inter-distinct
        self.dist_cnt = 0
        self.intra_dist_scores = defaultdict(list)
        self.gen_metrics = Metrics()
        self.optim_metrics = Metrics()

    def _load_embedding(self, language):
        resource = resources[language]
        dpath = os.path.join(EMBEDDING_PATH, language)
        build(dpath, resource['file'], resource['version'])

        model_file = os.path.join(dpath, f'cc.{language}.300.bin')
        self.ft = fasttext.load_model(model_file)
        logger.info(f'[Load {model_file} for embedding metric')

    def _get_sent_embedding(self, sent):
        return [self.ft[token] for token in sent.split()]

    def gen_evaluate(self, hyp, refs):
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

            # BLEU and Distinct metrics
            for k in range(1, 5):
                self.gen_metrics.add(f"bleu@{k}", BleuMetric.compute(hyp, refs, k))
            
            # ROUGE metrics - 使用正确的rouge类型
            for rouge_type in ['rouge1', 'rouge2', 'rougeL']:
                score = ROUGEMetric.compute(hyp, refs, rouge_type)
                self.gen_metrics.add(rouge_type, score)

            # Embedding-based metrics
            hyp_emb = self._get_sent_embedding(hyp)
            ref_embs = [self._get_sent_embedding(ref) for ref in refs]
            self.gen_metrics.add('greedy', GreedyMatch.compute(hyp_emb, ref_embs))
            self.gen_metrics.add('average', EmbeddingAverage.compute(hyp_emb, ref_embs))
            self.gen_metrics.add('extreme', VectorExtrema.compute(hyp_emb, ref_embs))

    def report(self):
        # 计算 inter-distinct 比例
        for k, v in self.dist_set.items():
            self.gen_metrics.add(k, AverageMetric(len(v) / max(self.total_ngrams[k], 1)))
        
        # Intra-distinct - 不需要再除以 dist_cnt
        for k in range(1, 5):
            intra_key = f"intra_dist@{k}"
            if intra_key in self.gen_metrics._data:
                value = self.gen_metrics._data[intra_key].value()  # 已经是平均值了
                self.gen_metrics._data[intra_key] = AverageMetric(value)  # 不再除以 dist_cnt
        

        reports = [self.gen_metrics.report(), self.optim_metrics.report()]
        if self.tensorboard and mode != 'test':
            for idx, task_report in enumerate(reports):
                for each_metric, value in task_report.items():
                    self.writer.add_scalars(f'{self.reports_name[idx]}/{each_metric}', {mode: value.value()}, epoch)

        logger.info('\n' + nice_report(aggregate_unnamed_reports(reports)))

    def reset_metrics(self):
        self.gen_metrics.clear()
        self.dist_cnt = 0
        self.dist_set.clear()
        self.total_ngrams.clear()
        self.optim_metrics.clear()