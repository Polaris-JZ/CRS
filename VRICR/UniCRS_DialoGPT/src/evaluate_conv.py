import re

import json
from nltk import ngrams
from nltk.translate.bleu_score import sentence_bleu
from rouge import Rouge
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
# year_pattern = re.compile(r'\(\d{4}\)')
slot_pattern = re.compile(r'<movie>')


class ConvEvaluator:
    def __init__(self, tokenizer, log_file_path):
        self.tokenizer = tokenizer
        self.rouge = Rouge()
        
        self.reset_metric()
        if log_file_path:
            self.log_file = open(log_file_path, 'w', buffering=1)
            self.log_cnt = 0

    def evaluate(self, preds, labels, log=False):
        decoded_preds = self.tokenizer.batch_decode(preds, skip_special_tokens=False)
        decoded_preds = [decoded_pred.replace('<pad>', '').replace('<|endoftext|>', '') for decoded_pred in
                         decoded_preds]
        decoded_preds = [pred.strip() for pred in decoded_preds]
        decoded_labels = self.tokenizer.batch_decode(labels, skip_special_tokens=False)
        decoded_labels = [decoded_label.replace('<pad>', '').replace('<|endoftext|>', '') for decoded_label in
                          decoded_labels]
        decoded_labels = [label.strip() for label in decoded_labels]

        if log and hasattr(self, 'log_file'):
            for pred, label in zip(decoded_preds, decoded_labels):
                self.log_file.write(json.dumps({
                    'pred': pred,
                    'label': label
                }, ensure_ascii=False) + '\n')

        self.collect_intra_distinct(decoded_preds)
        self.collect_inter_distinct(decoded_preds)
        self.compute_item_ratio(decoded_preds)
        self.compute_bleu(decoded_preds, decoded_labels)
        self.compute_rouge(decoded_preds, decoded_labels)
        self.sent_cnt += len([pred for pred in decoded_preds if len(pred) > 0])

    def collect_intra_distinct(self, strs):
        for str in strs:
            words = str.split()
            
            # unigrams
            unigram_set = set()
            unigram_total = len(words)
            for word in words:
                unigram_set.add(word)
            if unigram_total > 0:
                self.metric['intra_dist@1'].append(len(unigram_set) / unigram_total)
            
            # bigrams
            if len(words) >= 2:
                bigram_set = set()
                bigram_total = len(words) - 1
                for i in range(bigram_total):
                    bg = words[i] + ' ' + words[i + 1]
                    bigram_set.add(bg)
                self.metric['intra_dist@2'].append(len(bigram_set) / bigram_total)
            
            # trigrams
            if len(words) >= 3:
                trigram_set = set()
                trigram_total = len(words) - 2
                for i in range(trigram_total):
                    trg = words[i] + ' ' + words[i + 1] + ' ' + words[i + 2]
                    trigram_set.add(trg)
                self.metric['intra_dist@3'].append(len(trigram_set) / trigram_total)
            
            # quagrams
            if len(words) >= 4:
                quagram_set = set()
                quagram_total = len(words) - 3
                for i in range(quagram_total):
                    quag = words[i] + ' ' + words[i + 1] + ' ' + words[i + 2] + ' ' + words[i + 3]
                    quagram_set.add(quag)
                self.metric['intra_dist@4'].append(len(quagram_set) / quagram_total)

    def collect_inter_distinct(self, strs):
        all_unigrams = []
        all_bigrams = []
        all_trigrams = []
        all_quagrams = []

        for str in strs:
            words = str.split()
            
            # unigrams
            all_unigrams.extend(words)
            
            # bigrams
            for i in range(len(words) - 1):
                bg = words[i] + ' ' + words[i + 1]
                all_bigrams.append(bg)
                
            # trigrams
            for i in range(len(words) - 2):
                trg = words[i] + ' ' + words[i + 1] + ' ' + words[i + 2]
                all_trigrams.append(trg)
                
            # quagrams
            for i in range(len(words) - 3):
                quag = words[i] + ' ' + words[i + 1] + ' ' + words[i + 2] + ' ' + words[i + 3]
                all_quagrams.append(quag)

        self.metric['inter_dist@1'] = len(set(all_unigrams)) / max(len(all_unigrams), 1)
        self.metric['inter_dist@2'] = len(set(all_bigrams)) / max(len(all_bigrams), 1)
        self.metric['inter_dist@3'] = len(set(all_trigrams)) / max(len(all_trigrams), 1)
        self.metric['inter_dist@4'] = len(set(all_quagrams)) / max(len(all_quagrams), 1)

    def compute_bleu(self, preds, labels):
        for pred, label in zip(preds, labels):
            pred, label = pred.split(), [label.split()]
            for k in range(4):
                # bleu1s.append(sentence_bleu([reference], candidate, 
                #     weights=(1, 0, 0, 0), smoothing_function=smoother))
                # bleu2s.append(sentence_bleu([reference], candidate, 
                #     weights=(0.5, 0.5, 0, 0), smoothing_function=smoother))
                # bleu3s.append(sentence_bleu([reference], candidate, 
                #     weights=(0.33, 0.33, 0.33, 0), smoothing_function=smoother))
                # bleu4s.append(sentence_bleu([reference], candidate, 
                #     weights=(0.25, 0.25, 0.25, 0.25), smoothing_function=smoother))
                if k == 0:
                    weights = (1, 0, 0, 0)
                elif k == 1:
                    weights = (0.5, 0.5, 0, 0)
                elif k == 2:
                    weights = (0.33, 0.33, 0.33, 0)
                else:
                    weights = (0.25, 0.25, 0.25, 0.25)
                self.metric[f'bleu@{k + 1}'] += sentence_bleu(label, pred, weights, smoothing_function=SmoothingFunction().method1)
    def compute_item_ratio(self, strs):
        for str in strs:
            # items = re.findall(year_pattern, str)
            # self.metric['item_ratio'] += len(items)
            items = re.findall(slot_pattern, str)
            self.metric['item_ratio'] += len(items)

    def compute_rouge(self, preds, labels):
        for pred, label in zip(preds, labels):
            if len(pred.strip()) == 0:
                pred = "empty"
            if len(label.strip()) == 0:
                label = "empty"
            try:
                scores = self.rouge.get_scores(pred, label)[0]
                self.metric['rouge1'] += scores['rouge-1']['f']
                self.metric['rouge2'] += scores['rouge-2']['f']
                self.metric['rougeL'] += scores['rouge-l']['f']
            except ValueError:
                continue

    def report(self):
        report = {}
        for k, v in self.metric.items():
            if self.sent_cnt == 0:
                report[k] = 0
            else:
                if k.startswith('intra_dist'):
                    report[k] = sum(v) / len(v) if v else 0
                elif k.startswith('rouge') or k.startswith('bleu'):
                    report[k] = v / self.sent_cnt
                else:
                    report[k] = v
        report['sent_cnt'] = self.sent_cnt
        return report

    def reset_metric(self):
        self.metric = {
            'bleu@1': 0,
            'bleu@2': 0,
            'bleu@3': 0,
            'bleu@4': 0,
            'intra_dist@1': [],
            'intra_dist@2': [],
            'intra_dist@3': [],
            'intra_dist@4': [],
            'inter_dist@1': 0,
            'inter_dist@2': 0,
            'inter_dist@3': 0,
            'inter_dist@4': 0,
            'item_ratio': 0,
            'rouge1': 0,
            'rouge2': 0,
            'rougeL': 0,
        }
        self.sent_cnt = 0