import numpy as np

def distinct_metrics(outs):
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

def calculate_ndcg(gt_item, ranked_items, k):
    """
    Calculate NDCG@k for a single recommendation
    Args:
        gt_item: ground truth item id
        ranked_items: list of recommended item ids
        k: calculate NDCG@k
    """
    if not ranked_items:
        return 0.0
    
    # 截取前k个推荐项
    ranked_items = ranked_items[:k]
    
    # 计算DCG
    dcg = 0.0
    for i, item in enumerate(ranked_items):
        if item == gt_item:
            # relevance是二值的：1(匹配) 或 0(不匹配)
            # i+1 是因为位置从1开始计数
            dcg += 1.0 / np.log2(i + 2)  # log2(i+2)是因为位置从1开始
    
    # 计算IDCG (理想情况下的DCG)
    # 理想情况是相关项在第一位
    idcg = 1.0  # 因为只有一个相关项，且relevance=1
    
    # 如果IDCG为0，返回0
    if idcg == 0:
        return 0.0
        
    # 计算NDCG
    ndcg = dcg / idcg
    return ndcg
