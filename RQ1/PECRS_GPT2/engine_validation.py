import numpy as np
import torch
import tqdm
import time
import copy
import gc
import re
from sklearn.metrics import recall_score, precision_score, f1_score
from rouge_score import rouge_scorer
from evaluation import distinct_metrics, inter_distinct_metrics, calculate_ndcg
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
import nltk
from nltk.tokenize.punkt import PunktSentenceTokenizer
from nltk.tokenize import word_tokenize

# validation on the entire dataset
def validate(ep, dataloader, tokenizer, model, criterions, logger, accelerator, args):
    punkt_tokenizer = PunktSentenceTokenizer()
    logger.info("\n")
    logger.info("Validating...")
    model.eval()
    accelerator.unwrap_model(model).annoy_base_constructor()

    # collect all predictions
    turn_nums, n_points, n_rec = [], 0, 0 # metadata
    ndcg_1, ndcg_10, ndcg_50 = [], [], []
    ppl_losses, ppls = [], [] # response
    recall_losses, n_recall_success, recall_top100, recall_top300, recall_top500 = [], 0, [], [], [] # recall
    rerank_losses, total_rerank_top1, rerank_top1, rerank_top10, rerank_top50 = [], [], [], [], [] # re-ranking
    gt_ids, gt_ranks, total_predicted_ids, all_predicted_ids = [], [], [], [] # final recommendation
    batch_sizes = []  # 记录每个batch的实际数据量
    
    # 在处理每个batch时记录数据量
    for batch in tqdm.tqdm(dataloader, disable=not accelerator.is_main_process):
        metadata, response, recall, rerank, ndcg, recommendation = validate_one_iteration(
            batch, tokenizer, model, criterions, accelerator, args)
        batch_sizes.append(len([i for i in range(len(batch["targets"])) if batch["targets"][i] != -1]))  # 记录该batch中实际的推荐数据量
        (turn_nums_batch, n_points_batch, n_rec_batch) = metadata
        (ppl_losses_batch, ppls_batch) = response
        (recall_losses_batch, n_recall_success_batch, recall_top100_batch, recall_top300_batch, recall_top500_batch) = recall
        (rerank_losses_batch, total_rerank_top1_batch, rerank_top1_batch, rerank_top10_batch, rerank_top50_batch) = rerank
        (gt_ids_batch, gt_ranks_batch, total_predicted_ids_batch) = recommendation
        (ndcg_1_batch, ndcg_10_batch, ndcg_50_batch) = ndcg
        turn_nums += turn_nums_batch
        n_points += n_points_batch
        n_rec += n_rec_batch

        ppl_losses += ppl_losses_batch
        ppls += ppls_batch

        recall_losses += recall_losses_batch
        n_recall_success += n_recall_success_batch
        recall_top100 += recall_top100_batch
        recall_top300 += recall_top300_batch
        recall_top500 += recall_top500_batch

        rerank_losses += rerank_losses_batch
        total_rerank_top1 += total_rerank_top1_batch
        rerank_top1 += rerank_top1_batch
        rerank_top10 += rerank_top10_batch
        rerank_top50 += rerank_top50_batch
        ndcg_1 += ndcg_1_batch
        ndcg_10 += ndcg_10_batch
        ndcg_50 += ndcg_50_batch

        gt_ids += gt_ids_batch
        gt_ranks += gt_ranks_batch
        total_predicted_ids += total_predicted_ids_batch
        all_predicted_ids.append(total_predicted_ids_batch)

    turn_nums = np.array(turn_nums)
    ppl_losses, ppls = np.array(ppl_losses), np.array(ppls)
    gt_ids, total_predicted_ids = np.array(gt_ids), np.array(total_predicted_ids)
    recall_losses, recall_top100, recall_top300, recall_top500 = np.array(recall_losses), np.array(recall_top100), np.array(recall_top300), np.array(recall_top500)
    rerank_losses, rerank_top1, rerank_top10, rerank_top50 = np.array(rerank_losses), np.array(rerank_top1), np.array(rerank_top10), np.array(rerank_top50)
    ndcg_1, ndcg_10, ndcg_50 = np.array(ndcg_1), np.array(ndcg_10), np.array(ndcg_50)
    logger.info(f"# Data points: {n_points}, # with rec: {n_rec}, # recall successful: {n_recall_success}")
    logger.info(f"Epoch {ep}, ppl loss: {np.mean(ppl_losses):.4f}, recall loss: {np.mean(recall_losses):.4f}, rerank loss: {np.mean(rerank_losses):.4f}")
    logger.info(f"ppl: {np.mean(ppls):.4f}, min {np.min(ppls):.4f} 10%: {np.percentile(ppls, 10):.4f}, mean: {np.mean(ppls):.4f}, 90 %: {np.percentile(ppls, 90):.4f}, 99 %: {np.percentile(ppls, 99):.4f}, ppl max: {np.max(ppls):.4f}")

    if args.generate:
        batch_count, keep_ids, sources = 0, [], []
        gt_rec, raw_gt_sens, gt_sens, gt_n_tokens = [], [], [], []
        pred_rec, gen_sens, tok_gen_sens, gen_n_tokens = [], [], [], []
        scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)
        r1s, r2s, rls = [], [], []
        bleu1s, bleu2s, bleu3s, bleu4s = [], [], [], []
        smoother = SmoothingFunction().method1
        batch_sizes_gen = []  # 记录每个batch中生成任务的实际数据量
        
        for batch in tqdm.tqdm(dataloader, disable=not accelerator.is_main_process):
            # 计算当前batch中实际的生成数据量
            current_batch_size = sum(1 for x in batch["repeated"] if x == 0)
            batch_sizes_gen.append(current_batch_size)
            
            keep_ids_batch = [0] * len(batch["repeated"])
            for j in range(len(batch["repeated"])):
                if batch["repeated"][j] == 0:
                    keep_ids_batch[j] = 1
            keep_ids += keep_ids_batch
            sources_batch, ground_truths, predicted = validate_language_metrics_batch_embeds(
                tokenizer, batch, model, accelerator, all_predicted_ids[batch_count], args)
            (gt_rec_batch, raw_gt_sens_batch, gt_sens_batch) = ground_truths
            (pred_rec_batch, gen_sens_batch, tok_gen_sens_batch) = predicted

            sources += sources_batch
            for j in range(len(gt_sens_batch)):
                gt_n_tokens.append(len(tokenizer(gt_sens_batch[j], return_tensors='pt')['input_ids'][0]))
                gen_n_tokens.append(len(tokenizer(gen_sens_batch[j], return_tensors='pt')['input_ids'][0]))
                
                # calculate ROUGE scores
                rouge_scores = scorer.score(gt_sens_batch[j], gen_sens_batch[j])
                r1s.append(rouge_scores['rouge1'].fmeasure)
                r2s.append(rouge_scores['rouge2'].fmeasure)
                rls.append(rouge_scores['rougeL'].fmeasure)
                
                # calculate BLEU scores
                reference = [word.lower() for sent in punkt_tokenizer.tokenize(gt_sens_batch[j]) for word in sent.split()]
                candidate = [word.lower() for sent in punkt_tokenizer.tokenize(gen_sens_batch[j]) for word in sent.split()]
                
                # calculate BLEU-1,2,3,4
                bleu1s.append(sentence_bleu([reference], candidate, 
                    weights=(1, 0, 0, 0), smoothing_function=smoother))
                bleu2s.append(sentence_bleu([reference], candidate, 
                    weights=(0.5, 0.5, 0, 0), smoothing_function=smoother))
                bleu3s.append(sentence_bleu([reference], candidate, 
                    weights=(0.33, 0.33, 0.33, 0), smoothing_function=smoother))
                bleu4s.append(sentence_bleu([reference], candidate, 
                    weights=(0.25, 0.25, 0.25, 0.25), smoothing_function=smoother))

            gt_rec += gt_rec_batch
            pred_rec += pred_rec_batch
            raw_gt_sens += raw_gt_sens_batch
            gt_sens += gt_sens_batch
            gen_sens += gen_sens_batch
            tok_gen_sens += tok_gen_sens_batch
            batch_count += 1

        logger.info(f">>>>>>>>>>> Generation:")
        logger.info(f"Generated {len(r1s)} sentences, including {sum(gt_rec)} with a required recommendation")
        logger.info(f">>>>>>>>>>> Generation metrics:")
        gt_n_tokens = np.array(gt_n_tokens)
        gen_n_tokens = np.array(gen_n_tokens)
        logger.info(f"# Tokens (GT): {np.mean(gt_n_tokens):.4f}, # Tokens (predicted): {np.mean(gen_n_tokens):.4f}")
        gt_rec = np.array(gt_rec)
        pred_rec = np.array(pred_rec)
        r = recall_score(gt_rec, pred_rec)
        p = precision_score(gt_rec, pred_rec)
        f1 = f1_score(gt_rec, pred_rec)
        logger.info(f"Prediction of recommendation: recall: {r:.4f}, precision: {p:.4f}, F-1: {f1:.4f} (GT count: {np.sum(gt_rec)} / Pred count: {np.sum(pred_rec)})")
        intra_dist1, intra_dist2, intra_dist3, intra_dist4 = distinct_metrics(tok_gen_sens)
        inter_dist1, inter_dist2, inter_dist3, inter_dist4 = inter_distinct_metrics(tok_gen_sens)
        logger.info(f"Intra Dist1: {intra_dist1:.4f}, Intra Dist2: {intra_dist2:.4f}, Intra Dist3: {intra_dist3:.4f}, Intra Dist4: {intra_dist4:.4f}")
        logger.info(f"Inter Dist1: {inter_dist1:.4f}, Inter Dist2: {inter_dist2:.4f}, Inter Dist3: {inter_dist3:.4f}, Inter Dist4: {inter_dist4:.4f}")
        total_samples_gen = sum(batch_sizes_gen)
        r1 = sum(x * n for x, n in zip(r1s, batch_sizes_gen)) / total_samples_gen
        r2 = sum(x * n for x, n in zip(r2s, batch_sizes_gen)) / total_samples_gen
        rl = sum(x * n for x, n in zip(rls, batch_sizes_gen)) / total_samples_gen
        
        b1 = sum(x * n for x, n in zip(bleu1s, batch_sizes_gen)) / total_samples_gen
        b2 = sum(x * n for x, n in zip(bleu2s, batch_sizes_gen)) / total_samples_gen
        b3 = sum(x * n for x, n in zip(bleu3s, batch_sizes_gen)) / total_samples_gen
        b4 = sum(x * n for x, n in zip(bleu4s, batch_sizes_gen)) / total_samples_gen

        logger.info(f"ROUGE-1: {r1:.4f}, ROUGE-2: {r2:.4f}, ROUGE-L: {rl:.4f}")
        logger.info(f"BLEU-1: {b1:.4f}, BLEU-2: {b2:.4f}, BLEU-3: {b3:.4f}, BLEU-4: {b4:.4f}")
        
    total_samples = sum(batch_sizes)
    logger.info(f">>>>>>>>>>> Recommendation metrics:")
    gt_ids_unique = len(np.unique(gt_ids))
    predicted_ids_unique = len(np.unique(total_predicted_ids)) - 1
    logger.info(f"Unique (GT): {gt_ids_unique}, Unique (predicted): {predicted_ids_unique}")
    recall_ratio = n_recall_success / n_rec
    logger.info(f"Recall is successful (gt_id is in recommended ids): {recall_ratio:.4f}")
    rc100 = sum(x * n for x, n in zip(recall_top100, batch_sizes)) / total_samples
    rc300 = sum(x * n for x, n in zip(recall_top300, batch_sizes)) / total_samples
    rc500 = sum(x * n for x, n in zip(recall_top500, batch_sizes)) / total_samples
    mean_rc = (rc100 + rc300 + rc500) / 3
    logger.info(f"mean recall (%): {mean_rc:.4f}, recall top100 (%): {rc100:.4f}, top300 (%): {rc300:.4f}, top500( %): {rc500: .4f}")
    total_samples = sum(batch_sizes)
    rr1 = sum(x * n for x, n in zip(rerank_top1, batch_sizes)) / total_samples
    rr10 = sum(x * n for x, n in zip(rerank_top10, batch_sizes)) / total_samples
    rr50 = sum(x * n for x, n in zip(rerank_top50, batch_sizes)) / total_samples
    mean_rr = (rr1 + rr10 + rr50) / 3
    logger.info(f"mean rerank (%): {mean_rr:.4f}, rerank top1 (%): {rr1:.4f}, top10 (%): {rr10:.4f}, top50( %): {rr50:.4f}")
    ndcg1 = sum(x * n for x, n in zip(ndcg_1, batch_sizes)) / total_samples
    ndcg10 = sum(x * n for x, n in zip(ndcg_10, batch_sizes)) / total_samples
    ndcg50 = sum(x * n for x, n in zip(ndcg_50, batch_sizes)) / total_samples
    logger.info(f"mean ndcg1: {ndcg1:.4f}, ndcg10: {ndcg10:.4f}, ndcg50: {ndcg50:.4f}")
    logger.info(f'\n')

    model.train()

    return {
        'recall@1': rr1,
        'recall@10': rr10,
        'recall@50': rr50,
        'ndcg@1': ndcg1,
        'ndcg@10': ndcg10,
        'ndcg@50': ndcg50
    }

# validate on just 1 batch -> perplexity + recommendation part
def validate_one_iteration(batch, tokenizer, model, criterions, accelerator, args):
    (criterion_language, criterion_recall, criterion_rerank) = criterions

    # split data points in no rec / rec
    no_rec_idx = [i for i in range(len(batch["targets"])) if batch["targets"][i] == -1]
    has_rec_idx = [i for i in range(len(batch["targets"])) if batch["targets"][i] != -1]

    # metrics to track
    turn_nums, n_points, n_rec = [], 0, 0 # metadata
    ppl_losses, ppls = [], [] # response
    recall_losses, n_recall_success, recall_top100, recall_top300, recall_top500 = [], 0, [], [], [] # recall
    rerank_losses, total_rerank_top1, rerank_top1, rerank_top10, rerank_top50 = [], [0]*(len(no_rec_idx)+len(has_rec_idx)), [], [], []  # re-ranking
    ndcg_1, ndcg_10, ndcg_50 = [], [], []
    gt_ids, gt_ranks, total_predicted_ids = [], [], [-1]*(len(no_rec_idx)+len(has_rec_idx))  # recommendation

    embeds = []
    for i in range(batch["context_with_utterances"].shape[0]):
        embeds_context_i, embeds_utterance_i = [], []
        for j in range(batch["context_with_utterances"].shape[1]):
            if batch["context_with_utterances"][i, j].item() < len(tokenizer):
                embeds_i_j = accelerator.unwrap_model(model).language_model.transformer.wte(batch["context_with_utterances"][i, j])
            else:
                item_id = args.pseudo_tokens_to_item_ids[batch["context_with_utterances"][i, j].item()]
                embeds_i_j = accelerator.unwrap_model(model).compute_encoded_embeddings_for_items([item_id], args.items_db)[0]
                embeds_i_j = accelerator.unwrap_model(model).rerank_item_wte_mapper(embeds_i_j)
            if j < batch["context_lengths"][i]:
                embeds_context_i.append(embeds_i_j.unsqueeze(0))
            else:
                embeds_utterance_i.append(embeds_i_j.unsqueeze(0))
        embeds_context_i = torch.cat(embeds_context_i)
        embeds_utterance_i = torch.cat(embeds_utterance_i)
        embeds.append((embeds_context_i, embeds_utterance_i))
    embeds_no_rec = [embeds[x] for x in no_rec_idx]
    embeds_has_rec = [embeds[x] for x in has_rec_idx]

    n_points = len(batch["targets"])
    n_rec = len(has_rec_idx)

    # language only
    if len(no_rec_idx) > 0:
        language_targets = batch["context_with_utterances"][no_rec_idx][:, 1:].contiguous()
        language_targets[language_targets >= len(tokenizer)] = 0
        language_logits = accelerator.unwrap_model(model).forward_pure_language_turn(embeds_no_rec)

        language_targets_mask = torch.zeros_like(language_targets).float()
        for i in range(batch["context_with_utterances"][no_rec_idx].shape[0]):
            context_length = batch["context_lengths"][no_rec_idx[i]]
            utterance_length = batch["utterance_lengths"][no_rec_idx[i]]
            language_targets_mask[i, context_length:(context_length+utterance_length-1)] = 1
        loss_ppl_batch = criterion_language(
            language_logits, language_targets, language_targets_mask, label_smoothing=-1, reduce="sentence")
        loss_ppl = loss_ppl_batch.mean()
        ppl_losses.append(loss_ppl.item())
        perplexity = np.exp(loss_ppl.item())
        ppls.append(perplexity)

        del loss_ppl_batch

    # when there is a recommendation to make
    if len(has_rec_idx) > 0:
        # recall
        previous_ids = None
        if args.previous_recommended_ids_negative:
            previous_ids = [batch["previous_recommended_ids"][x] for x in has_rec_idx]
        recall_logits, recall_true_index, language_logits, language_targets, _ = accelerator.unwrap_model(
            model).forward_recall(
            batch["indices"][has_rec_idx],
            batch["context_with_utterances"][has_rec_idx],
            embeds_has_rec,
            batch["context_lengths"][has_rec_idx],
            batch["targets"][has_rec_idx],
            args.num_samples_recall_train,
            previous_recommended_ids=previous_ids,
        )

        # recall items loss
        recall_targets = torch.LongTensor(recall_true_index).to(accelerator.device)
        loss_recall = criterion_recall(recall_logits, recall_targets)
        recall_losses.append(loss_recall.item())
        del loss_recall, recall_targets

        # language loss in recall turn, REC_TOKEN, Language on conditional generation
        language_targets_mask = torch.zeros_like(language_targets).float()
        for i in range(batch["context_with_utterances"][has_rec_idx].shape[0]):
            context_length = batch["context_lengths"][has_rec_idx[i]]
            utterance_length = batch["utterance_lengths"][has_rec_idx[i]]
            language_targets_mask[i, (context_length-1):(context_length+utterance_length)] = 1
        language_targets[language_targets >= len(tokenizer)] = 0
        loss_ppl_batch = criterion_language(
            language_logits, language_targets, language_targets_mask, label_smoothing=-1, reduce="sentence")
        loss_ppl = loss_ppl_batch.mean()
        ppl_losses.append(loss_ppl.item())
        perplexity = np.exp(loss_ppl.item())
        ppls.append(perplexity)

        del loss_ppl, language_logits, language_targets

        recalled_ids = accelerator.unwrap_model(model).validation_perform_recall(
            batch["contexts"][has_rec_idx],
            batch["context_lengths"][has_rec_idx],
            args.validation_recall_size
        )

        for i in range(len(recalled_ids)):
            recommended_id = batch["targets"][has_rec_idx[i]]
            if recommended_id in recalled_ids[i]:
                gt_ranks.append(recalled_ids[i].index(recommended_id))
            else:
                gt_ranks.append(len(recalled_ids) + 1)
            recall_top100.append(int(recommended_id in recalled_ids[i][:100]))
            recall_top300.append(int(recommended_id in recalled_ids[i][:300]))
            recall_top500.append(int(recommended_id in recalled_ids[i][:500]))
            turn_nums.append(batch["turn_nums"][has_rec_idx[i]])

        # re-ranking
        rerank_logits = accelerator.unwrap_model(model).validation_perform_rerank(
            batch["contexts"][has_rec_idx],
            batch["context_lengths"][has_rec_idx],
            recalled_ids
        )
        n_recall_success_batch = 0

        # re-ranking loss
        loss_rerank = 0
        for i in range(rerank_logits.shape[0]):
            recommended_id = batch["targets"][has_rec_idx[i]]
            reranks = np.argsort(rerank_logits[i].cpu().detach().numpy())[::-1]
            reranked_ids = [recalled_ids[i][x] for x in reranks]
            
            # 计算不同K值的NDCG
            ndcg_1.append(calculate_ndcg(recommended_id, reranked_ids, 1))
            ndcg_10.append(calculate_ndcg(recommended_id, reranked_ids, 10))
            ndcg_50.append(calculate_ndcg(recommended_id, reranked_ids, 50))
            
            total_rerank_top1[has_rec_idx[i]] = int(recommended_id in reranked_ids[:1])
            rerank_top1.append(int(recommended_id in reranked_ids[:1]))
            rerank_top10.append(int(recommended_id in reranked_ids[:10]))
            rerank_top50.append(int(recommended_id in reranked_ids[:50]))

            # counts of movies
            gt_ids.append(recommended_id)
            predicted_id = recalled_ids[i][reranks[0]]
            total_predicted_ids[has_rec_idx[i]] = predicted_id

            if recommended_id not in recalled_ids[i]:
                continue
            n_recall_success += 1
            n_recall_success_batch += 1
            rerank_true_index = recalled_ids[i].index(recommended_id)
            rerank_targets = torch.LongTensor([rerank_true_index]).to(accelerator.device)
            loss_rerank_i = criterion_rerank(rerank_logits[i].unsqueeze(0), rerank_targets)
            loss_rerank += loss_rerank_i.item()

            del rerank_targets
        loss_rerank /= max(1, n_recall_success_batch)
        if loss_rerank > 0:
            rerank_losses.append(loss_rerank)

        del loss_rerank, rerank_logits

    metadata = (turn_nums, n_points, n_rec)
    response = (ppl_losses, ppls)
    recall = (recall_losses, n_recall_success, recall_top100, recall_top300, recall_top500)
    rerank = (rerank_losses, total_rerank_top1, rerank_top1, rerank_top10, rerank_top50)
    ndcg = (ndcg_1, ndcg_10, ndcg_50)
    recommendation = (gt_ids, gt_ranks, total_predicted_ids)

    return metadata, response, recall, rerank, ndcg, recommendation

# validate on just 1 batch I->I response generation part
def validate_language_metrics_batch_embeds(tokenizer, batch, model, accelerator, preds, args):
    model_to_use = accelerator.unwrap_model(model).language_model
    REC_wte = accelerator.unwrap_model(model).get_rec_token_wtes()
    REC_END_wte = accelerator.unwrap_model(model).get_rec_end_token_wtes()
    suffix_ids = torch.tensor([32, 25]).to(accelerator.device)
    suffix_embeds = model_to_use.transformer.wte(suffix_ids)

    not_repeated_idx = [i for i in range(len(batch["repeated"])) if batch["repeated"][i] == 0]

    sources = []
    gt_rec, raw_gt_sens, gt_sens = [], [], []
    pred_rec, gen_sens, tok_gen_sens = [], [], []
    if len(not_repeated_idx) > 0:
        for i in range(batch["contexts_padded_left"][not_repeated_idx].shape[0]):
            source = tokenizer.decode(batch["raw_contexts"][not_repeated_idx[i]], skip_special_tokens=True)
            sources.append(source)
            embeds_i = []
            for j in range(batch["contexts_padded_left"][not_repeated_idx].shape[1]):
                if batch["contexts_padded_left"][not_repeated_idx][i, j].item() == tokenizer.pad_token_id:
                    continue
                if batch["contexts_padded_left"][not_repeated_idx][i, j].item() < len(tokenizer):
                    embeds_i_j = accelerator.unwrap_model(model).language_model.transformer.wte(batch["contexts_padded_left"][not_repeated_idx][i, j])
                    embeds_i.append(embeds_i_j.unsqueeze(0))
                else:
                    pred = args.pseudo_tokens_to_item_ids[batch["contexts_padded_left"][not_repeated_idx][i, j].item()]
                    total_pooled = accelerator.unwrap_model(model).annoy_base_rerank.get_item_vector(pred)
                    total_pooled = np.asarray(total_pooled)
                    item_embeds = torch.tensor(total_pooled, dtype=torch.float).unsqueeze(0).to(accelerator.device)
                    embeds_i += [REC_wte[0], item_embeds, REC_END_wte[0]]
            # add the prediction on that data point
            if preds[not_repeated_idx[i]] != -1:
                pred = preds[not_repeated_idx[i]]
                total_pooled = accelerator.unwrap_model(model).annoy_base_rerank.get_item_vector(pred)
                total_pooled = np.asarray(total_pooled)
                item_embeds = torch.tensor(total_pooled, dtype=torch.float).unsqueeze(0).to(accelerator.device)
                embeds_i += [REC_wte[0], item_embeds, REC_END_wte[0]]
            embeds_i = torch.cat(embeds_i)
            embeds_i = torch.cat((embeds_i, suffix_embeds))
            embeds_i = embeds_i.unsqueeze(0)

            gen_ids_i = make_generation_embeds(embeds_i, model_to_use, args)

            raw_gen_sens_i = tokenizer.batch_decode(gen_ids_i, skip_special_tokens=True)[0]
            if args.placeholder_token in raw_gen_sens_i:
                pred_rec.append(1)
            else:
                pred_rec.append(0)
            gen_sens_i = "A: " + " ".join(raw_gen_sens_i.replace("\n", " ").split())
            gen_sens.append(gen_sens_i)
            tok_gen_sens_i = ("A: " + raw_gen_sens_i).strip().split()
            tok_gen_sens.append(tok_gen_sens_i)

        for i in range(len(batch["targets"][not_repeated_idx])):
            if batch["targets"][not_repeated_idx][i] != -1:
                gt_rec.append(1)
            else:
                gt_rec.append(0)
        raw_gt_sens = tokenizer.batch_decode(batch["raw_utterances"][not_repeated_idx],skip_special_tokens=True)
        raw_gt_sens = [" ".join(x.replace("\n", " ").split()) for x in raw_gt_sens]
        gt_sens = tokenizer.batch_decode(batch["utterances"][not_repeated_idx], skip_special_tokens=True)
        gt_sens = [" ".join(x.replace("\n", " ").split()) for x in gt_sens]

    ground_truths = (gt_rec, raw_gt_sens, gt_sens)
    predicted = (pred_rec, gen_sens, tok_gen_sens)

    return sources, ground_truths, predicted

# response generation with the LM, using directly word embeddings (not tokens)
def make_generation_embeds(inputs_embeds, model_to_use, args):
    with torch.no_grad():
        if args.generation_method == "beam_search":
            generated = model_to_use.generate(
                inputs_embeds = inputs_embeds,
                max_new_tokens = args.utt_max_length,
                num_return_sequences=1,
                num_beams=args.num_beams,
                eos_token_id=628
            )
        elif args.generation_method == "diverse_beam_search":
            generated = model_to_use.generate(
                inputs_embeds = inputs_embeds,
                max_new_tokens = args.utt_max_length,
                num_return_sequences=1,
                num_beams=args.num_beams,
                num_beam_groups=args.num_beam_groups,
                diversity_penalty=args.diversity_penalty,
                eos_token_id=628
            )
        elif args.generation_method == "top_k_sampling":
            generated = model_to_use.generate(
                inputs_embeds=inputs_embeds,
                max_new_tokens=args.utt_max_length,
                num_return_sequences=1,
                do_sample=True,
                num_beams=args.num_beams,
                top_k=args.top_k,
                temperature=args.sampling_temperature,
                eos_token_id=628
            )

    return generated
