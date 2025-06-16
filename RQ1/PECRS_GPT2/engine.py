import numpy as np
import torch
import tqdm
import time
import copy
import gc
from engine_validation import validate


# overall training loop, on the entire dataset
def training_loop(train_dataloader, valid_dataloader, test_dataloader, tokenizer, model, optimizer, scheduler, criterions, logger, accelerator, args):
    # 使用 recall@k 作为指标，越大越好
    best_valid_recall = float('-inf')
    best_model_state = None
    early_stop_count = 0
    
    if args.validate and args.epoch_0:
        validate(0, valid_dataloader, tokenizer, model, criterions, logger, accelerator, args)
        model.train()
    else:
        model.eval()
        accelerator.unwrap_model(model).annoy_base_constructor()
        model.train()

    ppls, all_loss_ppl, all_loss_recall, all_loss_rerank = [], [], [], []
    for ep in range(1, args.num_epochs + 1):
        if args.previous_recommended_ids_negative:
            args.previous_count = []
        # training round of the epoch
        logger.info("\n")
        logger.info(f"Training epoch {ep}...")
        model.train()
        update_count, optim_count = 0, 0
        for batch in tqdm.tqdm(train_dataloader, disable=not accelerator.is_main_process):
            batch = {k: v.cuda() if isinstance(v, torch.Tensor) else v for k, v in batch.items()}
            with torch.cuda.amp.autocast(enabled=True):
                # batch size of train_dataloader is 1
                avg_ppl, loss_ppl, loss_recall, loss_rerank = train_one_iteration(
                    batch, tokenizer, model, criterions, accelerator, args)
                avg_ppl = np.nan_to_num(avg_ppl)
                loss_ppl = np.nan_to_num(loss_ppl)
                loss_recall = np.nan_to_num(loss_recall)
                loss_rerank = np.nan_to_num(loss_rerank)
                ppls.append(avg_ppl)
                all_loss_ppl.append(loss_ppl)
                all_loss_recall.append(loss_recall)
                all_loss_rerank.append(loss_rerank)
                update_count += 1
                if args.only_tune_new_tokens:
                    accelerator.unwrap_model(model).language_model.transformer.wte.weight.grad[args.n_original_tokens] = 0
                optimizer.step()
                if scheduler is not None:
                    scheduler.step()
                optimizer.zero_grad()

                if (update_count % args.num_gradients_accumulation == args.num_gradients_accumulation - 1) or (update_count == len(train_dataloader)):
                    # update for gradient accumulation
                    optim_count += 1
                    lr = optimizer.param_groups[0]['lr']

                if (update_count % args.print_every == 0):
                    median_ppl = np.percentile(np.array(ppls), 50)
                    mean_ppl = np.mean(np.array(ppls))
                    mean_loss_ppl = np.mean(np.array(all_loss_ppl))
                    mean_loss_recall = np.mean(np.array(all_loss_recall))
                    mean_loss_rerank = np.mean(np.array(all_loss_rerank))
                    lr = optimizer.param_groups[0]['lr']
                    logger.info(f"Epoch {ep}, Batch {update_count}, # optim steps: {optim_count}, LR: {lr:.10f}")
                    logger.info(f"median ppl: {median_ppl:.4f}, mean ppl: {mean_ppl:.4f}, loss ppl: {mean_loss_ppl: .4f}, loss recall: {mean_loss_recall: .4f}, loss_rerank: {mean_loss_rerank: .4f}")
                    ppls, all_loss_ppl, all_loss_recall, all_loss_rerank = [], [], [], []

        # if (update_count % args.eval_every == 0):
        # 在验证集上评估
        valid_metrics = validate(ep, valid_dataloader, tokenizer, model, criterions, logger, accelerator, args)
        valid_recall = valid_metrics['ndcg@10']  # 或者使用 recall@k，k 根据实际需求设置
        
        # 如果当前模型在验证集上的 recall 更好
        if valid_recall > best_valid_recall:
            best_valid_recall = valid_recall
            early_stop_count = 0
            # 保存最佳模型状态
            best_model_state = copy.deepcopy(accelerator.unwrap_model(model).state_dict())
            # 保存最佳模型
            if args.save:
                save_path = args.model_saved_path + "best_model.pt"
                accelerator.save(best_model_state, save_path)
                logger.info(f"Saved best model! at {save_path} with valid recall@1: {best_valid_recall:.4f}")
            
            # 在测试集上评估当前最佳模型
            test_metrics = validate(ep, test_dataloader, tokenizer, model, criterions, logger, accelerator, args)
            logger.info(f"Test performance at best valid checkpoint - recall@1: {test_metrics['recall@1']:.4f}")
        else:
            early_stop_count += 1
            if early_stop_count >= args.early_stop_patience:
                logger.info(f"Early stopping at epoch {ep} due to no improvement in validation recall.")
                break
        model.train()

        if args.previous_recommended_ids_negative:
            previous_count = np.mean(args.previous_count)
            logger.info(f"Added {previous_count:.4f} hard negatives on average through previously mentioned movies")
            args.previous_count = []
        # validation round of the epoch
        # if args.validate:
        #     validate(ep, valid_dataloader, tokenizer, model, criterions, logger, accelerator, args)
        #     model.train()
        # if args.save:
        #     save_path = args.model_saved_path + str(ep) + ".pt"
        #     state_dict = accelerator.unwrap_model(model).state_dict()
        #     accelerator.save(state_dict, save_path)
        #     logger.info(f"saved model! at {save_path}")

    # 训练结束后，加载最佳模型并在测试集上进行最后的评估
    logger.info("\n=== Training finished! Evaluating best model on test set ===")
    if best_model_state is not None:
        accelerator.unwrap_model(model).load_state_dict(best_model_state)
        final_test_metrics = validate(args.num_epochs, test_dataloader, tokenizer, model, criterions, logger, accelerator, args)
        logger.info(f"Final test performance with best valid model - recall@1: {final_test_metrics['recall@1']:.4f}")

# training on 1 batch
def train_one_iteration(batch, tokenizer, model, criterions, accelerator, args):
    (criterion_language, criterion_recall, criterion_rerank_train) = criterions
    ppl_history = []
    all_loss_ppl, all_loss_recall, all_loss_rerank = [], [], []

    no_rec_idx = [i for i in range(len(batch["targets"])) if batch["targets"][i] == -1]
    has_rec_idx = [i for i in range(len(batch["targets"])) if batch["targets"][i] != -1]

    embeds = []
    for i in range(batch["context_with_utterances"].shape[0]):
        embeds_context_i, embeds_utterance_i = [], []
        for j in range(batch["context_with_utterances"].shape[1]):
            if batch["context_with_utterances"][i,j].item() < len(tokenizer):
                embeds_i_j = accelerator.unwrap_model(model).language_model.transformer.wte(batch["context_with_utterances"][i,j])
            else:
                item_id = args.pseudo_tokens_to_item_ids[batch["context_with_utterances"][i,j].item()]
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

    # data points without recommendation (just response generation aka language modeling)
    if len(no_rec_idx) > 0:
        with accelerator.autocast():
            language_targets = batch["context_with_utterances"][no_rec_idx][:, 1:].contiguous()
            language_targets[language_targets >= len(tokenizer)] = 0
            language_logits = accelerator.unwrap_model(model).forward_pure_language_turn(embeds_no_rec)
            language_targets_mask = torch.zeros_like(language_targets).float()
            for i in range(batch["context_with_utterances"][no_rec_idx].shape[0]):
                context_length = batch["context_lengths"][no_rec_idx[i]]
                utterance_length = batch["utterance_lengths"][no_rec_idx[i]]
                language_targets_mask[i, context_length:(context_length+utterance_length-1)] = 1
            loss_ppl = criterion_language(
                language_logits, language_targets, language_targets_mask, label_smoothing=args.ls, reduce='batch')
            perplexity = np.exp(min(300, torch.nan_to_num(loss_ppl).item()))
            ppl_history.append(perplexity)
            all_loss_ppl.append(loss_ppl.item())
            loss_ppl = args.gamma * loss_ppl
            accelerator.backward(loss_ppl)

            del loss_ppl, language_logits, language_targets
            gc.collect()

    # data points with recommended items
    if len(has_rec_idx) > 0:
        with accelerator.autocast():
            # recall
            previous_ids = None
            if args.previous_recommended_ids_negative:
                previous_ids = [batch["previous_recommended_ids"][x] for x in has_rec_idx]
            recall_logits, recall_true_index, language_logits, language_targets, encoded_items_embeddings = accelerator.unwrap_model(model).forward_recall(
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
            all_loss_recall.append(loss_recall.item())
            loss_recall = args.alpha * loss_recall
            # language loss in recall turn, REC_TOKEN, Language on conditional generation
            language_targets_mask = torch.zeros_like(language_targets).float()
            for i in range(batch["context_with_utterances"][has_rec_idx].shape[0]):
                context_length = batch["context_lengths"][has_rec_idx[i]]
                utterance_length = batch["utterance_lengths"][has_rec_idx[i]]
                language_targets_mask[i, (context_length-1):(context_length-1+utterance_length)] = 1
            language_targets[language_targets >= len(tokenizer)] = 0
            loss_ppl = criterion_language(
                language_logits, language_targets, language_targets_mask, label_smoothing=args.ls, reduce="batch")
            perplexity = np.exp(min(300, torch.nan_to_num(loss_ppl).item()))
            ppl_history.append(perplexity)
            all_loss_ppl.append(loss_ppl.item())
            loss_ppl = args.gamma * loss_ppl

            # rerank
            encoded_items_transfer = None
            if args.tie_sampled_ids_recall_rerank:
                encoded_items_transfer = encoded_items_embeddings
            rerank_logits, rerank_true_index = accelerator.unwrap_model(model).forward_rerank(
                batch["indices"][has_rec_idx],
                batch["contexts"][has_rec_idx],
                batch["context_lengths"][has_rec_idx],
                batch["targets"][has_rec_idx],
                args.num_samples_rerank_train,
                encoded_items_embeddings=encoded_items_transfer,
                previous_recommended_ids=None,
            )
            rerank_logits /= args.temperature

            # rerank loss
            rerank_targets = torch.LongTensor(rerank_true_index).to(accelerator.device)
            loss_rerank = criterion_rerank_train(rerank_logits, rerank_targets)
            all_loss_rerank.append(loss_rerank.item())
            loss_rerank = args.beta * loss_rerank
            # 总loss
            total_loss = loss_recall + loss_rerank + loss_ppl
            accelerator.backward(total_loss)

            del loss_ppl, language_logits, language_targets, loss_recall, recall_logits, recall_targets
            gc.collect()

    mean_ppl_history = np.mean(ppl_history)
    mean_loss_ppl = np.mean(all_loss_ppl)
    mean_loss_recall = np.mean(all_loss_recall)
    mean_loss_rerank = np.mean(all_loss_rerank)

    return mean_ppl_history, mean_loss_ppl, mean_loss_recall, mean_loss_rerank
