import os
import torch
import logging
from tqdm import tqdm
from torch.utils.data import DataLoader
import torch.optim as optim
from resource.model.rec_model import rec_model
from resource.input.session_dataset import SessionDataset
from resource.option.train_option import TrainOption
from resource.option.dataset_option import DatasetOption
from resource.tensor_nl_interpreter import TensorNLInterpreter
from resource.util.file_util import mkdir_if_necessary
from resource.util.json_writer import JsonWriter
from resource.input.session_dataset import my_collate_fn
import  transformers
import json
import numpy as np

engine_logger = logging.getLogger("main.rec_engine")
torch.set_default_tensor_type(torch.FloatTensor)
class Rec_Engine:
    def __init__(self,
                 model: rec_model,
                 train_dataset: SessionDataset,
                 test_dataset: SessionDataset,
                 valid_dataset: SessionDataset,
                 graph=None,
                 edge_list=None,
                 toca=None,
                 topics_num=None
                 ):

        self.model = model
        edge_list = list(set(edge_list))
        self.edge_sets = torch.LongTensor(edge_list)
        self.edge_idx = self.edge_sets[:, :2].t()
        self.edge_type = self.edge_sets[:, 2]

        self.toca = toca
        self.graph = graph
        self.train_dataloader = DataLoader(train_dataset,
                                           batch_size=TrainOption.train_batch_size,
                                           shuffle=True,
                                           collate_fn=lambda x:my_collate_fn(x),
                                           num_workers=TrainOption.data_load_worker_num,pin_memory=True)
        self.test_dataloader = DataLoader(test_dataset,
                                          batch_size=TrainOption.test_batch_size,
                                          shuffle=False,
                                          collate_fn=lambda x:my_collate_fn(x),
                                          num_workers=TrainOption.data_load_worker_num,pin_memory=True)
        self.valid_dataloader = DataLoader(valid_dataset,
                                           batch_size=TrainOption.valid_batch_size,
                                           shuffle=False,
                                           collate_fn=lambda x:my_collate_fn(x),
                                           num_workers=TrainOption.data_load_worker_num,pin_memory=True)

        self.topics_num = topics_num
        self.tensor_nl_interpreter = TensorNLInterpreter(vocab=self.topics_num)
        self.json_writer = JsonWriter()
        self.init_optim()
        movie_ids_filename = DatasetOption.movie_ids if DatasetOption.dataset == "Redial" else DatasetOption.movie_ids_TG
        self.movie_ids = json.load(open(movie_ids_filename, 'r', encoding='utf-8'))


    def init_optim(self):
        param_optimizer = list(self.model.named_parameters())  # 模型参数名字列表
        encoder_name = ['topath_encoder', 'profile_encoder', 'his_encoder']
        graph_encoder_name = ['RGCN','topic_embedder', 'GCN']

        optimizer_bert_parameters = [
                p for n, p in param_optimizer
                if any(nd in n for nd in encoder_name)
            ]
        optimizer_graph_encoder_parameters = [
            p for n, p in param_optimizer
            if any(nd in n for nd in graph_encoder_name)
        ]
        optimizer_decoder_parameters = [
                p for n, p in param_optimizer
                if not any(nd in n for nd in encoder_name) and not any(nd in n for nd in graph_encoder_name)
            ]

        self.optimizer_decoder = optim.Adam(optimizer_decoder_parameters, lr=1e-3)
        self.optimizer_graph_encoder = optim.Adam(optimizer_graph_encoder_parameters, lr=5e-4)
        self.optimizer_bert = transformers.AdamW(optimizer_bert_parameters,lr=1e-5)
        self.optimizer = [self.optimizer_decoder, self.optimizer_graph_encoder, self.optimizer_bert]


    def train(self,pretrain=False):

        optim_interval = int(TrainOption.efficient_train_batch_size / TrainOption.train_batch_size)
        global_step = 0
        best_ndcg10 = 0
        engine_logger.info("optim interval = {}".format(optim_interval))
        if pretrain:
            #********************************************stage1 for pretraining********************************************
            for epoch in range(TrainOption.epoch1+1 if DatasetOption.dataset=="TG" else TrainOption.epoch1):#since TG datasest is smaller
                pbar = tqdm(self.train_dataloader)
                engine_logger.info("EPOCH {}".format(epoch))
                # FOR BATCH
                for batch_data in pbar:
                    subgraphs = [self.edge_idx.to(TrainOption.device), self.edge_type.to(TrainOption.device)]
                    [batch_data,all_movies,identities ] = batch_data

                    sub_graph = batch_data[-1].to(TrainOption.device)
                    batch_data = [data.to(TrainOption.device) for data in batch_data[:-1]]
                    global_step += 1
                    do_optim = (global_step % optim_interval == 0)

                    # LOAD DATA
                    targets = batch_data[3]
                    outputs, p_relation, q_relation = self.model.forward(graph=subgraphs, inputs=batch_data,
                                                                         pretrain=True)
                    loss1 = self.model.loss(
                        outputs.reshape(-1, self.topics_num + 1),
                        targets.reshape(-1).detach())
                    loss2 = self.model.loss_KL(p_relation, q_relation.detach(), es_mask=batch_data[-1])
                    loss3 = self.model.loss_pretrain(p_relation, q_relation, batch_data[-1], sub_graph)
                    loss=loss1+ TrainOption.lambda_1*loss3
                    print("step:",global_step," NLL loss: {:.4f}".format(loss1.item()), " KL loss: {:.4f}".format(loss2.item()), " BCE loss: {:.4f}".format(loss3.item())
                          ,"KL loss is useless in pretrain process")
                    del outputs, p_relation, q_relation
                    # OPTIMIZATION
                    loss /= optim_interval
                    loss.backward()
                    loss_info = "loss: {:.4f}".format(loss.item())
                    pbar.set_description("TASK-ID: {}.log - ".format(TrainOption.task_uuid) + loss_info)
                    if do_optim:
                        for optimizer in self.optimizer:
                            optimizer.step()
                            optimizer.zero_grad()

                    # LOG LOSS INFO
                    if global_step % TrainOption.log_loss_interval == 0:
                        engine_logger.info("Pretrain STEP: {}, loss {}".format(global_step, loss_info))

                # EVALUATION
                # valid
                metrics_valid = self.test(self.valid_dataloader, mode="valid", pretrain=True)
                metric_str_valid = "(" + "-".join(["{:.3f}".format(x) for x in metrics_valid]) + ")"
                engine_logger.info("STEP {}, valid metric: {}".format(global_step,metric_str_valid))

                # test
                metrics = self.test(self.test_dataloader, pretrain=True)
                metric_str = "(" + "-".join(["{:.3f}".format(x) for x in metrics]) + ")"
                engine_logger.info("STEP {}, test metric: {}".format(global_step,metric_str))

                ckpt_filename = DatasetOption.ckpt_filename_template.format(
                    dataset=DatasetOption.dataset,
                    task=DatasetOption.task,
                    uuid=TrainOption.task_uuid,
                    global_step=global_step)
                mkdir_if_necessary(os.path.dirname(ckpt_filename))
                self.model.dump(ckpt_filename)
                engine_logger.info("dump model checkpoint to {}\n".format(ckpt_filename))


        else:
            # ******************************************** stage2 for train ********************************************
            for epoch in range(TrainOption.epoch2+3 if DatasetOption.dataset=="TG" else TrainOption.epoch2):
                pbar = tqdm(self.train_dataloader)
                engine_logger.info("EPOCH {}".format(epoch))
                for batch_data in pbar:
                    subgraphs = [self.edge_idx.to(TrainOption.device), self.edge_type.to(TrainOption.device)]
                    [batch_data,all_movies,identities ] = batch_data

                    sub_graph = batch_data[-1].to(TrainOption.device)
                    batch_data=[data.to(TrainOption.device) for data in batch_data[:-1]]
                    global_step += 1
                    do_optim = (global_step % optim_interval == 0)

                    # LOAD DATA
                    targets = batch_data[3]
                    outputs, p_relation, q_relation, gumbel_relation = self.model.forward(graph=subgraphs,inputs=batch_data,pretrain=False)
                    loss1 = self.model.loss(outputs.reshape(-1, self.topics_num+1), targets.reshape(-1).detach())
                    loss2 = self.model.loss_KL(p_relation, q_relation.detach() ,es_mask = batch_data[-1])
                    loss3 = self.model.loss_pretrain(p_relation, q_relation, batch_data[-1], sub_graph)
                    loss=loss1+TrainOption.lambda_0*loss2+ TrainOption.lambda_2*loss3
                    print("step:",global_step," NLL loss: {:.4f}".format(loss1.item()), " KL loss: {:.4f}".format(loss2.item()), " BCE loss: {:.4f}".format(loss3.item()))

                    del outputs, p_relation, q_relation

                    loss /= optim_interval
                    loss.backward()
                    loss_info = "loss: {:.4f}".format(loss.item())
                    pbar.set_description("TASK-ID: {}.log - ".format(TrainOption.task_uuid) + loss_info )

                    if do_optim:
                        for optimizer in self.optimizer:
                            optimizer.step()
                            optimizer.zero_grad()

                    # EVALUATION
                    if global_step % TrainOption.test_eval_interval == 0:
                        # VALID
                        metrics_valid = self.test(self.valid_dataloader, mode="valid")
                        metric_str_valid = "(" + "-".join(["{:.3f}".format(x) for x in metrics_valid]) + ")"
                        engine_logger.info("[Valid]: STEP {}, metric: recall@1-recall@10-recall@50-ndcg@5-ndcg@10-ndcg@50: {}".format(global_step, metric_str_valid))

                        if metrics_valid[-1] > best_ndcg10:
                            best_ndcg10 = metrics_valid[-1]
                            ckpt_filename = './data/ckpt/ckpt_redial_rec.model.ckpt'
                            mkdir_if_necessary(os.path.dirname(ckpt_filename))
                            self.model.dump(ckpt_filename)
                            engine_logger.info("dump model checkpoint to {}\n".format(ckpt_filename))

                            # TEST
                            metrics= self.test(self.test_dataloader)
                            metric_str = "(" + "-".join(["{:.3f}".format(x) for x in metrics]) + ")"
                            engine_logger.info("[Test]: STEP {}, metric: recall@1-recall@10-recall@50-ndcg@5-ndcg@10-ndcg@50: {}".format(global_step, metric_str))


                    if global_step % TrainOption.log_loss_interval == 0:
                        engine_logger.info("STEP: {}, loss {}".format(global_step, loss_info))


    def test(self, dataloader, mode="test" , pretrain = False):
        """test or valid"""
        assert mode == "test" or mode == "valid"
        self.model.eval()
        all_targets_index = []
        all_outputs_index = []
        all_movie_list = []
        engine_logger.info("{} START INFERENCE ...".format(mode.upper()))
        pbar = tqdm(dataloader)

        with torch.no_grad():
            for batch_data in pbar:
                pbar.set_description("TASK-ID: {}.log".format(TrainOption.task_uuid))

                # LOAD DATA
                subgraphs = [self.edge_idx.to(TrainOption.device), self.edge_type.to(TrainOption.device)]
                [ batch_data,all_movies,identities] = batch_data

                batch_data = [data.to(TrainOption.device) for data in batch_data[:-1]]
                # LOAD DATA
                targets = batch_data[3]

                outputs = self.model.forward(graph=subgraphs, inputs=batch_data,
                                                          pretrain=pretrain)  # outputs:B,T   hit_cal:B,T,vocab_size

                all_outputs_index+=outputs.tolist()  #[all,T]
                all_targets_index+=targets.tolist()  #[all,T]
                all_movie_list.extend(all_movies)
        engine_logger.info("{} INFERENCE FINISHED".format(mode.upper()))
        self.model.train()

        # metric
        metrics,cnt = self.eval_hit(all_targets_index, all_outputs_index,all_movie_list)
        # 更新日志输出，添加ndcg@50
        if mode == "valid":
            engine_logger.info("[Valid]: STEP, metric: recall@1-recall@10-recall@50-ndcg@5-ndcg@10-ndcg@50: {}".format(
                "(" + "-".join(["{:.3f}".format(x) for x in metrics]) + ")"))
        else:
            engine_logger.info("[Test]: STEP, metric: recall@1-recall@10-recall@50-ndcg@5-ndcg@10-ndcg@50: {}".format(
                "(" + "-".join(["{:.3f}".format(x) for x in metrics]) + ")"))
        

        return metrics

    def eval_hit(self,targets,hit_cals,all_movie_list):
        metrics_rec = {
            "recall@1": 0, "recall@10": 0, "recall@50": 0, 
            "ndcg@5": 0, "ndcg@10": 0, "ndcg@50": 0,
            "loss": 0, "count": 0
        }

        cnt = 0
        hit_cals = torch.tensor(hit_cals)
        hit_cals = hit_cals[:, torch.LongTensor(self.movie_ids)]
        _, hit_cals = torch.topk(hit_cals, k=100, dim=1)
        
        for target, hit_cal, all_movie in zip(targets, hit_cals.tolist(), all_movie_list):
            tgt = target
            target_idx = self.movie_ids.index(tgt)
            
            # 计算recall
            metrics_rec["recall@1"] += int(target_idx in hit_cal[:1])
            metrics_rec["recall@10"] += int(target_idx in hit_cal[:10])
            metrics_rec["recall@50"] += int(target_idx in hit_cal[:50])
            
            # 为NDCG准备相关性列表
            relevance_scores = [1 if idx == target_idx else 0 for idx in hit_cal[:50]]
            
            # 计算NDCG@5、NDCG@10和NDCG@50
            metrics_rec["ndcg@5"] += calculate_ndcg(relevance_scores, 5)
            metrics_rec["ndcg@10"] += calculate_ndcg(relevance_scores, 10)
            metrics_rec["ndcg@50"] += calculate_ndcg(relevance_scores, 50)
            
            metrics_rec["count"] += 1

        recall_1 = metrics_rec["recall@1"] / metrics_rec["count"]
        recall_10 = metrics_rec["recall@10"] / metrics_rec["count"]
        recall_50 = metrics_rec["recall@50"] / metrics_rec["count"]
        ndcg_5 = metrics_rec["ndcg@5"] / metrics_rec["count"]
        ndcg_10 = metrics_rec["ndcg@10"] / metrics_rec["count"]
        ndcg_50 = metrics_rec["ndcg@50"] / metrics_rec["count"]
        
        return [recall_1, recall_10, recall_50, ndcg_5, ndcg_10, ndcg_50], cnt
    
def calculate_ndcg(scores, k):
    """
    计算NDCG@k
    Args:
        scores: 预测的相关性分数列表
        k: 截断位置
    Returns:
        ndcg_k: NDCG@k的值
    """
    # 计算DCG@k
    dcg_k = 0.0
    for i in range(min(k, len(scores))):
        # 使用二进制相关性(0或1)
        rel = 1 if scores[i] else 0
        dcg_k += (2 ** rel - 1) / np.log2(i + 2)  # i+2 是因为log2从1开始
    
    # 计算IDCG@k (理想DCG)
    # 对于二进制相关性，IDCG就是把所有相关项排在前面
    ideal_scores = sorted(scores, reverse=True)
    idcg_k = 0.0
    for i in range(min(k, len(ideal_scores))):
        rel = 1 if ideal_scores[i] else 0
        idcg_k += (2 ** rel - 1) / np.log2(i + 2)
    
    # 计算NDCG@k
    ndcg_k = dcg_k / idcg_k if idcg_k > 0 else 0.0
    return ndcg_k