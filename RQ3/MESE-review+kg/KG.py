import math
import os

import torch
from torch import nn
from torch.nn import functional as F
from torch_geometric.nn import RGCNConv
import json

class KGModel(nn.Module):
    def __init__(
        self, hidden_size, 
        n_entity, num_relations, num_bases, edge_index, edge_type,
    ):
        super(KGModel, self).__init__()
        self.hidden_size = hidden_size
        entity_hidden_size = hidden_size // 2
        self.kg_encoder = RGCNConv(entity_hidden_size, entity_hidden_size, num_relations=num_relations,
                                   num_bases=num_bases)
        self.node_embeds = nn.Parameter(torch.empty(n_entity, entity_hidden_size))
        stdv = math.sqrt(6.0 / (self.node_embeds.size(-2) + self.node_embeds.size(-1)))
        self.node_embeds.data.uniform_(-stdv, stdv)
        self.edge_index = nn.Parameter(edge_index, requires_grad=False)
        self.edge_type = nn.Parameter(edge_type, requires_grad=False)
        self.entity_proj1 = nn.Sequential(
            nn.Linear(entity_hidden_size, entity_hidden_size // 2),
            nn.ReLU(),
            nn.Linear(entity_hidden_size // 2, entity_hidden_size),
        )
        self.entity_proj2 = nn.Linear(entity_hidden_size, hidden_size)


    def set_and_fix_node_embed(self, node_embeds: torch.Tensor):
        self.node_embeds.data = node_embeds
        self.node_embeds.requires_grad_(False)

    def get_entity_embeds(self):
        node_embeds = self.node_embeds
        
        # 添加验证检查
        if not torch.isfinite(node_embeds).all():
            raise ValueError("节点嵌入包含NaN或无限值")
        
        if self.edge_index.max() >= node_embeds.size(0):
            raise ValueError(f"边索引包含无效的节点索引。最大索引: {self.edge_index.max()}, "
                            f"但节点总数只有 {node_embeds.size(0)}")
        
        if self.edge_type.max() >= self.kg_encoder.num_relations:
            raise ValueError(f"边类型包含无效的关系索引。最大类型: {self.edge_type.max()}, "
                            f"但关系总数只有 {self.kg_encoder.num_relations}")
        
        # 将张量移动到与node_embeds相同的设备上
        edge_index = self.edge_index.to(node_embeds.device)
        edge_type = self.edge_type.to(node_embeds.device)
        
        entity_embeds = self.kg_encoder(node_embeds, edge_index, edge_type) + node_embeds
        entity_embeds = self.entity_proj1(entity_embeds) + entity_embeds
        entity_embeds = self.entity_proj2(entity_embeds)
        return entity_embeds

    def save(self, save_dir):
        os.makedirs(save_dir, exist_ok=True)
        state_dict = {k: v for k, v in self.state_dict().items() if 'edge' not in k}
        save_path = os.path.join(save_dir, 'kg_model.pt')
        torch.save(state_dict, save_path)

    def load(self, load_dir):
        load_path = os.path.join(load_dir, 'kg_model.pt')
        missing_keys, unexpected_keys = self.load_state_dict(
            torch.load(load_path, map_location=torch.device('cpu')), strict=False
        )
        print(missing_keys, unexpected_keys)


class DBpedia:
    def __init__(self, dataset_dir, debug=False):
        self.debug = debug
        self.dataset_dir = dataset_dir  
        # self.dataset_dir = os.path.join('data', dataset)

        with open(os.path.join(self.dataset_dir, './dbpedia_subkg.json'), 'r', encoding='utf-8') as f:
            self.entity_kg = json.load(f)
        with open(os.path.join(self.dataset_dir, './entity2id.json'), 'r', encoding='utf-8') as f:
            self.entity2id = json.load(f)
        # with open(os.path.join(self.dataset_dir, './relation2id.json'), 'r', encoding='utf-8') as f:
        #     self.relation2id = json.load(f)
        # with open(os.path.join(self.dataset_dir, './item_ids.json'), 'r', encoding='utf-8') as f:
        #     self.item_ids = json.load(f)

        self._process_entity_kg()
        # train_item, train_entity = self._build_item_entity_mapping('train')
        # valid_item, valid_entity = self._build_item_entity_mapping('valid')
        # test_item, test_entity = self._build_item_entity_mapping('test')

        # self.item2entity_id = {}
        # for item_id, entity_id in zip(train_item, train_entity):
        #     self.item2entity_id[item_id] = entity_id

        # for item_id, entity_id in zip(valid_item, valid_entity):
        #     self.item2entity_id[item_id] = entity_id

        # for item_id, entity_id in zip(test_item, test_entity):
        #     self.item2entity_id[item_id] = entity_id
    
    # def _build_item_entity_mapping(self, mode):
    #     if mode == "train":
    #         ori_train_path = './DATA/train_data_processed'
    #         process_train_path = './DATA/nltk/train_data.json'
    #     elif mode == "test":
    #         ori_train_path = './DATA/test_data_processed'
    #         process_train_path = './DATA/nltk/test_data.json'
    #     elif mode == 'valid':
    #         ori_train_path = './DATA/valid_data_processed'
    #         process_train_path = './DATA/nltk/valid_data.json'


    #     ori_train_data = torch.load(ori_train_path)
    #     with open(process_train_path, 'r', encoding='utf-8') as f:
    #         process_train_data = json.load(f)
        
    #     item_id_list = []
    #     process_id_list = []
    #     for row_cnt, (row_ori_data, row_process_data) in enumerate(zip(ori_train_data, process_train_data)):
    #         conv = row_process_data['dialog']
    #         # print(row_ori_data)
    #         for col_cnt, mess in enumerate(row_ori_data[1]):
    #             # print(mess)
    #             item_id = mess[1]
    #             process_text = conv[col_cnt]['movies']
    #             # filter out the element start with "@"
    #             # filter_process_text = [word for word in process_text if word.startswith('@')]
    #             if item_id != None:
    #                 for cnt, item_id_ in enumerate(item_id):
    #                     item_id_list.append(item_id_)
    #                     final_text = process_text[cnt]
    #                     final_entity_id = self.entity2id[final_text]
    #                     process_id_list.append(final_entity_id)

    #     return item_id_list, process_id_list
    
    def _process_entity_kg(self):
        edge_list = set()  # [(entity, entity, relation)]
        for entity in self.entity2id.values():
            if str(entity) not in self.entity_kg:
                continue
            for relation_and_tail in self.entity_kg[str(entity)]:
                edge_list.add((entity, relation_and_tail[1], relation_and_tail[0]))
                edge_list.add((relation_and_tail[1], entity, relation_and_tail[0]))
        edge_list = list(edge_list)

        edge = torch.as_tensor(edge_list, dtype=torch.long)
        self.edge_index = edge[:, :2].t()
        self.edge_type = edge[:, 2]
        # self.num_relations = len(self.relation2id)
        self.pad_entity_id = max(self.entity2id.values()) + 1
        self.num_entities = max(self.entity2id.values()) + 2

        self.relation2id = dict()
        for h, t, r in edge_list:
            if r not in self.relation2id:
                self.relation2id[r] = len(self.relation2id)
        self.num_relations = len(self.relation2id) + 1

    def get_entity_kg_info(self):
        kg_info = {
            'edge_index': self.edge_index,
            'edge_type': self.edge_type,
            'num_entities': self.num_entities,
            'num_relations': self.num_relations,
            'pad_entity_id': self.pad_entity_id,
        }
        return kg_info
