import os
import logging
import torch
import pickle
import csv
import json
from tqdm import tqdm

logger = logging.getLogger(__name__)

def write_pkl(obj, filename):
    with open(filename, 'wb') as f:
        pickle.dump(obj, f)

def extract_movie_ids(text):
    """从文本中提取@后面的电影ID"""
    import re
    movie_ids = re.findall(r'@(\d+)', ' '.join(text))
    return [int(mid) for mid in movie_ids] if movie_ids else []  # 如果没有电影ID则返回[0]

def extract_knowledge(entity):
    """从entity URL中提取知识"""
    return [url.split('/')[-1].replace('_', ' ') for url in entity] if entity else []

def replace_movie_ids_with_names(utt, items_db):
    text = utt['text']
    """将文本中的@item_id替换为电影名称，如果没有@则返回原文本"""
    import re
    text = ' '.join(text) if isinstance(text, list) else text
    if '@' not in text:  # 如果文本中没有@，直接返回
        return text, []
        
    movie_ids = re.findall(r'@(\d+)', text)
    replaced_text = text

    meta_list = []
    movie_cnt = len(utt['movies'])
    for mid in movie_ids:
        mid = int(mid)
        if mid in items_db:
            movie_name = items_db[mid]['movieName']
            movie_meta = items_db[mid]['meta']
            meta_list.append(movie_name + movie_meta)
            replaced_text = replaced_text.replace(f'@{mid}', movie_name)
        else:
            meta_list.append('')
    meta_list = meta_list[:movie_cnt]
    return replaced_text, meta_list

def read_pkl(filename):
    with open(filename, 'rb') as f:
        return pickle.load(f)

def load_and_cache_examples(args, tokenizer, mode, evaluate=False):
    # mode = 'test' if evaluate else 'train'
    # Load data features from cache or dataset file
    cached_features_file = os.path.join(args.data_dir, 'cached_nl_{}_{}_{}_{}_{}'.format(
        args.data_name,
        mode,
        list(filter(None, args.model_name_or_path.split('/'))).pop(),
        str(args.max_seq_length),
        str(args.max_target_length)))
    logger.info("Creating features from dataset file at %s", args.data_dir)
    features = convert_to_features(args, tokenizer, mode)
    print("Loaded number of instance:", len(features['resp']['source_ids']))

    logger.info("Saving features into cached file %s", cached_features_file)
    write_pkl(features, cached_features_file)
    return features

def get_item_dict(items_db):
    item_dict = {}
    for mid, item in items_db.items():
        item_dict[item['movieName']] = item['new_item_id']
    item_dict[len(item_dict)] = '<PAD>'
    return item_dict

def convert_to_features(args, tokenizer, mode):
    print("Special tokens mapping:")
    print(tokenizer.special_tokens_map)
    print("\nEncoding of special tokens:")
    print("'[knowledge]' ->", tokenizer.encode('[knowledge]'))
    print("'[item]' ->", tokenizer.encode('[item]'))
    sid = 21131
    print(f"Token {sid} decodes to:", tokenizer.decode([sid]))

    token = 0
    print(f"Token {token} decodes to:", tokenizer.decode([token]))

    # load items db
    csv_path = '/projects/prjs1158/KG/redail/MESE_review/DATA/nltk/movies_with_mentions.csv'
    items_db, _ = load_movie_data(csv_path, args)

    _, item_dict = old_load_movie_data(args)


    # train_path = '/projects/prjs1158/KG/redail/MESE_review/DATA/train_data_processed'
    # valid_path = '/projects/prjs1158/KG/redail/MESE_review/DATA/valid_data_processed'
    # test_path = '/projects/prjs1158/KG/redail/MESE_review/DATA/test_data_processed'

    # train_data = json.load(open(train_path))
    # valid_data = json.load(open(valid_path))
    # test_data = json.load(open(test_path))

    # get mapping from movie id to review info
    # name2review = get_name2meta(torch.load(train_path), torch.load(valid_path), torch.load(test_path))


    # 初始化数据字典，移除了'goal'任务
    data_dict = {
        'resp': {'source_ids':[], 'target_ids':[], 'item_ids':[]}, 
        'item': {'source_ids':[], 'target_ids':[], 'item_ids':[]}, 
        'know': {'source_ids':[], 'target_ids':[], 'item_ids':[]}
    }
    
    path = os.path.join(args.data_dir, '{}/nltk/{}_data.json'.format(args.data_name, mode))
    print('tokenizing {}'.format(path))

    with open(path, 'r', encoding='utf-8') as f:
        data = json.load(f)   

    
    # with open(path, 'r', encoding='utf-8') as infile:
    max_dia_len = 0
    avg_dia_len = []
    max_res_len = 0
    avg_res_len = []
    source_ids = []
    target_ids = []
    item_ids = []
    rec_index = []
    i = 0

    for d in tqdm(data):
        # print(line)
        # d = eval(line.strip())
        # print(d)
        conv = d['dialog']
        source_id = []
        source_know_id = []
        target_id = []

        # 处理第一个话语
        first_utt = conv[0]
        first_text, _ = replace_movie_ids_with_names(first_utt, items_db)
        # 从entity URL中提取知识
        knowledge = extract_knowledge(first_utt['entity'])
        # print(f'first_text: {first_text}')
        # movie_ids = extract_movie_ids(first_utt['text'])
        
        if knowledge:
            source_know_id += tokenizer.encode('[knowledge]' + '|'.join(knowledge))[1:]
        source_id += tokenizer.encode('[{}]'.format(first_utt['role']) + first_text)[1:]
        source_know_id += tokenizer.encode('[{}]'.format(first_utt['role']) + first_text)[1:]

        for utt in conv[1:]:
            if utt['role'] == 'Seeker':
                # 用户话语处理
                user_text, _ = replace_movie_ids_with_names(utt, items_db)
                source_id += tokenizer.encode('[user]' + user_text)[1:]
                knowledge = extract_knowledge(utt['entity'])
                # if knowledge:
                source_know_id += tokenizer.encode('[knowledge]' + '|'.join(knowledge))[1:]
                source_know_id += tokenizer.encode('[user]' + user_text)[1:]
                continue
            # 系统回复处理
            system_text, movies_text = replace_movie_ids_with_names(utt, items_db)
            # movie_ids = extract_movie_ids(utt['text'])
            knowledge = extract_knowledge(utt['entity'])

            # 1. 响应生成任务
            target_id = tokenizer.encode(system_text)
            know_len = int(args.max_seq_length/2)
            new_source_id = source_id
            # if knowledge:
            # movies_text = [items_db[mid]['movieName'] for mid in movie_ids]
            # movies_text = []
            # for movie in utt['movies']:
            #     if movie in name2review:
            #         movies_text.append(movie + ' '.join(name2review[movie]))
            #     else:
            #         movies_text.append(movie)
            # movies_text = [movie + ' '.join(name2review[movie]) for movie in utt['movies']]
            new_source_id += tokenizer.encode('[knowledge]')[1:-1] + tokenizer.encode('|'.join(knowledge))[1:][-know_len:] + tokenizer.encode('[item]' + '|'.join(movies_text))[1:] + tokenizer.encode('Generate the response:')[1:]
            # if utt['role'] 
            # new_source_id += tokenizer.encode('[system]')[1:]
            
            source_ids.append([101] + new_source_id[-args.max_seq_length+1:])
            target_ids.append([101] + target_id[-args.max_target_length+1:])
            item_ids.append([len(item_dict)-1])
            data_dict['resp']['source_ids'].append(source_ids[-1])
            data_dict['resp']['target_ids'].append(target_ids[-1])
            data_dict['resp']['item_ids'].append(item_ids[-1])

            # 2. 知识预测任务（只在有knowledge时添加）
            # if knowledge:
            target_id = tokenizer.encode('|'.join(knowledge))
            new_source_id = source_know_id + tokenizer.encode('Predict the next topic:')[1:]
            source_know_id += (tokenizer.encode('[knowledge]' + '|'.join(knowledge))[1:] + 
                            tokenizer.encode('[{}]'.format(utt['role']) + system_text)[1:])
            
            source_ids.append([101] + new_source_id[-args.max_seq_length+1:])
            target_ids.append([101] + target_id[-args.max_target_length+1:])
            item_ids.append([len(item_dict)-1])
            data_dict['know']['source_ids'].append(source_ids[-1])
            data_dict['know']['target_ids'].append(target_ids[-1])
            data_dict['know']['item_ids'].append(item_ids[-1])

            # 3. 物品推荐任务（只在有推荐时添加）
            # if movie_ids != [0]:  # 只有在实际有电影推荐时才添加
            if utt['movies']:
                # movies_text = [items_db[mid]['movieName'] for mid in movie_ids]
                item_new_ids = [item_dict[movie_id] for movie_id in utt['movies']]
                # item_new_ids = [items_db[movie_id]['new_item_id'] for movie_id in movie_ids]
                for i, item_new_id in enumerate(item_new_ids):
                    target_text = []
                    # for item, item_id in zip(movies_text, movie_ids):
                    target_text = ['<'+str(item_new_id)+'>'+ movies_text[i]]
                    target_id = tokenizer.encode('|'.join(target_text))
                    new_source_id = source_id
                    if knowledge:
                        new_source_id += tokenizer.encode('[knowledge]' + '|'.join(knowledge))[1:]
                    new_source_id += tokenizer.encode('Recommend an item:')[1:]
                    
                    source_ids.append([101] + new_source_id[-args.max_seq_length+1:])
                    target_ids.append([101] + target_id[-args.max_target_length+1:])
                    item_ids.append([item_new_id])
                    data_dict['item']['source_ids'].append(source_ids[-1])
                    data_dict['item']['target_ids'].append(target_ids[-1])
                    data_dict['item']['item_ids'].append(item_ids[-1])
                    rec_index.append(i)
            source_id += tokenizer.encode('[{}]'.format(utt['role']) + system_text)[1:]
            i += 1

        # print('{} set, max_res_len: {}, max_dia_len: {}, avg_res_len: {}, avg_dia_len: {}'.format(
        #     mode, max_res_len, max_dia_len, 
        #     float(sum(avg_res_len))/len(avg_res_len), 
        #     float(sum(avg_dia_len))/len(avg_dia_len)))

    if mode == 'train':
        data_dict['item_dict'] = item_dict
        return data_dict
    else:
        data_dict['rec_index'] = rec_index
        data_dict['item_dict'] = item_dict
        return data_dict

def merge_dataset(ft_dataset):
    source_ids = []
    target_ids = []
    item_ids = []
    item_dict = ft_dataset['item_dict']
    for task in ['resp','know','item']:
        task_dataset = ft_dataset[task]
        for source_id, target_id, item_id in zip(task_dataset['source_ids'], task_dataset['target_ids'], task_dataset['item_ids']):
            source_ids.append(source_id)
            target_ids.append(target_id)
            item_ids.append(item_id)
    return {'source_ids':source_ids, 'target_ids':target_ids, 'item_ids':item_ids, 'item_dict':item_dict}

def process_pipeline_data(args, tokenizer, data, all_preds, task):
    # decode 21131 to word
    print(f'tokenizer.decode(21131): {tokenizer.decode(21131)}')
    
    if task == 'resp':
        if args.data_name == 'durecdial':
            path = os.path.join(args.data_dir, 'kb_{}.jsonl'.format(args.data_name))
            kbs = []
            with open(path, 'r', encoding='utf-8') as infile:
                for line in infile:
                    kbs.append(eval(line.strip('\n')))
            assert len(kbs) == len(data['resp']['source_ids'])
        sid = 21131
        new_source_ids = []
        count = 0
        rec_index = data['rec_index']
        item_dict = data['item_dict']
        i = 0
        j = 0
        for source_id, know_pred in zip(data['resp']['source_ids'], all_preds['know']):
            assert source_id.count(sid) <= 1
            old_source_id = source_id.copy()
            uid = source_id[-6:]
            if sid in source_id:
                source_id = source_id[1:source_id.index(sid)]
            else:
                source_id = []
            # goal_pred = ''.join(goal_pred.split(' '))
            if args.data_name == 'durecdial':
                kb = kbs[j]
                know_text = []
                knows = ''.join(know_pred.split(' ')).split('|')
                for obj in knows:
                    if obj not in kb:
                        continue
                    tup = kb[obj]
                    if type(tup) is str:
                        know_text.append(obj+'：'+tup)
                    elif type(tup) is dict:
                        flag = True
                        for key in tup:
                            if key in knows:
                                know_text.append(obj+'，'+key+'，'+'、'.join(tup[key]))
                                flag = False
                        if flag:
                            for key in tup:
                                know_text.append(obj+'，'+key+'，'+'、'.join(tup[key]))
                if len(know_text) == 0 and knows != ['']:
                    for obj in kb:
                        tup = kb[obj]
                        if type(tup) is str:
                            continue
                        else:
                            for key in tup:
                                know_text.append(obj+'，'+key+'，'+'、'.join(tup[key]))
                know_pred = '|'.join(know_text)
            else:
                know_pred = ''.join(know_pred.split(' '))

            if j in rec_index:
                item_pred = item_dict[all_preds['item'][i][0]]
                i += 1
            else:
                item_pred = ''
            j += 1

            know_len = int(args.max_seq_length/2)
            source_id += (tokenizer.encode('[knowledge]')[1:-1] + tokenizer.encode(know_pred)[1:][-know_len:] + tokenizer.encode('[item]' + item_pred)[1:] + uid)
            new_source_ids.append([101] + source_id[-args.max_seq_length+1:])
            if old_source_id == new_source_ids[-1]:
                count += 1
            else:
                pass
                # print(know_pred)
                # print(tokenizer.decode(old_source_id, skip_special_tokens=True, clean_up_tokenization_spaces=True))
                #print(tokenizer.decode(new_source_ids[-1], skip_special_tokens=True, clean_up_tokenization_spaces=True))
        print(float(count)/len(new_source_ids))
        data['resp']['source_ids'] = new_source_ids
        return data['resp']
    elif task == 'know':
        sid = 21131
        new_source_ids = []
        count = 0
        for source_id in data['know']['source_ids']:
            print(f'source_id: {source_id}')
            # list decode to word
            print(f'tokenizer.decode(source_id): {tokenizer.decode(source_id)}')
            assert source_id.count(sid) == 1
            old_source_id = source_id.copy()
            source_id = source_id[1:source_id.index(sid)]
            source_id += tokenizer.encode('Predict the next topic:')[1:]
            #print(old_source_id, source_id[source_id.index(sid):])
            new_source_ids.append([101] + source_id[-args.max_seq_length+1:])
            if old_source_id == new_source_ids[-1]:
                count += 1
            else:
                pass
                #print(tokenizer.decode(old_source_id, skip_special_tokens=True, clean_up_tokenization_spaces=True))
                #print(tokenizer.decode(new_source_ids[-1], skip_special_tokens=True, clean_up_tokenization_spaces=True))
        print(float(count)/len(new_source_ids))
        data['know']['source_ids'] = new_source_ids
        return data['know']
    elif task == 'item':
        rec_index = data['rec_index']
        # filtered_preds = []
        filtered_knows = []
        # for i, pred in enumerate(all_preds['goal']):
        #     if i in rec_index:
        #         filtered_preds.append(pred)
        for i, pred in enumerate(all_preds['know']):
            if i in rec_index:
                filtered_knows.append(pred)
        assert len(filtered_knows) == len(data['item']['source_ids'])
        sid = 21131
        new_source_ids = []
        count = 0
        for source_id, pred_know in zip(data['item']['source_ids'], filtered_knows):
            assert source_id.count(sid) == 1
            old_source_id = source_id.copy()
            source_id = source_id[1:source_id.index(sid)]
            source_id += tokenizer.encode('[knowledge]' + ''.join(pred_know.split(' ')))[1:] + tokenizer.encode('Recommend an item:')[1:]
            new_source_ids.append([101] + source_id[-args.max_seq_length+1:])
            if old_source_id == new_source_ids[-1]:
                count += 1
            else:
                pass
                #print(tokenizer.decode(old_source_id, skip_special_tokens=True, clean_up_tokenization_spaces=True))
                #print(tokenizer.decode(new_source_ids[-1], skip_special_tokens=True, clean_up_tokenization_spaces=True))
        print(float(count)/len(new_source_ids))
        data['item']['source_ids'] = new_source_ids
        return data['item']


def load_movie_data(csv_path, args):
    # 首先获取所有在对话中被提到的电影ID
    mentioned_ids = get_mentioned_movie_ids(args.train_path)
    mentioned_ids.update(get_mentioned_movie_ids(args.valid_path))
    mentioned_ids.update(get_mentioned_movie_ids(args.test_path))

    # print(mentioned_ids)

    items_db = {}
    new_item_db = {}

    meta_path = '/projects/prjs1158/KG/redail/efficient_unified_crs_place/data/REDIAL/movie_db'

    meta_info = torch.load(meta_path)
    meta_dict = {}
    for key in meta_info.keys():
        meta = meta_info[key]
        # get the name before [SEP]
        title = meta.split('[SEP]')[0].strip()
        meta_dict[title] = meta.split('[SEP]')[1].strip()


    with open(csv_path, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        cnt = 0
        for row in reader:
            movie_id = int(row['movieId'])
            # 只有当电影ID在对话中被提到过时才保存
            if movie_id in mentioned_ids:
                # Remove content inside brackets and strip any extra whitespace
                import re
                new_movie_name = re.sub(r'\s*\(.*?\)\s*', '', row['movieName']).strip()
                if new_movie_name in meta_dict:         
                    items_db[movie_id] = {
                        "movieName": new_movie_name,
                        'meta': meta_dict[new_movie_name],
                        "nbMentions": int(row['nbMentions']),
                        "new_item_id": cnt
                    }
                    new_item_db[cnt] = {
                        "movieName": new_movie_name,
                        'meta': meta_dict[new_movie_name],}
                else:
                    items_db[movie_id] = {
                        "movieName": new_movie_name,
                        'meta': '',
                        "nbMentions": int(row['nbMentions']),
                        "new_item_id": cnt
                    }
                    new_item_db[cnt] = {
                        "movieName": new_movie_name,
                        'meta': '',}
                cnt += 1
    # print(items_db)
    return items_db, new_item_db

def get_mentioned_movie_ids(data):
    with open(data, 'r', encoding='utf-8') as f:
        data = json.load(f)
    """从对话数据中提取所有被提到的电影ID（格式为@[ID]）"""
    mentioned_ids = set()
    
    for conv in data:
        for turn in conv["dialog"]:
            text = " ".join(turn["text"])
            # 在文本中查找所有@开头的ID
            import re
            movie_mentions = re.findall(r'@(\d+)', text)
            mentioned_ids.update(movie_mentions)
    # change to int
    mentioned_ids = set([int(x) for x in mentioned_ids])
    return mentioned_ids

def old_load_movie_data(args):
    # 首先获取所有在对话中被提到的电影ID
    mentioned_ids = old_get_mentioned_movie_ids(args.train_path)
    mentioned_ids.update(old_get_mentioned_movie_ids(args.valid_path))
    mentioned_ids.update(old_get_mentioned_movie_ids(args.test_path))

    # print(mentioned_ids)

    items_db = {}
    reverse_items_db = {}
    for cnt, movie_id in enumerate(mentioned_ids):
        items_db[cnt] = {"movieName": movie_id}
        reverse_items_db[movie_id] = cnt
    reverse_items_db[len(reverse_items_db)] = '<PAD>'
    return items_db, reverse_items_db

def old_get_mentioned_movie_ids(data):
    with open(data, 'r', encoding='utf-8') as f:
        data = json.load(f)
    """从对话数据中提取所有被提到的电影ID（格式为@[ID]）"""
    mentioned_ids = set()
    
    for conv in data:
        for turn in conv["dialog"]:  # 注意这里是messages而不是dialog
            if len(turn["movies"]) > 0:
                # 处理每个电影ID，去掉两端空格
                movie_ids = [movie_id.strip() for movie_id in turn["movies"]]
                mentioned_ids.update(movie_ids)
    return mentioned_ids