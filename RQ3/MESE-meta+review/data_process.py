import json
import csv
import torch

def load_movie_data():
    # 首先获取所有在对话中被提到的电影ID
    train_path = '/data/zhaoj9/KG_repro/PLM_based/redail/MESE/DATA/nltk/train_data.json'
    valid_path = '/data/zhaoj9/KG_repro/PLM_based/redail/MESE/DATA/nltk/valid_data.json'
    test_path = '/data/zhaoj9/KG_repro/PLM_based/redail/MESE/DATA/nltk/test_data.json'
    csv_path = '/data/zhaoj9/KG_repro/PLM_based/redail/MESE/DATA/nltk/movies_with_mentions.csv'
    mentioned_ids = get_mentioned_movie_ids(train_path)
    mentioned_ids.update(get_mentioned_movie_ids(valid_path))
    mentioned_ids.update(get_mentioned_movie_ids(test_path))

    # print(mentioned_ids)

    items_db = {}
    new_item_db = {}
    with open(csv_path, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        cnt = 0
        for row in reader:
            movie_id = int(row['movieId'])
            # 只有当电影ID在对话中被提到过时才保存
            if movie_id in mentioned_ids:
                items_db[movie_id] = {"movieName": row['movieName'], "new_item_id": cnt}
                new_item_db[cnt] = row['movieName']
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


def convert_json_to_redial_format(json_file, items_db):
    converted_data = []
    with open(json_file, 'r', encoding='utf-8') as f:
        json_data = json.load(f)
    
    for dialogue in json_data:
        conv_id = int(dialogue["conv_id"])
        dialog_pairs = []
        
        # 处理对话内容
        for i in range(len(dialogue["dialog"])):
            utterance = dialogue["dialog"][i]
            
            # 获取角色和文本
            role = "B: " if utterance["role"] == "Seeker" else "A: "
            
            # 处理文本，将@id替换为[MOVIE_ID]
            text = []
            movie_id = []
            for token in utterance["text"]:
                if token.startswith("@"):
                    # if not just @
                    if len(token) > 1:
                        text.append("[MOVIE_ID]")
                        # get movie new id
                        movie_ori_id = token.replace("@", "")
                        try:
                            movie_id.append(items_db[int(movie_ori_id)]["new_item_id"])
                        except:
                            print(f"movie {movie_ori_id} not found")
                else:
                    text.append(token)
            text = " ".join(text)

            if movie_id != []:
                dialog_pairs.append((role + text, movie_id))
            else:
                dialog_pairs.append((role + text, None))
        
        converted_data.append((conv_id, dialog_pairs))
    
    return converted_data


items_db, new_item_db = load_movie_data()
train_path = '/data/zhaoj9/KG_repro/PLM_based/redail/MESE/DATA/nltk/train_data.json'
valid_path = '/data/zhaoj9/KG_repro/PLM_based/redail/MESE/DATA/nltk/valid_data.json'
test_path = '/data/zhaoj9/KG_repro/PLM_based/redail/MESE/DATA/nltk/test_data.json'
train_converted_data = convert_json_to_redial_format(train_path, items_db)
valid_converted_data = convert_json_to_redial_format(valid_path, items_db)
test_converted_data = convert_json_to_redial_format(test_path, items_db)

# save to path
train_path = "/data/zhaoj9/KG_repro/PLM_based/redail/MESE/DATA/train_data_processed"
valid_path = "/data/zhaoj9/KG_repro/PLM_based/redail/MESE/DATA/valid_data_processed"
test_path = "/data/zhaoj9/KG_repro/PLM_based/redail/MESE/DATA/test_data_processed"
items_db_path = "/data/zhaoj9/KG_repro/PLM_based/redail/MESE/DATA/movie_db"

torch.save(train_converted_data, train_path)
torch.save(valid_converted_data, valid_path)
torch.save(test_converted_data, test_path)
torch.save(new_item_db, items_db_path)