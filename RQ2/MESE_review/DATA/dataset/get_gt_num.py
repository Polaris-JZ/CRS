import json

# Check Inspired dataset
data_path = '/data/zhaoj9/KG_repro/CRSLab/data/dataset/inspired/nltk/test_data.json'

with open(data_path, 'r') as f:
    data = json.load(f)

found = False
for item in data:
    conv = item['dialog']
    for turn in conv:
        if len(turn['movies']) > 1:
            print("inspired dataset exist data with multi gt")
            found = True
            break
    if found:
        break

# Check Redial dataset
data_path = '/data/zhaoj9/KG_repro/CRSLab/data/dataset/redial/nltk/test_data.json'

with open(data_path, 'r') as f:
    data = json.load(f)

found = False
for item in data:
    conv = item['dialog']
    for turn in conv:
        if len(turn['movies']) > 1:
            print("redial dataset exist data with multi gt")
            found = True
            break
    if found:
        break

# Check TG-Redial dataset
data_path = '/data/zhaoj9/KG_repro/CRSLab/data/dataset/tgredial/pkuseg/test_data.json'

with open(data_path, 'r') as f:
    data = json.load(f)

found = False
for item in data:
    conv = item['messages']  # 注意这里是 'messages' 而不是 'dialog'
    for turn in conv:
        if len(turn['movie']) > 1:  # 注意这里是 'movie' 而不是 'movies'
            print("tg-redial dataset exist data with multi gt")
            found = True
            break
    if found:
        break
