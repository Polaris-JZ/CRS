import torch

path = '/data/zhaoj9/KG_repro/PLM_based/redail/efficient_unified_crs/data/REDIAL/test_data_processed'
data = torch.load(path)
print(data)