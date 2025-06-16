# Towards Unified Conversational Recommender Systems via Knowledge-Enhanced Prompt Learning
The source code for the KDD 2022 Paper [**"Towards Unified Conversational Recommender Systems via Knowledge-Enhanced Prompt Learning"**](https://arxiv.org/abs/2206.09363)


Compared with the original source code, we inject KG and review information into the model.


## Running
### Prompt Pre-training
```bash
cd src
sbatch train_pre.sh
```

### Conversation Task Training and Inference
```bash
cd src
sbatch train_conv.sh
```


### infer Conversation Task
```bash
cd src
sbatch infer_conv.sh
```


### Recommendation Task
```bash
cd src
sbatch train_rec.sh
```
