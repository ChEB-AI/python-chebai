# ChEBai

Run predefined setup using 


```
python chem/run.py [BATCH_SIZE]
```

Add your own settings by altering run.py

## How to run electra

The interaction between pre-training and fine-tuning are not implementet, yet. Therefore, this part requires some manual steps:

### Pretraining

1. Create a folder `data/SWJpre/raw`
2. Create a file `smiles.txt` that contains the unlabeled pretraining data.
3. Run the first config from `run.py`
4. A successful run should create a new log in `logs` as well as checkpoints that can be used for fine-tuning

### Fine-tuning

1. Create a folder `data/JCI/raw`
2. Add raw `JCI`-data to this folder
3. Add the path one checkpoint from pretraining to `run.py`
4. Run the second config from `run.py`
