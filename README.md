# ChEBai

Run predefined training setup using 


```
python -m chebai train [BATCH_SIZE]
```

Add your own settings by altering run.py

## How to run electra

The interaction between pre-training and fine-tuning are not implementet, yet. Therefore, this part requires some manual steps:

### Pretraining

1. Create a folder `data/SWJpre/raw`
2. Create a file `smiles.txt` that contains the unlabeled pretraining data.
3. Run the pre-training:`python -m chebai train ElectraPre+SWJ [BATCH_SIZE]`
4. A successful run should create a new log in `logs` as well as *checkpoints* that can be used for fine-tuning

### Fine-tuning

1. Create a folder `data/JCI/raw`
2. Add raw `JCI`-data to this folder
3. Run the fine-tuning with the checkpoint from point 4 of the pre-training:
   ```python -m chebai train Electra+JCI [BATCH_SIZE] [PATH_TO_THE_CHECKPOINT]```

