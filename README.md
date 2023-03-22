# GBTI-Exp

## Requirements
- Python 3.8.13
- PyTorch >= 1.6.0
- dgl >= 0.5.3

## Usage
Data Preprocessing
```bash
cd data
python preprocess.py --dataset FB15kET
python preprocess.py --dataset YAGO43kET
```

Training
```bash
python run.py hydra.job.chdir=False hydra.verbose=True
```

Testing
```bash
python eval.py hydra.job.chdir=False　hydra.verbose=True
```

if you change the dataset e.g. from FB15kET to YAGO45kET, then you could change the config.
