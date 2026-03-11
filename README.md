## Minimal NumPy Word2Vec Experiment

This is a minimal pure NumPy implementation of Word2Vec on the Shakespeare corpus
[`shakespeare.txt`](https://storage.googleapis.com/download.tensorflow.org/data/shakespeare.txt).

It supports:
- **CBOW** and **Skip-gram** training
- Simple **SGD** with manual backprop
- Basic **hyperparameter search** over a small grid

### Usage

```bash
python run_experiments.py
```

You can adjust hyperparameters and grids in `run_experiments.py`.

