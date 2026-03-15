## Minimal NumPy Word2Vec Experiment

This is a minimal pure NumPy implementation of Word2Vec on the Shakespeare corpus
[`shakespeare.txt`](https://storage.googleapis.com/download.tensorflow.org/data/shakespeare.txt). It is downloaded and prepared automatically by `run_experiments.py` script. Use `--help` to get to know all CLI parameters. You can also configure it using JSON config like in `config.json`.

It supports:
- **CBOW** and **Skip-gram** training (plus negative sampling classes)
- Simple **SGD** with manual backprop
- Basic **hyperparameter search** over a small grid
- Absolute **early stopping** with patience on validation set

### Usage

```bash
python run_experiments.py
```

### Results
TODO

