# Next Steps

## Current solutions

### To test

#### Modelisation

- Add hyperparameters search
- Add early stopping and checkpoints for training
- Calibrate gamma parameter of focal loss to reduce the weight of incorrect predictions 
#### Input data

- Add price for CNN approach
- Add product description
- Replace TF-IDF of LGBM by a pre-trained public transformer like model (Bert, RoBerta..)

### Limitation of current solutions

Should be reviewed for a multilingual purpose.

## Transformers

Next things to test:

- FineTune a transformer like model for classification (Bert, RoBerta..)
- Train a transformer like model language from scratch (or from pre-trained models) with product textual data and then fine tune it for classification