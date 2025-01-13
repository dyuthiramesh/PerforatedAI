## Testing Custom Model with PAI

### 1. Convert Model to PAI Version and Copy Buffer Values
Run the below command to convert model to PAI version and copy buffer values from given file best_model_pai.pt 

```bash
python pai.py --alpha 100 --lr 0.0001 --batch-size 50 --rec-hidden 256 --gen-hidden 50 --latent-dim 20 --enc mtan_rnn --dec mtan_rnn --n 8000 --quantization 0.016 --save 1 --classif --norm --kl --learn-emb --k-iwae 1 --dataset physionet --multiplier 0.125 --justTest 1
```


2. Copy weights to saved model SecondModel.pt from weights stored in file best_model_pai.pt
```bash
python copy_weights.py --alpha 100 --lr 0.0001 --batch-size 50 --rec-hidden 256 --gen-hidden 50 --latent-dim 20 --enc mtan_rnn --dec mtan_rnn --n 8000 --quantization 0.016 --save 1 --classif --norm --kl --learn-emb --k-iwae 1 --dataset physionet --multiplier 0.125 --justTest 1
```


3. Run file test_model.py to get evalutation of model saved that is SecondModelCopied.pt
```bash
python test_model.py --alpha 100 --lr 0.0001 --batch-size 50 --rec-hidden 256 --gen-hidden 50 --latent-dim 20 --enc mtan_rnn --dec mtan_rnn --n 8000 --quantization 0.016 --save 1 --classif --norm --kl --learn-emb --k-iwae 1 --dataset physionet --multiplier 0.125 --justTest 1
```
## OR

Run file test_model.py which loads SecondModelCopied.pt which has weights and buffers copied from the original model and it has values printed of all parameters and buffers to cross check

```bash
python test_model.py --alpha 100 --lr 0.0001 --batch-size 50 --rec-hidden 256 --gen-hidden 50 --latent-dim 20 --enc mtan_rnn --dec mtan_rnn --n 8000 --quantization 0.016 --save 1 --classif --norm --kl --learn-emb --k-iwae 1 --dataset physionet --multiplier 0.125 --justTest 1
```

File results/results_mtan_base_train.txt has train log of 83.29 test auc
