## Testing Custom Model with PAI

### 1. Convert Model to PAI Version and Copy Buffer Values
Run the below command to convert model to PAI version and copy buffer values from given file best_model_pai.pt 
```bash
CUDA_VISIBLE_DEVICES=1 python3 pai.py --alpha 100 --lr 0.0001 --batch-size 50 --rec-hidden 256 --gen-hidden 50 --latent-dim 20 --enc mtan_rnn --dec mtan_rnn --n 8000 --quantization 0.016 --save 1 --classif --norm --kl --learn-emb --k-iwae 1 --dataset physionet --multiplier 0.125 --justTest 1
```

```bash
2. Copy weights to saved model SecondModel.pt from weights stored in file best_model_pai.pt

CUDA_VISIBLE_DEVICES=1 python3 copy_weights.py --alpha 100 --lr 0.0001 --batch-size 50 --rec-hidden 256 --gen-hidden 50 --latent-dim 20 --enc mtan_rnn --dec mtan_rnn --n 8000 --quantization 0.016 --save 1 --classif --norm --kl --learn-emb --k-iwae 1 --dataset physionet --multiplier 0.125 --justTest 1
```

```bash
3. Run file test_model.py to get evalutation of model saved that is SecondModelCopied.pt

CUDA_VISIBLE_DEVICES=1 python3 test_model.py --alpha 100 --lr 0.0001 --batch-size 50 --rec-hidden 256 --gen-hidden 50 --latent-dim 20 --enc mtan_rnn --dec mtan_rnn --n 8000 --quantization 0.016 --save 1 --classif --norm --kl --learn-emb --k-iwae 1 --dataset physionet --multiplier 0.125 --justTest 1
```
## OR

Run file test_model.py which loads SecondModelCopied.pt which has weights and buffers copied from the original model and it has values printed of all parameters and buffers to cross check

```bash
CUDA_VISIBLE_DEVICES=1 python3 test_model.py --alpha 100 --lr 0.0001 --batch-size 50 --rec-hidden 256 --gen-hidden 50 --latent-dim 20 --enc mtan_rnn --dec mtan_rnn --n 8000 --quantization 0.016 --save 1 --classif --norm --kl --learn-emb --k-iwae 1 --dataset physionet --multiplier 0.125 --justTest 1
```

File results/results_mtan_base_train.txt has train log of 83.29 test auc
