# Discrete VAE
### Alexandros Graikos

The VQ-VAE experiments code is a modified version of https://github.com/bshall/VectorQuantizedVAE. To train the VQ-VAE model use

```python train.py --num-embeddings {K} --num-epochs {E} --reduce_lr_on {E'} --device {cpu/cuda}```

The results will be stored in the `runs` directory. 
The argument `--run {R}` can be used to differentiate between different runs with the same hyperparameters.


The GFlowNet experiments code uses the encoder and decoder networks from the VQ-VAE implementation. To train the base GFlowNet model run

```python train_gfn.py --dictionary_size {K} --greedy_prob {GP} --epochs {E} --reduce_lr_on {E'} --importance_samples {M} --device {cpu/cuda}```

The jointly learned prior model can be trained using

```python train_gfn_prior.py --dictionary_size {K} --greedy_prob {GP} --epochs {E} --reduce_lr_on {E'} --importance_samples {M} --device {cpu/cuda}```

The results will be stored in the `models` directory.
Again the `--run {R}` argument can be used to accumulate results over multiple runs.

## Replicating the Experiments

To replicate the VQ-VAE experiments run in the `vqvae` directory
```
python train.py --num-embeddings 4 --num-epochs 50 --reduce_lr_on 25 --device cuda
python train.py --num-embeddings 8 --num-epochs 50 --reduce_lr_on 25 --device cuda
python train.py --num-embeddings 10 --num-epochs 80 --reduce_lr_on 50 --device cuda
```

To replicate the GFlowNet encoder experiments run in the `gfn` directory
- GFlowNet-EM
```
python train_gfn.py --dictionary_size 4 --greedy_prob 0 --epochs 500 --reduce_lr_on 400 --importance_samples 5000 --device cuda
python train_gfn.py --dictionary_size 8 --greedy_prob 0 --epochs 500 --reduce_lr_on 400 --importance_samples 5000 --device cuda
python train_gfn.py --dictionary_size 10 --greedy_prob 0 --epochs 500 --reduce_lr_on 400 --importance_samples 5000 --device cuda
```

- GFlowNet-EM + Greedy Decoder Training
```
python train_gfn.py --dictionary_size 4 --greedy_prob 1 --epochs 250 --reduce_lr_on 180 --importance_samples 5000 --device cuda
python train_gfn.py --dictionary_size 8 --greedy_prob 1 --epochs 250 --reduce_lr_on 180 --importance_samples 5000 --device cuda
python train_gfn.py --dictionary_size 10 --greedy_prob 1 --epochs 250 --reduce_lr_on 180 --importance_samples 5000 --device cuda
```

- GFlowNet-EM + Greedy Decoder Training + Jointly Learned Prior
```
python train_gfn_prior.py --dictionary_size 4 --greedy_prob 1 --epochs 400 --reduce_lr_on 300 --importance_samples 5000 --device cuda
python train_gfn_prior.py --dictionary_size 8 --greedy_prob 1 --epochs 400 --reduce_lr_on 300 --importance_samples 5000 --device cuda
python train_gfn_prior.py --dictionary_size 10 --greedy_prob 1 --epochs 400 --reduce_lr_on 300 --importance_samples 5000 --device cuda
```













