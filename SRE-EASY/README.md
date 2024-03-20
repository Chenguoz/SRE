# SRE-EASY

## Description
This repository is the official implementation of **Exploring Sample Relationship for Few-Shot Classification**.
         
> Xingye Chen, Wenxiao Wu, Li Ma, Xinge You, Changxin Gao,Nong Sang, Yuanjie Shao      


## Environment setup
```
pip install tqdm scikit-learn
pip install torch==1.8.1+cu111 torchvision==0.9.1+cu111 torchaudio==0.8.1 -f https://download.pytorch.org/whl/torch_stable.html
```

## Dataset

We use the same data preprocessing method as [EASY](https://github.com/ybendou/easy).


## Training
Train a model on miniimagenet using manifold mixup, self-supervision and cosine scheduler. The best backbone is based on the 1-shot performance in the validation set. In order to get the best 5-shot performing model during validation, change --n-shots to 5.
```
python main.py --dataset-path "YOUR_DATAPATH" --dataset miniimagenet --skip-epochs 450   --model resnet12 --epochs 0 --sample-aug 20   --manifold-mixup 500 --rotations --cosine --gamma 0.9 --milestones 100 --batch-size 128 --preprocessing ME  --log-path  "YOUR_LOGPATH" --n-shots 5 --ls 0.75 --lu 0.25
```

## Acknowledgment

Our SRE-EASY implementation is mainly based on the following codebase. We gratefully thank the authors for their wonderful works.

[EASY](https://github.com/ybendou/easy)
