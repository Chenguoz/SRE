# SRE-ProtoNet

## Description
This repository is the official implementation of **Exploring Sample Relationship for Few-Shot Classification**.
         
> Xingye Chen, Wenxiao Wu, Li Ma, Xinge You, Changxin Gao,Nong Sang, Yuanjie Shao      


## Environment setup
```
pip install tqdm scikit-learn
pip install torch==1.8.1+cu111 torchvision==0.9.1+cu111 torchaudio==0.8.1 -f https://download.pytorch.org/whl/torch_stable.html
```

## Dataset

We use the same data preprocessing method as [DeepBDC](https://github.com/Fei-Long121/DeepBDC).


## Training
We followed the two-stage training method in [DeepBDC](https://github.com/Fei-Long121/DeepBDC), and loaded our pre-trained backbone from [SRE-EASY](../SRE-EASY/README.md). You can also download the pre-trained backbone from [here](https://drive.google.com/file/d/1K9YiHnYqKAJZJZN7oJSB_f4bNEuVJEHw/view?usp=sharing).
```
python meta_train.py --dataset mini_imagenet --model ResNet12Prior --method priornet  --n_shot 5 --repeat_num 15 --pretrain_path ./checkpoints/model.pt5  --ls 0.75 --lu 0.25   --extra_dir SRE --data_path your_data
```

## Acknowledgment

Our implementation is mainly based on the following codebases. We gratefully thank the authors for their wonderful works.

[CloserLookFewShot](https://github.com/wyharveychen/CloserLookFewShot),
[DeepBDC](https://github.com/Fei-Long121/DeepBDC)
