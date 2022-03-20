# code for "Learning to Learn by jointly optimizing neural architecture and weights" CVPR2022

## 1. dataset prepare

download datasets from [mini-imagenet](https://drive.google.com/file/d/1qQCoGoEJKUCQkk8roncWH7rhPN7aMfBr/view), [omniglot](https://drive.google.com/file/d/1hYVghZT0U6VHFxcC4mQWtrygULhUXzaM/view?usp=sharing).
put the datasets under the `datasets` folder like:
```
datasets
    |
mini-imagenet - omniglot - FC100
```

## 2. init the envirnment

> conda git branch -M mainenv create >f torch.yaml

## 3. use the code

> cd scrips
> sh search_imagenet.sh

## 4. cite our work

```yaml
@inproceedings{ding2022learning,
      title={Learning to Learn by Jointly Optimizing Neural Architecture and Weights}, 
      author={Ding, Yadong and Wu, Yu and Huang, Chengyue and Tang, Siliang and Yang, Yi and Wei, Longhui and Zhuang, Yueting and Tian, Qi},
      booktitle={Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition},
      year={2022}
}
```

## 5. contact us

```yaml
Yadong Ding: yadong97@outlook.com
```
