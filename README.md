# 机器学习课程作业：PyTorch实现

主要包括：引入新的骨干网络和损失函数；两视角同步变换数据增强(STTV，STTVCF)；训练和测试过程中不同的数据增强使用方法。

STTV是针对跨视角图像地理定位任务的一种两视角同步变换数据增强策略。具体而言，STTV对于地面图像使用平移变换，而对于卫星图像使用旋转变换；且平移变换每次向右平移图像宽度的1/4，旋转变换每次顺时针旋转$90^{\circ}$。经过上述变换，训练数据集可以扩充到原来的4倍，同时两视角同步变换保持了视角间图像之间的空间布局对应关系，不会增加额外的训练困难。示例如下：

<div style="text-align:center;">
    <image src="images/dataaug.jpg" style="width:80%;" style="height:50%;"/>
    <p>
        <strong>
             两视角同步变换示例
        </strong>
    </p>
</div>

在训练时，本文提出两种使用数据增强（STTV和STTVCF）的方式：

1. 对于每对地面卫星图像对，等概率的使用数据增强，即增强后的训练数据集扩充为原来的4倍，且原始的图像对与增强得到的图像在训练时获得相同的比重（“权重”）。
2. 对于每对地面卫星图像对，非等概率的使用数据增强，即原始的图像对以0.5的概率出现，而增强后的图像对以0.5的概率随机出现一种情况，在这种情况下模型将更加关注于原始图像，因为原始的图像对每个周期都以0.5的概率出现，而增强后的图像对虽然也是0.5的概率，但是却每次随机产生不一样的增强后图像对（旋转角度与平移距离）。

我们最好的结果是通过将STTV和情况1结合得到的。

STTVCF：探索任意旋转角度

<div style="text-align:center;">
    <image src="images/rotate.jpg" style="width:80%;" style="height:50%;"/>
    <p>
        <strong>
             卫星图像旋转任意角度带来的问题。黑色方框表示图像的边界。显然除旋转角度为90°的倍数时，会出现超出边界和缺失值的问题
        </strong>
    </p>
</div>

# 环境要求

```python
- Python >= 3.8, numpy, matplotlib, pillow, ptflops, timm
- PyTorch >= 1.8.1, torchvision >= 0.11.1
```

# 数据集

请下载 [CVUSA](http://mvrl.cs.uky.edu/datasets/cvusa/). 你可能需要修改 "dataloader"下的数据集路径。

# 预训练模型

| Dataset | R@1   | R@5   | R@10  | R@1%  | Weighted                                                     | Log                                                          |
| ------- | ----- | ----- | ----- | ----- | ------------------------------------------------------------ | ------------------------------------------------------------ |
| CVUSA   | 96.26 | 99.02 | 99.52 | 99.81 | [model](https://drive.google.com/file/d/1MUoBCmhmz9Je2WmFDPujdmb-lCF73wM3/view?usp=sharing) | [log](https://drive.google.com/file/d/1zNHZCjDCVHcrnCi4R7NLSiLY69K8DJUV/view?usp=sharing) |

# 使用用法

## 训练

在CVUSA数据集上训练我们的方法：

```shell
python -u train.py --lr 0.0001 --batch-size 32 --dist-url 'tcp://localhost:10000' --multiprocessing-distributed --world-size 1 --rank 0  --epochs 100 --save_path ./result --op sam --wd 0.03 --mining --dataset cvusa --cos --dim 384 --asam --rho 2.5
```

## 测试

你应该以下列方式组织下载的预训练模型：

```
./result
	model_best.pth.tar
	checkpoint.pth.tar
```

为了评估我们的方法，在CVUSA上运行：

```shell
python -u train.py --dist-url 'tcp://localhost:8080' --multiprocessing-distributed --world-size 1 --rank 0  --save_path ./result --dataset cvusa --dim 384 -e
```

# 热力图

<div style="text-align:center;">
    <image src="images/heatmap.jpg" style="width:80%;" style="height:50%;"/>
    <p>
        <strong>
             我们提出的方法热力图可视化
        </strong>
    </p>
</div>

# 检索示例

<div style="text-align:center;">
    <image src="images/retrieval.jpg" style="width:80%;" style="height:50%;"/>
    <p>
        <strong>
             我们的方法在CVUSA数据集上的检索示例
        </strong>
    </p>
</div>

# 参考

[TransGeo](https://github.com/Jeff-Zilence/TransGeo2022)，[CVUSA](http://mvrl.cs.uky.edu/datasets/cvusa/)，[VIGOR](https://github.com/Jeff-Zilence/VIGOR)，[OriCNN](https://github.com/Liumouliu/OriCNN)，[Deit](https://github.com/facebookresearch/deit)，[MoCo](https://github.com/facebookresearch/moco)

有任何疑问，请通过[邮箱](zhangqingwang2022@email.szu.edu.cn)联系我，我将要在空闲时间及时解答您的问题。