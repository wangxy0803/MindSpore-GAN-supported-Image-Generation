import mindspore.dataset as de
from mindspore import dtype as mstype
import mindspore.dataset.vision as C
import mindspore.dataset.transforms as C2
from mindspore.dataset.vision import ImageBatchFormat
from mindspore.dataset.vision import AutoAugmentPolicy, Inter
import pandas as pd
import os
import shutil

# 训练集路径
train_path = './plant_dataset/train'
# 验证集路径
val_path = './plant_dataset/val'
# 测试集路径
test_path = './plant_dataset/test'

#对数据集进行增强操作
def create_dataset(dataset, repeat_num=1, batch_size=32, target='train', image_size=224):
    mean = [0.485 * 255, 0.456 * 255, 0.406 * 255]
    std = [0.229 * 255, 0.224 * 255, 0.225 * 255]
    scale = 32
    type_op = C2.TypeCast(mstype.float32)
    if target == "train":
        # Define map operations for training dataset
        trans = [
            C.Resize(size=[image_size, image_size]),
            C.RandomHorizontalFlip(prob=0.5),
            C.RandomRotation(degrees=15),
            C.RandomColorAdjust(brightness=0.4, contrast=0.4, saturation=0.4),
            C.AutoAugment(policy=AutoAugmentPolicy.IMAGENET,interpolation=Inter.NEAREST,fill_value=0),
            C.Normalize(mean=mean, std=std),
            C.HWC2CHW(),
            type_op
        ]
    else:
        # Define map operations for inference dataset
        trans = [
            C.Resize(size=[image_size + scale, image_size + scale]),
            C.CenterCrop(image_size),
            C.Normalize(mean=mean, std=std),
            C.HWC2CHW(),
            type_op
        ]
    cutmix_batch_op = C.CutMixBatch(ImageBatchFormat.NCHW, 1.0, 0.5)

    dataset = de.GeneratorDataset(dataset, ["image", "label"])
    dataset = dataset.map(operations=trans, input_columns="image", num_parallel_workers=8)
    dataset = dataset.batch(batch_size, drop_remainder=True)
    if target == "train":
        dataset = dataset.map(operations=cutmix_batch_op, input_columns=["image", "label"], num_parallel_workers=8)
    dataset = dataset.map(operations=type_op, input_columns="label", num_parallel_workers=8)
    dataset = dataset.repeat(repeat_num)
    return dataset

if __name__ == "__main__":
    train_dataset = create_dataset(train_path)
    val_dataset = create_dataset(val_path)
    test_dataset = create_dataset(test_path)
