# Plant Pathology-2021
任务旨在对plant pathology-2021数据集进行叶片病害分类
## 目录
- [模型架构](#模型架构)
- [数据集](#数据集)
- [环境要求](#环境要求)
- [脚本说明](#脚本说明)
- [测试说明](#测试说明)
## 模型架构
采用RESNET的改进网络SE-RESET,对模型的详细介绍在`作业文档.pdf`中查看
## 数据集
- 数据集大小：共4200张彩色图像及三个.csv文件
    - 训练集：3000张彩色图像及对应标签的csv文件
    - 验证集：600张彩色图像及对应标签的csv文件
    - 测试集：600张彩色图像及对应标签的csv文件
- 数据集格式：jpg格式的RGB图像及csv文件
## 环境要求
- 硬件
    - 使用GPU处理器来搭建硬件环境
- 深度学习框架
    - Mindspore 1.8
- 其他包
    - numpy
    - matplotlib
    - os
    - seaborn
    - pandas
## 脚本说明
```
    model.py                  #模型
    README.md               
    test.py                   #测试代码
    test_new_label.csv        #多标签分类结果，用于test.py
    test_visualization.jpg    #测试集测试后可视化结果
    train.py                  #训练代码
    trained_model_param.ckpt  #训练得到的模型
    作业文档.pdf           
```
## 测试说明
运行`test.py`测试训练得到的模型,`--path=测试集图片路径`
请保证数据集路径正确，可以根据实际路径修改`test.py`中的路径代码
```
 python test.py --path=plant_dataset/test/images
 ```
运行结果
```
--------TESTING...PLEASE WAIT A MINUTE...--------
TESTING ACCURACY: 89.3%
TESTING LOSS: 0.082288

--------TEST_VISUALIZATION.JPG HAS BEEN SAVED ALREADY--------
--------FISHING TESTING!--------
```
