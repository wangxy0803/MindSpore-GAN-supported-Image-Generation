
"""" TRAINING """
#导入相关库
from mindspore.train import Model
from mindspore import context
context.set_context(mode=context.GRAPH_MODE, device_target="GPU")
from mindvision.engine.callback import ValAccMonitor
import mindspore as ms
from mindspore import ops
import mindspore.nn as nn
from model import se_resnext50_32x4d
from dataset_transforms import create_dataset
import pandas as pd
from sklearn.model_selection import KFold
from PIL import Image

net_loss = nn.MultiClassDiceLoss(weights=None, ignore_indiex=None, activation="softmax")

#交叉验证
class KF_Dataset():

    def __init__(self, csv,spilt='train'):
        super(KF_Dataset, self).__init__()
        self.train = csv
        self.spilt = spilt
        self.imgs = self.train['images'].values
        self.labels = self.train.drop(['images'], axis=1).values


    def __getitem__(self, index):
        if self.spilt == 'train':
            img = Image.open('./plant_dataset/train/images/'+self.imgs[index]).convert('RGB')
        else:
            img = Image.open('./plant_dataset/test/images/'+self.imgs[index]).convert('RGB')
        return img, self.labels[index]

    def __len__(self):
        return len(self.imgs)

kf = KFold(n_splits=5, shuffle=True, random_state=2022)
train_path = './plant_dataset/train'
train = pd.read_csv(train_path + '/train_label.csv')
train_index = []
val_index = []
for train_index_, val_index_ in kf.split(train):
    train_index.append(train_index_)
    val_index.append(val_index_)
train_set = []
val_set = []
for i in range(5):
    train_set.append(KF_Dataset(train.iloc[train_index[i]],spilt='train'))
    val_set.append(KF_Dataset(train.iloc[val_index[i]],spilt='train'))

def main():

 def train_main(model, dataset, loss_fn, optimizer):
# Define forward function
    def forward_fn(data, label):
        logits = model(data)
        loss = loss_fn(logits, label)
        return loss, logits

    # Get gradient function
    grad_fn = ops.value_and_grad(forward_fn, None, optimizer.parameters, has_aux=True)

    # Define function of one-step training
    def train_step(data, label):
        (loss, _), grads = grad_fn(data, label)
        loss = ops.depend(loss, optimizer(grads))
        return loss

    size = dataset.get_dataset_size()
    model.set_train()
    for batch, (data, label) in enumerate(dataset.create_tuple_iterator()):
        loss = train_step(data, label)

        if batch % 100 == 0:
            loss, current = loss.asnumpy(), batch
            print(f"loss: {loss:>7f}\n")


    se_resnext = se_resnext50_32x4d(num_classes=5)
    dataset_train = create_dataset(train_set[0], batch_size=128, target='train', image_size=224)
    dataset_val = create_dataset(val_set[0], batch_size=128, target='val', image_size=224)
    net_opt = nn.Adam(se_resnext.trainable_params(), learning_rate=0.001, beta1=0.9, beta2=0.999, eps=1e-08, loss_scale=1.0)
    epochs = 50
    for t in range(epochs):
        print(f"Epoch {t+1}\n")
        train_main(se_resnext, dataset_train, net_loss, net_opt)
    print("--------Successfully Trained!--------")

    se_resnext = se_resnext50_32x4d(num_classes=5)
    dataset_train = create_dataset(train_set[1], batch_size=128, target='train', image_size=224)
    dataset_val = create_dataset(val_set[1], batch_size=128, target='val', image_size=224)
    net_opt = nn.Adam(se_resnext.trainable_params(), learning_rate=0.001, beta1=0.9, beta2=0.999, eps=1e-08, loss_scale=1.0)
    epochs = 50
    for t in range(epochs):
        print(f"Epoch {t+1}\n")
        train_main(se_resnext, dataset_train, net_loss, net_opt)
    print("--------Successfully Trained!--------")

    dataset_train = create_dataset(train_set[2], batch_size=128, target='train', image_size=224)
    dataset_val = create_dataset(val_set[2], batch_size=128, target='val', image_size=224)
    net_opt = nn.Adam(se_resnext.trainable_params(), learning_rate=0.001, beta1=0.9, beta2=0.999, eps=1e-08, loss_scale=1.0)
    epochs = 50
    for t in range(epochs):
        print(f"Epoch {t+1}\n")
        train_main(se_resnext, dataset_train, net_loss, net_opt)
    print("--------Successfully Trained!--------!")
    
    se_resnext = se_resnext50_32x4d(num_classes=5)
    dataset_train = create_dataset(train_set[3], batch_size=128, target='train', image_size=224)
    dataset_val = create_dataset(val_set[3], batch_size=128, target='val', image_size=224)
    net_opt = nn.Adam(se_resnext.trainable_params(), learning_rate=0.001, beta1=0.9, beta2=0.999, eps=1e-08, loss_scale=1.0)
    epochs = 50
    for t in range(epochs):
        print(f"Epoch {t+1}\n")
        train_main(se_resnext, dataset_train, net_loss, net_opt)
        print("--------Successfully Trained!--------!")
    
    se_resnext = se_resnext50_32x4d(num_classes=5)
    dataset_train = create_dataset(train_set[4], batch_size=128, target='train', image_size=224)
    dataset_val = create_dataset(val_set[4], batch_size=128, target='val', image_size=224)
    net_opt = nn.Adam(se_resnext.trainable_params(), learning_rate=0.001, beta1=0.9, beta2=0.999, eps=1e-08, loss_scale=1.0)
    epochs = 50
    for t in range(epochs):
        print(f"Epoch {t+1}\n")
        train_main(se_resnext, dataset_train, net_loss, net_opt)
    print("--------Successfully Trained!--------!")
    
    se_resnext = se_resnext50_32x4d(num_classes=5)
    param_dict = ms.load_checkpoint("trained_model_param.ckpt")
    param_not_load = ms.load_param_into_net(se_resnext, param_dict)
    if param_not_load==[]:
        print("------Successfully Saved Checkpoints!------")
        
        
if __name__ == "__main__":
    main()