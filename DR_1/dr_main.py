import os
import pandas as pd
import torch
import torch.nn.functional as F
import pytorch_lightning as pl
from torch.utils import data
from torchvision.datasets import MNIST
import torchvision
import torchvision.transforms as transforms
from efficientnet_pytorch import EfficientNet
from PIL import Image
from sklearn.model_selection import KFold
from torch.utils.data.sampler import SubsetRandomSampler

BASE_TRAIN_PATH = 'E:\Dataset\DR\DeepDr'
''' Training Dataset Directory '''
BASE_TEST_PATH = 'E:\Dataset\DR\DeepDr\Onsite-Challenge1-2-Evaluation'
''' Test Dataset Directory '''


class Dataset(data.Dataset):
    def __init__(self, csv_path, images_path, transform=None):
        ''' Initialise paths and transforms '''
        self.train_set = pd.read_csv(csv_path, keep_default_na=False)  # Read The CSV and create the dataframe
        self.train_path = images_path  # Images Path
        self.transform = transform  # Augmentation Transforms

    def __len__(self):
        return len(self.train_set)

    def __getitem__(self, idx):
        '''
        Receive element index, load the image from the path and transform it
        :param idx:
        Element index
        :return:
        Transformed image and its grade label
        '''
        # file_name = self.train_set['image_path'][idx]
        img_id = self.train_set['image_id'][idx]
        patient_id = self.train_set['patient_id'][idx]
        file_path = os.path.join(str(patient_id), str(img_id) + '.jpg')
        label = self.train_set['patient_DR_Level'][idx]
        path = os.path.join(self.train_path, file_path)
        path = path.replace('\\', '/')
        img = Image.open(path)  # Loading Image

        if self.transform is not None:
            img = self.transform(img)
        return img, label


'''
 Hyper Parameters 
'''

batch_size = 4
''' batch size '''
params = {'batch_size': batch_size,
          'shuffle': True
          }
learning_rate = 1e-4
''' The learning rate '''

transform_train = transforms.Compose([transforms.Resize((200, 200)), transforms.RandomApply([
    torchvision.transforms.RandomRotation(30),
    transforms.RandomHorizontalFlip()], 0.7),
                                      transforms.ToTensor()])
''' Transform Images to specific size and randomly rotate and flip them '''

training_set = Dataset(os.path.join(BASE_TRAIN_PATH, 'merged_tr_vl', 'merged_tr_vl.csv'),
                       os.path.join(BASE_TRAIN_PATH, 'merged_tr_vl'),
                       transform=transform_train)

train_generator = data.DataLoader(training_set, **params)


class LitModel(pl.LightningModule):

    def __init__(self):
        super().__init__()
        self.model = EfficientNet.from_pretrained('efficientnet-b0', num_classes=5)

    def forward(self, x):
        x = self.model(x)
        # x = F.log_softmax(x, dim=1)
        return x

    def training_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)

        eye = torch.eye(5).cuda()
        y = eye[y]
        loss = F.smooth_l1_loss(y_hat, y)
        result = pl.TrainResult(loss)
        # print(loss)
        result.log('train_loss', loss)
        return result

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=learning_rate)

    def validation_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        eye = torch.eye(5).cuda()
        y = eye[y]
        loss = F.smooth_l1_loss(y_hat, y)
        result = pl.EvalResult(loss)
        result.log('val_loss', loss)
        return result


trainer = pl.Trainer(gpus=1, limit_test_batches=1.0)
model = LitModel()

kfold = KFold(5, True, 1)
fold_count = 0
torch.save(model.state_dict(), 'init.pt')
for fold, (train_index, val_index) in enumerate(kfold.split(training_set)):
    fold_count += 1
    print('>>>>>>>Fold ' + str(fold_count))
    model.load_state_dict(torch.load('init.pt'))
    train_sampler = SubsetRandomSampler(train_index)
    valid_sampler = SubsetRandomSampler(val_index)
    train_loader = torch.utils.data.DataLoader(training_set,
                                               sampler=train_sampler,
                                               batch_size=batch_size)
    val_loader = torch.utils.data.DataLoader(training_set,
                                             sampler=valid_sampler,
                                             batch_size=batch_size)

    trainer.fit(model, train_loader, val_loader)
