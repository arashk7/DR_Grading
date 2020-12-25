import os
import pandas as pd
import torch
import torch.nn.functional as F
import pytorch_lightning as pl
from torch.utils import data
import torchvision
import torchvision.transforms as transforms
from efficientnet_pytorch import EfficientNet
from PIL import Image

from torch.utils.data.sampler import SubsetRandomSampler
from sklearn.metrics import cohen_kappa_score
from pytorch_lightning.callbacks import EarlyStopping

BASE_TRAIN_PATH = 'E:\Dataset\DR\DeepDr/Sample_DS'
''' Training Dataset Directory '''


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

batch_size = 2
''' batch size '''
params = {'batch_size': batch_size,
          'shuffle': True
          }
learning_rate = 1e-4
''' The learning rate '''

transform_train = transforms.Compose([transforms.Resize((100, 100)), transforms.RandomApply([
    torchvision.transforms.RandomRotation(30),
    transforms.RandomHorizontalFlip()], 0.7),
                                      transforms.ToTensor()])

''' Transform Images to specific size and randomly rotate and flip them '''

training_set = Dataset(os.path.join(BASE_TRAIN_PATH, 'sample_ds.csv'),
                       BASE_TRAIN_PATH,
                       transform=transform_train)

train_generator = data.DataLoader(training_set, **params)


class DRModel(pl.LightningModule):

    def __init__(self):
        super().__init__()
        self.model = EfficientNet.from_pretrained('efficientnet-b0', num_classes=5)
        self.preds = []
        self.labels = []

    def quadratic_kappa_cpu(self, y_hat, y):
        return cohen_kappa_score(y_hat, y, weights='quadratic')

    def forward(self, x):
        x = self.model(x)
        return x

    def training_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)

        eye = torch.eye(5).cuda()
        y = eye[y]
        metric = pl.metrics.MSE()
        loss = metric(y_hat, y)
        result = pl.TrainResult(loss)

        return result

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=learning_rate)

    def validation_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        eye = torch.eye(5).cuda()
        yy = eye[y]
        pred = torch.reshape(y_hat.argmax(dim=1, keepdim=True), (batch_size, 1))
        self.preds += pred.tolist()
        self.labels += y.tolist()

        loss = F.smooth_l1_loss(y_hat, yy)
        return {"val_loss": loss}

    def validation_epoch_end(self, outputs):

        loss = sum(x['val_loss'] for x in outputs) / len(outputs)

        qkappa = torch.tensor(self.quadratic_kappa_cpu(self.preds, self.labels))

        print({'val_epoch_qkappa': qkappa.item()})
        self.preds = []
        self.labels = []
        return {'log': {'val_loss': loss, 'val_epoch_qkappa': qkappa}}

model = DRModel()

test_size = int(0.2 * len(training_set))
indices = list(range(len(training_set)))
train_indices, val_indices = indices[test_size:], indices[:test_size]
train_sampler = SubsetRandomSampler(train_indices)
valid_sampler = SubsetRandomSampler(val_indices)

train_loader = torch.utils.data.DataLoader(training_set,
                                           sampler=train_sampler,
                                           batch_size=batch_size)
val_loader = torch.utils.data.DataLoader(training_set,
                                         sampler=valid_sampler,
                                         batch_size=batch_size)

early_stopping = EarlyStopping(
    monitor='val_loss',
    min_delta=0.00,
    patience=3,
    verbose=False,
    mode='min'
)

''' Early Stopping '''

trainer = pl.Trainer(gpus=1,early_stop_callback=early_stopping)
trainer.fit(model, train_loader, val_loader)


