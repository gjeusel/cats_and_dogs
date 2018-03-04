from pathlib import Path
import logging
from timeit import default_timer as timer
import random

import numpy as np
import pandas as pd

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.utils.data.sampler import SubsetRandomSampler
from torchvision import transforms

from ml_utils.pytorch.dataset import CustomDataset, train_valid_split
from ml_utils.pytorch.train import train, save_snapshot, load_snapshot
from ml_utils.pytorch.predict import predict, validate
from networks import ShortNet


logging.basicConfig(filename='first_run.log',
                    filemode='w',
                    level=logging.INFO,
                    format='%(asctime)s %(levelname)s:%(message)s',
                    datefmt='%d/%m/%Y %H:%M:%S',
                    )

# define a Handler which writes INFO messages or higher to the sys.stderr
console = logging.StreamHandler()

# add the handler to the root logger
logging.getLogger('').addHandler(console)

data_dir = Path(__file__).parent/'data'
train_dir = data_dir / 'train'
test_dir = data_dir / 'test1'

class CatAndDogHandler():
    """Wrapper class."""

    def __init__(self,
                 ds_transform_augmented, ds_transform_raw,
                 batch_size=4,
                 num_workers=4,
                 sampler=SubsetRandomSampler,
                 limit_load=None,
                 submit=False):

        self.ds_transform_augmented = ds_transform_augmented
        self.ds_transform_raw = ds_transform_raw

        # Loading the dataset
        X_train = CustomDataset(data_dir / 'train.csv', train_dir,
                                transform=ds_transform_augmented,
                                limit_load=limit_load,
                                )
        X_val = CustomDataset(data_dir / 'train.csv', train_dir,
                              transform=ds_transform_raw,
                              limit_load=limit_load,
                              )

        # Creating a validation split
        train_idx, valid_idx = train_valid_split(X_train, 0.2)

        if submit:
            train_idx = range(len(X_train))

        if sampler is not None:
            train_sampler = SubsetRandomSampler(train_idx)
            valid_sampler = SubsetRandomSampler(valid_idx)
        else:
            train_sampler, valid_sampler = None, None

        # Both dataloader loads from the same dataset but with different indices
        train_loader = DataLoader(X_train,
                                  batch_size=batch_size,
                                  sampler=train_sampler,
                                  num_workers=num_workers,
                                  # pin_memory=True,
                                  )

        valid_loader = DataLoader(X_val,
                                  batch_size=batch_size,
                                  sampler=valid_sampler,
                                  num_workers=num_workers,
                                  # pin_memory=True,
                                  )

        self.X_train, self.X_val = X_train, X_val
        self.train_idx, self.valid_idx = train_idx, valid_idx
        self.train_loader, self.valid_loader = train_loader, valid_loader

    def train(self, epochs, model, loss_func, optimizer, and_validate=True):
        for epoch in range(epochs):
            epoch_timer = timer()

            # Train and validate
            train(epoch, self.train_loader, model, loss_func, optimizer)

            if and_validate:
                score, loss = validate(epoch, self.valid_loader, model, loss_func)

            save_snapshot(epoch+1, model, loss, optimizer)

            end_epoch_timer = timer()
            logging.info("#### End epoch {}, elapsed time: {}".format(
                epoch, end_epoch_timer - epoch_timer))

        self.model = model


    # def continue_training(self, epochs, model, loss_func, optimizer, fname,
    #                       and_validate=True):
    #     start_epoch, net_state_dict, loss, optimizer_state_dict = load_model(fname)
    #     model.load_state_dict(net_state_dict)
    #     optimizer = optimizer.load_state_dict(optimizer_state_dict)

    #     for epoch in range(epoch_start, epochs):
    #         epoch_timer = timer()

    #         # Train and validate
    #         train(epoch, self.train_loader, model, loss_func, optimizer)

    #         if and_validate:
    #             score, loss = validate(epoch, self.valid_loader, model, loss_func)

    #         save_snapshot(epoch+1, model, loss, optimizer)

    #         end_epoch_timer = timer()
    #         logging.info("#### End epoch {}, elapsed time: {}".format(
    #             epoch, end_epoch_timer - epoch_timer))


    # def predict(self, id_model, model, batch_size=4, num_workers=4, limit_load=None):
    #     X_test = RoofDataset(DATA_DIR / 'sample_submission.csv',
    #                          IMAGE_DIR,
    #                          transform=self.ds_transform_raw,
    #                          limit_load=limit_load,
    #                          )

    #     test_loader = DataLoader(X_test,
    #                              batch_size=batch_size,
    #                              num_workers=num_workers,
    #                              # pin_memory=True,
    #                              )

    #     self.X_test, self.test_loader = X_test, test_loader

    #     # Load model from best iteration
    #     logging.info('===> loading best model for prediction')
    #     model_state_dict, optimizer_state_dict, epoch_start = \
    #         self.dbmodel.get_existing_cnn(id_model)
    #     model.load_state_dict(model_state_dict)

    #     if torch.cuda.is_available():
    #         model.cuda()
    #         model = torch.nn.DataParallel(
    #             model, device_ids=range(torch.cuda.device_count()))

    #     # Predict
    #     predictions = predict(test_loader, model)

    #     write_submission_file(predictions,
    #                           self.X_test.ids,
    #                           SUBMISSION_DIR,
    #                           id_model,
    #                           self.dbmodel.df.loc[0]['best_score'],
    #                           )

if __name__ == '__main__':
    # Initiate timer:
    global_timer = timer()

    # Setting random seeds for reproducibility. (Caveat, some CuDNN algorithms are non-deterministic)
    torch.manual_seed(1337)
    # torch.cuda.manual_seed(1337)
    np.random.seed(1337)
    random.seed(1337)

    ##### Preprocessing parameters: #####

    # Normalization on ImageNet mean/std for finetuning
    # normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
    #                                  std=[0.229, 0.224, 0.225])
    normalize = transforms.Normalize(mean=[0.5, 0.5, 0.5],
                                     std=[0.5, 0.5, 0.5])

    image_size = 64

    # Augmentation + Normalization for full training
    ds_transform_augmented = transforms.Compose([
        transforms.RandomResizedCrop(image_size),
        # transforms.RandomHorizontalFlip(),
        # transforms.RandomVerticalFlip(),
        # transforms.ColorJitter(),  # Randomly change the brightness, contrast and saturation of an image.
        # transforms.RandomRotation(10),
        transforms.ToTensor(),
        normalize,
    ])

    # Normalization only for validation and test
    ds_transform_raw = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        normalize
    ])

    batch_size = 8
    num_workers = 4
    logging.info('Initializing model with: num_workers = {workers} | '
                 'batch_size = {batch}'.format(workers=num_workers,
                                             batch=batch_size))
    cad = CatAndDogHandler(ds_transform_augmented,
                           ds_transform_raw,
                           batch_size=batch_size,
                           num_workers=num_workers)

    ##### Model parameters: #####
    model = ShortNet((3, image_size, image_size))
    # model = ResNet(num_classes=4, resnet=18)
    # model = ResNet(num_classes=4, resnet=34)
    # model = ResNet(num_classes=4, resnet=50)
    # model = DPN26()

    if torch.cuda.is_available():
        model.cuda()
        model = torch.nn.DataParallel(
            model, device_ids=range(torch.cuda.device_count()))

    # criterion = ConvolutedLoss()
    # weight = torch.Tensor([1., 1.971741, 3.972452, 1.824547])
    # criterion = torch.nn.MultiLabelSoftMarginLoss(weight=weight)
    # criterion = SmoothF2Loss()
    # criterion = torch.nn.CrossEntropyLoss(weight=weight)
    criterion = torch.nn.CrossEntropyLoss()

    # Note, p_training has lr_decay automated
    optimizer = optim.SGD(model.parameters(), lr=1e-2, momentum=0.9,
                          weight_decay=0.0005)  # Finetuning whole model

    # Training:
    cad.train(epochs=45, model=model, loss_func=criterion, optimizer=optimizer)


    # Predict
    # id_model = dr.dbmodel.get_id_model(model, optimizer, criterion,
    #                                    ds_transform_augmented, ds_transform_raw)
    # dr.predict(id_model, model)

    end_global_timer = timer()
    logger.info("################## Success #########################")
    logger.info("Total elapsed time: %s" % (end_global_timer - global_timer))
