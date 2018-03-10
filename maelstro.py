import logging
from datetime import datetime
from pathlib import Path
from timeit import default_timer as timer

import torch
from torch.utils.data import DataLoader
from torch.utils.data.sampler import SubsetRandomSampler

from ml_utils.pytorch.dataset import CustomDataset, train_valid_split
from ml_utils.pytorch.train import train, save_snapshot, load_snapshot
from ml_utils.pytorch.predict import predict, validate


logger = logging.getLogger(__name__)

root_dir = Path(__file__).parent
data_dir = root_dir / 'data'
train_dir = data_dir / 'train'
test_dir = data_dir / 'test'
snapshot_dir = root_dir / 'snapshots'
sub_dir = root_dir / 'submission'

class Maelstro():
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
        train_idx, valid_idx = train_valid_split(X_train, 0.1)

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

    def train(self, epochs, net, loss_func, optimizer):
        """Train the network."""
        best_score = 1.0
        for epoch in range(epochs):
            epoch_timer = timer()

            # Train and validate
            train(epoch, self.train_loader, net, loss_func, optimizer)

            score = validate(self.valid_loader, net, log_loss)

            if best_score > score:
                best_score = score
                save_snapshot(epoch+1, net, score, optimizer, snapshot_dir)

            end_epoch_timer = timer()
            logger.info("#### End epoch {}, elapsed time: {}".format(
                epoch, end_epoch_timer - epoch_timer))

        self.net = net

    def continue_training(self, epochs, net, loss_func, optimizer, pth_path):
        """Continue training a network."""
        epoch_start, net_state_dict, score, optimizer_state_dict = load_snapshot(pth_path)
        assert epochs >= epochs

        net.load_state_dict(net_state_dict)
        optimizer.load_state_dict(optimizer_state_dict)

        for epoch in range(epoch_start, epochs):
            epoch_timer = timer()

            # Train and validate
            train(epoch, self.train_loader, net, loss_func, optimizer)

            score = validate(self.valid_loader, net)

            if best_score < score:
                best_score = score
                save_snapshot(epoch+1, net, score, optimizer)

            end_epoch_timer = timer()
            logger.info("#### End epoch {}, elapsed time: {}".format(
                epoch, end_epoch_timer - epoch_timer))

    def predict(self, net, pth_path, batch_size=4, num_workers=4, limit_load=None):
        X_test = CustomDataset(data_dir / 'sample_submission.csv',
                               test_dir,
                               transform=ds_transform_raw,
                               limit_load=limit_load,
                               )

        test_loader = DataLoader(X_test,
                                 batch_size=batch_size,
                                 num_workers=num_workers,
                                 # pin_memory=True,
                                 )

        self.X_test, self.test_loader = X_test, test_loader

        # Load net from best iteration
        epoch, net_state_dict, score, _ = load_snapshot(pth_path)
        net.load_state_dict(net_state_dict)

        if torch.cuda.is_available():
            net.cuda()
            net = torch.nn.DataParallel(
                net, device_ids=range(torch.cuda.device_count()))

        # Predict
        proba_pred = predict(test_loader, net, score_type='proba')

        proba_pred_its_a_dog = proba_pred[:, 1]

        # Submission
        df_sub = X_test.df
        df_sub['label'] = proba_pred_its_a_dog

        sub_name = 'submission_{net_name}_{now}_score_{score}_epoch_{epoch}.csv'.format(
            net_name=str(net.__class__.__name__),
            now=datetime.now().strftime('%Y-%M-%d-%H-%m'),
            score=score,
            epoch=epoch,
        )
        sub_path = sub_dir / sub_name
        df_sub.to_csv(sub_path)
        logger.info('Submission file saved in {}'.format(sub_path))
