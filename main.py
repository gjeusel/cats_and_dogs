import logging
from timeit import default_timer as timer
import random

import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torchvision import transforms

from sklearn.metrics import log_loss

from networks import ShortNet
from maelstro import Maelstro


logging.basicConfig(filename='first_run.log',
                    filemode='w',
                    level=logging.INFO,
                    format='%(asctime)s %(levelname)s:%(message)s',
                    datefmt='%d/%m/%Y %H:%M:%S',
                    )

# define a Handler which writes INFO messages or higher to the sys.stderr
console = logging.StreamHandler()
# formatter = logging.Formatter('%(asctime)s %(levelname)s:%(message)s')
# console.setFormatter(formatter)

# add the handler to the root logger
logging.getLogger('').addHandler(console)


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
    logging.info('Initializing net with: num_workers = {workers} | '
                 'batch_size = {batch}'.format(workers=num_workers,
                                               batch=batch_size))
    cad = Maelstro(ds_transform_augmented,
                   ds_transform_raw,
                   batch_size=batch_size,
                   num_workers=num_workers,
                   # limit_load=100,
                   )

    ##### net parameters: #####
    net = ShortNet((3, image_size, image_size), n_classes=2)
    # net = ResNet(num_classes=4, resnet=18)
    # net = ResNet(num_classes=4, resnet=34)
    # net = ResNet(num_classes=4, resnet=50)
    # net = DPN26()

    if torch.cuda.is_available():
        net.cuda()
        net = torch.nn.DataParallel(
            net, device_ids=range(torch.cuda.device_count()))

    # criterion = ConvolutedLoss()
    # weight = torch.Tensor([1., 1.971741, 3.972452, 1.824547])
    # criterion = torch.nn.MultiLabelSoftMarginLoss(weight=weight)
    # criterion = SmoothF2Loss()
    # criterion = torch.nn.CrossEntropyLoss(weight=weight)
    criterion = torch.nn.CrossEntropyLoss()

    # Note, p_training has lr_decay automated
    optimizer = optim.SGD(net.parameters(), lr=1e-2, momentum=0.9,
                          weight_decay=0.0005)  # Finetuning whole net

    from IPython import embed; embed() # Enter Ipython
    # Training:
    # cad.train(epochs=45, net=net, loss_func=criterion, optimizer=optimizer)


    # Predict
    # id_net = dr.dbnet.get_id_net(net, optimizer, criterion,
    #                                    ds_transform_augmented, ds_transform_raw)
    # dr.predict(id_net, net)

    end_global_timer = timer()
    logging.info("################## Success #########################")
    logging.info("Total elapsed time: %s" % (end_global_timer - global_timer))
