import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler

from torchvision import models

from dataloading import *
from training import Trainer



def DeepLearning_classifier():

    resnet = models.resnet18(pretrained=True)
    num_ftrs = resnet.fc.in_features
    resnet.fc = nn.Linear(num_ftrs, 2)
    criterion = nn.CrossEntropyLoss()
    # Observe that all parameters are being optimized
    optimizer_ft = optim.Adam(resnet.parameters(), lr=0.001)
    # Decay LR by a factor of 0.1 every 7 epochs
    exp_lr_scheduler = lr_scheduler.StepLR(optimizer_ft, step_size=7, gamma=0.1)

    ML_trainer = Trainer(model=resnet, criterion=criterion, optimizer=optimizer_ft,scheduler=exp_lr_scheduler, dataloader=dataloaders, dataset_sizes=dataset_sizes,training_epochs=50)
    ML_trainer.full_cycle_training()



DeepLearning_classifier()