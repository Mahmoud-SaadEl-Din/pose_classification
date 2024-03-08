import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler

from torchvision import models

from dataloading import *
from training import Trainer
from model import CustomFeedforwardNN



def DeepLearning_classifier():

    # Set random seed for reproducibility
    torch.manual_seed(42)

    NN = CustomFeedforwardNN(input_size=24,hidden_sizes=[16,8],output_size=1)
    criterion = nn.BCEWithLogitsLoss()
    # Observe that all parameters are being optimized
    optimizer_ft = optim.SGD(NN.parameters(), lr=0.001)
    # Decay LR by a factor of 0.1 every 7 epochs
    exp_lr_scheduler = lr_scheduler.StepLR(optimizer_ft, step_size=7, gamma=0.1)

    ML_trainer = Trainer(model=NN, criterion=criterion, optimizer=optimizer_ft,scheduler=exp_lr_scheduler, dataloader=NN_dataloaders,dataset_sizes=NN_dataset_sizes,training_epochs=50)
    ML_trainer.full_cycle_training()



DeepLearning_classifier()