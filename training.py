import torch, torch.nn as nn
import time
import os
from tqdm import tqdm
from sklearn.metrics import confusion_matrix, precision_score, recall_score, f1_score
from torch.utils.tensorboard import SummaryWriter

class Trainer():
    def __init__(self, model, criterion, optimizer, scheduler, dataloader, dataset_sizes, training_epochs) -> None:
        self.model = model
        self.criterion = criterion
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.train_dataloader = dataloader["train"]
        self.val_dataloader = dataloader["val"]
        self.epochs = training_epochs
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.training_data_size = dataset_sizes["train"]
        self.val_data_size = dataset_sizes["val"]
        self.ckpt = os.path.join("best_weights", 'best.pt')
        self.model = self.model.to(self.device)
        # if torch.cuda.device_count() > 1:
        #     print("Let's use", torch.cuda.device_count(), "GPUs!")
        #     self.model = nn.DataParallel(self.model)
        self.criterion = self.criterion.to(self.device)
        self.logger = SummaryWriter("Logs_no_norm")



    def full_cycle_training(self):
        since = time.time()
        best_acc = 0.0
        print("Training Started")
        for epoch in tqdm(range(self.epochs), desc="Epochs progress bar"):
            # Each epoch has a training and validation phase
            train_acc, train_loss, train_tn, train_fp, train_fn, train_tp, train_precision, train_recall, train_f1 = self.train_one_epoch()
            val_acc, val_loss, val_tn, val_fp, val_fn, val_tp, val_precision, val_recall, val_f1 = self.val_one_epoch()
            self.logger.add_scalars('loss', {'train':train_loss, 'val':val_loss}, epoch)
            self.logger.add_scalars('Acc', {'train':train_acc, 'val':val_acc}, epoch)
            self.logger.add_scalars('PR_train', {'train_precision':train_precision, 'train_recall':train_recall, 'train_f1':train_f1}, epoch)
            self.logger.add_scalars('PR_val', {'val_precision':val_precision, 'val_recall':val_recall, 'val_f1':val_f1}, epoch)
            self.logger.add_scalars('confusion_matrix_train', {'train_TN':train_tn, 'train_FP':train_fp, 'train_FN':train_fn,'train_TP':train_tp}, epoch)
            self.logger.add_scalars('confusion_matrix_val', {'val_TN':val_tn, 'val_FP':val_fp, 'val_FN':val_fn, 'val_TP': val_tp}, epoch)

            
            

            # deep copy the model
            if val_acc > best_acc:
                best_acc = val_acc
                torch.save(self.model.state_dict(), self.ckpt)

        time_elapsed = time.time() - since
        print(f'Training complete in {time_elapsed // 60:.0f}m {time_elapsed % 60:.0f}s')
        print(f'Best val Acc: {best_acc:4f}')
        self.logger.flush() 
        self.logger.close()


    def train_one_epoch(self):
        # Each epoch has a training and validation phase
        self.model.train()  # Set model to training mode
        running_loss = 0.0
        running_corrects = 0

        all_preds = []
        all_labels = []

        # Iterate over data.
        for inputs, labels in self.train_dataloader:#tqdm(self.train_dataloader, desc="Training loop"):
            inputs = inputs.to(self.device)
            labels = labels.to(self.device)

            # zero the parameter gradients
            self.optimizer.zero_grad()

            outputs = self.model(inputs)
            _, preds = torch.max(outputs, 1)
            loss = self.criterion(outputs, labels)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

            loss.backward()
            self.optimizer.step()
            self.scheduler.step()
            
            # statistics
            running_loss += loss.item() * inputs.size(0)
            running_corrects += torch.sum(preds == labels.data)
            

        epoch_loss = running_loss / self.training_data_size
        epoch_acc = running_corrects.double() / self.training_data_size
        # Calculate confusion matrix
        cm = confusion_matrix(all_labels, all_preds)

        # Extract TP, TN, FP, FN from confusion matrix
        tn, fp, fn, tp = cm.ravel()

        # Calculate Precision, Recall, F1 Score
        precision = precision_score(all_labels, all_preds)
        recall = recall_score(all_labels, all_preds)
        f1 = f1_score(all_labels, all_preds)

        return epoch_acc, epoch_loss, tn, fp, fn, tp, precision, recall, f1

    def val_one_epoch(self):
        
        self.model.eval()   # Set model to evaluate mode
        running_loss = 0.0
        running_corrects = 0
        all_preds = []
        all_labels = []
        # Iterate over data.
        for inputs, labels in self.val_dataloader:#tqdm(self.val_dataloader,desc="Validation loop"):
            inputs = inputs.to(self.device)
            labels = labels.to(self.device)

            # zero the parameter gradients
            self.optimizer.zero_grad()

            outputs = self.model(inputs)
            _, preds = torch.max(outputs, 1)
            loss = self.criterion(outputs, labels)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            # statistics
            running_loss += loss.item() * inputs.size(0)
            running_corrects += torch.sum(preds == labels.data)

        epoch_loss = running_loss / self.val_data_size
        epoch_acc = running_corrects.double() / self.val_data_size
        # Calculate confusion matrix
        cm = confusion_matrix(all_labels, all_preds)

        # Extract TP, TN, FP, FN from confusion matrix
        tn, fp, fn, tp = cm.ravel()

        # Calculate Precision, Recall, F1 Score
        precision = precision_score(all_labels, all_preds)
        recall = recall_score(all_labels, all_preds)
        f1 = f1_score(all_labels, all_preds)

        return epoch_acc, epoch_loss, tn, fp, fn, tp, precision, recall, f1