import os
import time
import torch
import sys
import torchvision
import torchvision.transforms as transforms
import torchvision.transforms.functional as F
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score, precision_score, recall_score
import warnings
warnings.filterwarnings('ignore')


METRICS = {
    'accuracy': {
        'f': accuracy_score,
        'args': {}
    },
    'balanced_accuracy': {
        'f': balanced_accuracy_score,
        'args': {}
    },
    'f1': {
        'f': f1_score,
        'args': {'average': 'weighted'}
    },
    'precision': {
        'f': precision_score,
        'args': {'average': 'weighted'}
    },
    'recall': {
        'f': recall_score,
        'args': {'average': 'weighted'}
    }
}


NORM_MEAN = [0.485, 0.456, 0.406]
NORM_STD = [0.229, 0.224, 0.225]


def batch2image(dataloader):
    images, targets = next(iter(dataloader))
    channels = images.shape[1]
    if channels not in (3, 1):
        raise ValueError("Images must have 1 or 3 channels")
    mean = NORM_MEAN if channels == 3 else [sum(NORM_MEAN) / 3]
    std = NORM_STD if channels == 3 else [sum(NORM_STD) / 3]
    mean = torch.tensor(mean)
    std = torch.tensor(std)
    img_grid = torchvision.utils.make_grid(images)
    img_grid = F.normalize(img_grid, mean=(-mean / std).tolist(), std=(1.0 / std).tolist())
    return img_grid


class Transforms(transforms.Compose):

    def __init__(self, in_channels=3, out_channels=3, size=224, train=False, random_crop=False, horizontal_flip=False,
                 color_jitter=False):
        if out_channels not in (3, 1) or in_channels not in (3, 1):
            raise ValueError("Images must have 1 or 3 channels")
        mean = NORM_MEAN if out_channels == 3 else [sum(NORM_MEAN) / 3]
        std = NORM_STD if out_channels == 3 else [sum(NORM_STD) / 3]
        transforms_list = []
        if in_channels != out_channels:
            transforms_list.append(transforms.Grayscale(out_channels))
        if train:
            if random_crop:
                transforms_list.append(transforms.RandomResizedCrop(size, scale=(0.9, 1.1), ratio=(0.75, 1.33)))
            else:
                transforms_list.append(transforms.Resize(size))
            if horizontal_flip:
                transforms_list.append(transforms.RandomHorizontalFlip(p=0.5))
            if color_jitter:
                transforms_list.append(transforms.ColorJitter(brightness=(0.75, 1.5), contrast=(0.75, 1.5),
                                                              saturation=(0.75, 1.5), hue=(-0.1, 0.1)))
        else:
            transforms_list.append(transforms.Resize(size))
        transforms_list.extend([
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=std)
        ])
        super(Transforms, self).__init__(transforms_list)


class Timer(object):

    def __init__(self, v='ms', fmt='.3f'):
        self.start = 0
        self.val = 0
        self.v = v
        if v == 'ms':
            self.d = 1000.0
        elif v == 'mcs':
            self.d = 1000000.0
        elif v == 'ns':
            self.d = 1000000000.0
        else:
            self.v = 's'
            self.d = 1.0
        self.fmt = fmt

    def tick(self):
        self.start = time.time()

    def tock(self):
        self.val = (time.time() - self.start) * self.d
        return self.val

    def __str__(self):
        fmtstr = ('{0:' + self.fmt + '} ' + self.v).format(self.val)
        return fmtstr


class ProgressBar(object):

    def __init__(self, num):
        self.num = num
        self.done = 0
        self.percent = 0.0

    def update(self, percent):
        self.percent = percent if percent <= 1.0 else 1.0
        self.done = round(self.num * self.percent)

    def __str__(self):
        fmtstr = '[' + '=' * self.done + ' ' * (self.num - self.done) + '] ' + f"{self.percent * 100.0:6.2f}%"
        return fmtstr


class AverageMeter(object):

    def __init__(self, fmt='.3f'):
        self.fmt = fmt
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.cnt = 0

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.cnt = 0

    def update(self, val):
        self.val = val
        self.sum += val
        self.cnt += 1
        self.avg = self.sum / self.cnt

    def __str__(self):
        fmtstr = ('{0:' + self.fmt + '}').format(self.avg)
        return fmtstr


class AvgTimer(Timer):

    def __init__(self, v='ms', fmt='.3f'):
        super(AvgTimer, self).__init__(v=v, fmt=fmt)
        self.avg_meter = AverageMeter(fmt=fmt)

    def tock(self):
        val = super(AvgTimer, self).tock()
        self.avg_meter.update(val)
        return val

    def reset(self):
        self.avg_meter.reset()

    def __str__(self):
        fmtstr = '{0}/{1} {2}'.format(super(AvgTimer, self).__str__(), str(self.avg_meter), self.v)
        return fmtstr


class PyTorchTrainer(object):

    def __init__(self, device=None, logger=None, path=None, metrics=None):
        self.device = torch.device(device) if device else torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        self.logger = logger
        self.num_epochs = 10
        self.train_batches = 1
        self.val_batches = 1
        self.path = path or "./"
        self.metrics = metrics or METRICS
        self.models_path = os.path.join(self.path, "models")
        os.mkdir(self.models_path)
        self.batch_timer = AvgTimer(v='ms', fmt='.3f')
        self.epoch_timer = AvgTimer(v='s', fmt='.3f')
        self.train_timer = AvgTimer(v='s', fmt='.3f')
        self.epoch_progress_bar = ProgressBar(10)

    def train(self, model, optimizer, loss_criterion, train_dataloader, val_dataloader, scheduler=None, num_epochs=10):
        # Variables
        self.num_epochs = num_epochs
        self.train_batches = len(train_dataloader)
        self.val_batches = len(val_dataloader)
        # Print log
        self.log_start_train()
        # Timings
        self.train_timer.tick()
        self.epoch_timer.reset()
        for epoch in range(num_epochs):
            # Timings
            self.epoch_timer.tick()
            # Print log
            self.log_start_epoch(epoch)
            # Train batch
            train_loss, predictions, targets = self.forward_batches(model, optimizer, loss_criterion, train_dataloader,
                                                                    epoch, train=True)
            print()
            # Val batches
            val_loss, predictions, targets = self.forward_batches(model, optimizer, loss_criterion, val_dataloader,
                                                                  epoch, train=False)
            # Timings
            self.epoch_timer.tock()
            # Print log
            self.log_end_epoch(train_loss, val_loss, predictions.cpu().numpy(), targets.cpu().numpy(), epoch)
            # Save model
            self.save_model(model, epoch)
            # Make scheduler step
            if scheduler:
                scheduler.step(epoch=epoch)
        # Timings
        self.train_timer.tock()
        # Print log
        self.log_end_train()

    def forward_batches(self, model, optimizer, loss_criterion, data_loader, epoch, train=True):
        # Set model train or eval due to current phase
        if train:
            model.train()
        else:
            model.eval()
        # Preset variables
        avg_loss_value = AverageMeter(fmt='.3f')
        all_predictions = None
        all_targets = None
        batches = self.train_batches if train else self.val_batches
        # Timings
        self.batch_timer.reset()
        for batch_i, data in enumerate(data_loader, 1):
            # Timings
            self.batch_timer.tick()
            # Forward batch
            if train:
                loss_value = self.forward_batch(model, optimizer, loss_criterion, data, train=train)
            else:
                loss_value, predictions, targets = self.forward_batch(model, optimizer, loss_criterion, data, train=train)
                all_predictions = torch.cat((all_predictions, predictions)) if all_predictions is not None else predictions
                all_targets = torch.cat((all_targets, targets)) if all_targets is not None else targets
            avg_loss_value.update(loss_value)
            # Timings
            self.batch_timer.tock()
            # Print log
            self.log_batch(loss_value, batch_i, batches, epoch, train)
        # Return
        return avg_loss_value.avg, all_predictions, all_targets

    def forward_batch(self, model, optimizer, loss_criterion, batch_data, train=True):
        # Get Inputs and Targets and put them to device
        inputs, targets = batch_data
        inputs = inputs.to(self.device)
        targets = targets.to(self.device)
        with torch.set_grad_enabled(train):
            # Forward model to get outputs
            outputs = model.forward(inputs)
            # Calculate Loss Criterion
            loss = loss_criterion(outputs, targets)
        if train:
            # Zero optimizer gradients
            optimizer.zero_grad()
            # Calculate new gradients
            loss.backward()
            # Make optimizer step
            optimizer.step()
        # Variables
        loss_value = loss.item()
        if train:
            return loss_value
        predictions = outputs.argmax(dim=1).data
        targets = targets.data
        return loss_value, predictions, targets

    def save_model(self, model, epoch):
        torch.save(model.state_dict(), os.path.join(self.models_path, f"model_epoch_{epoch + 1}.pt"))

    def log_start_train(self):
        print(f"================================== Train ==================================")
        print(f"== Number of epochs:           {self.num_epochs:6d}")
        print(f"== Number of train batches:    {self.train_batches:6d}")
        print(f"== Number of validate batches: {self.val_batches:6d}")
        print(f"===========================================================================")

    def log_start_epoch(self, epoch):
        print(f"======== Epoch {epoch + 1}/{self.num_epochs}")

    def log_batch(self, loss, batch, batches, epoch, train):
        self.epoch_progress_bar.update(batch / batches)
        s = f"\r==== {'Train' if train else '  Val'} Batch {batch:6d}/{batches} {str(self.epoch_progress_bar)} " \
            f"loss[{loss:6.3f}] time[{str(self.batch_timer)}]"
        sys.stdout.write(s)
        if self.logger and train:
            self.logger.add_scalar(f"loss/{'train' if train else 'val'}_batch", loss, batch + epoch * batches)

    def log_end_epoch(self, train_loss, val_loss, predictions, targets, epoch):
        s = f"\n======== train_loss[{train_loss:6.3f}] val_loss[{val_loss:6.3f}] "
        calculated_metrics = {}
        for metric in self.metrics:
            calculated_metrics[metric] = self.metrics[metric]['f'](predictions, targets, **self.metrics[metric]['args'])
            s += f"{metric}[{calculated_metrics[metric]:6.3f}] "
        s += f"time[{str(self.epoch_timer)}]\n"
        print(s)
        if self.logger:
            self.logger.add_scalars('loss/epoch', {'train': train_loss, 'val': val_loss}, epoch)
            for metric in calculated_metrics:
                self.logger.add_scalar('epoch_metrics/' + metric, calculated_metrics[metric], epoch)

    def log_end_train(self):
        print(f"===========================================================================")
        print(f"== time[{str(self.train_timer)}]")
        print(f"================================== Train ==================================")
