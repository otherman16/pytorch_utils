import os
import math
import random
import torch
import torchvision
import torchvision.transforms as transforms
import torchvision.transforms.functional as F
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score, precision_score, recall_score
import matplotlib.pyplot as plt
from torch.utils.tensorboard import SummaryWriter

from tqdm.autonotebook import tqdm


METRICS = {
    'accuracy': {
        'f': accuracy_score,
        'args': {}
    },
    # 'balanced_accuracy': {
    #     'f': balanced_accuracy_score,
    #     'args': {}
    # },
    # 'f1': {
    #     'f': f1_score,
    #     'args': {'average': 'weighted'}
    # },
    # 'precision': {
    #     'f': precision_score,
    #     'args': {'average': 'weighted'}
    # },
    # 'recall': {
    #     'f': recall_score,
    #     'args': {'average': 'weighted'}
    # }
}


NORM_MEAN = [0.485, 0.456, 0.406]
NORM_STD = [0.229, 0.224, 0.225]


def make_image_label_grid(images, labels=None, class_names=None):
    channels = images.shape[1]
    if channels not in (3, 1):
        raise ValueError("Images must have 1 or 3 channels")
    mean = NORM_MEAN if channels == 3 else [sum(NORM_MEAN) / 3]
    std = NORM_STD if channels == 3 else [sum(NORM_STD) / 3]
    mean = torch.tensor(mean)
    std = torch.tensor(std)
    mean = (-mean / std).tolist()
    std = (1.0 / std).tolist()
    img_grid = torchvision.utils.make_grid(images)
    img_grid = F.normalize(img_grid, mean=mean, std=std)
    return img_grid


def make_image_label_figure(images, labels=None, class_names=None):
    channels = images.shape[1]
    if channels not in (3, 1):
        raise ValueError("Images must have 1 or 3 channels")
    mean = NORM_MEAN if channels == 3 else [sum(NORM_MEAN) / 3]
    std = NORM_STD if channels == 3 else [sum(NORM_STD) / 3]
    mean = torch.tensor(mean)
    std = torch.tensor(std)
    mean = (-mean / std).tolist()
    std = (1.0 / std).tolist()
    n = int(math.sqrt(len(images)))
    figure = plt.figure(figsize=(n, n))
    figure.subplots_adjust(hspace=0.4, wspace=0.4)
    for i in range(n*n):
        image, label = images[i], (0 if labels is None else labels[i])
        image = F.normalize(image, mean=mean, std=std)
        image = image.permute(1, 2, 0)
        image = torch.squeeze(image)
        image = (image * 255).int()
        plt.subplot(n, n, i + 1, title='NA' if class_names is None else class_names[label])
        plt.xticks([])
        plt.yticks([])
        plt.grid(False)
        plt.imshow(image, cmap='gray' if channels == 1 else None)
    return figure


class Transforms(transforms.Compose):

    def __init__(self, in_channels=1, out_channels=1, size=(32, 32)):
        if out_channels not in (3, 1) or in_channels not in (3, 1):
            raise ValueError("Images must have 1 or 3 channels")
        mean = NORM_MEAN if out_channels == 3 else [sum(NORM_MEAN) / 3]
        std = NORM_STD if out_channels == 3 else [sum(NORM_STD) / 3]
        transforms_list = []
        if in_channels != out_channels:
            transforms_list.append(transforms.Grayscale(out_channels))
        transforms_list.extend([
            transforms.Resize(size),
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=std)])
        super(Transforms, self).__init__(transforms_list)


class TrainTransforms(transforms.Compose):

    def __init__(self, in_channels=1, out_channels=1, size=(32, 32), random_crop=False, random_affine=False,
                 horizontal_flip=False, color_jitter=False, random_erasing=False):
        if out_channels not in (3, 1) or in_channels not in (3, 1):
            raise ValueError("Images must have 1 or 3 channels")
        mean = NORM_MEAN if out_channels == 3 else [sum(NORM_MEAN) / 3]
        std = NORM_STD if out_channels == 3 else [sum(NORM_STD) / 3]
        transforms_list = []
        if in_channels != out_channels:
            transforms_list.append(transforms.Grayscale(out_channels))
        if random_affine:
            transforms_list.append(transforms.RandomAffine(degrees=10.0, translate=(0.25, 0.25),
                                                           shear=(-10, 10, -10, 10)))
        if random_crop:
            transforms_list.append(transforms.RandomResizedCrop(size, scale=(0.9, 1.1), ratio=(0.75, 1.33)))
        else:
            transforms_list.append(transforms.Resize(size))
        if horizontal_flip:
            transforms_list.append(transforms.RandomHorizontalFlip(p=0.5))
        if color_jitter:
            transforms_list.append(transforms.ColorJitter(brightness=(0.75, 1.5), contrast=(0.75, 1.5),
                                                          saturation=(0.75, 1.5), hue=(-0.1, 0.1)))
        transforms_list.append(transforms.ToTensor())
        if random_erasing:
            transforms_list.append(transforms.RandomErasing(p=0.2, scale=(0.02, 0.2), ratio=(0.3, 3.3), value=0))
        transforms_list.append(transforms.Normalize(mean=mean, std=std))
        super(TrainTransforms, self).__init__(transforms_list)


class TrainerProgressBar(tqdm):

    def __init__(self, desc=None, total=10, unit='it', position=None):
        super(TrainerProgressBar, self).__init__(
            desc=desc, total=total, leave=True, unit=unit, position=position, dynamic_ncols=True
        )

    def reset(self, total=None, desc=None, ordered_dict=None):
        # super(TrainerProgressBar, self).reset(total)
        self.last_print_n = self.n = 0
        self.last_print_t = self.start_t = self._time()
        if total is not None:
            self.total = total
        super(TrainerProgressBar, self).refresh()
        if desc is not None:
            super(TrainerProgressBar, self).set_description(desc)
        if ordered_dict is not None:
            super(TrainerProgressBar, self).set_postfix(ordered_dict)

    def update(self, desc=None, ordered_dict=None, n=1):
        if desc is not None:
            super(TrainerProgressBar, self).set_description(desc)
        if ordered_dict is not None:
            super(TrainerProgressBar, self).set_postfix(ordered_dict)
        super(TrainerProgressBar, self).update(n)


class TensorBoardLogger(SummaryWriter):

    def __init__(self, log_dir=None, comment='', purge_step=None, max_queue=10,
                 flush_secs=120, filename_suffix='', class_names=None, testloader=None, device=None):
        super(TensorBoardLogger, self).__init__(log_dir=log_dir, comment=comment, purge_step=purge_step,
                                                max_queue=max_queue, flush_secs=flush_secs,
                                                filename_suffix=filename_suffix)
        self.class_names = class_names
        self.testloader = testloader
        self.device = torch.device(device) if device else torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    def epoch_callback(self, net, epoch, lr, train_loss, val_loss, train_metrics_dict, val_metrics_dict):
        super(TensorBoardLogger, self).add_scalar(f"epoch/lr", lr, epoch)
        super(TensorBoardLogger, self).add_scalars(f"epoch/loss", {'train': train_loss, 'val': val_loss}, epoch)
        for metric_name in train_metrics_dict:
            super(TensorBoardLogger, self).add_scalars(f"epoch/{metric_name}",
                                                       {'train': train_metrics_dict[metric_name],
                                                        'val': val_metrics_dict[metric_name]}, epoch)
        if self.testloader:
            inputs, targets = next(iter(self.testloader))
            _inputs = inputs.to(self.device)
            predictions = net.forward(_inputs).argmax(dim=1).data.cpu()
            start = random.randrange(0, len(targets)-9)
            stop = start + 9
            super(TensorBoardLogger, self).add_figure(
                'examples/real', make_image_label_figure(inputs[start:stop], targets[start:stop], self.class_names)
            )
            super(TensorBoardLogger, self).add_figure(
                'examples/predicted', make_image_label_figure(inputs[start:stop], predictions[start:stop],
                                                              self.class_names)
            )

    def batch_callback(self, train, epoch, batch, batches, loss, metrics_dict):
        section = 'train_batch' if train else 'val_batch'
        super(TensorBoardLogger, self).add_scalar(f"{section}/loss", loss, batch + epoch * batches)
        for metric_name, metric_value in metrics_dict.items():
            if isinstance(metric_value, dict):
                super(TensorBoardLogger, self).add_scalars(f"{section}/{metric_name}", metric_value,
                                                           batch + epoch * batches)
            else:
                super(TensorBoardLogger, self).add_scalar(f"{section}/{metric_name}", metric_value,
                                                          batch + epoch * batches)


class PyTorchTrainer(object):

    def __init__(self, device=None, metrics=None, epoch_callback=None, batch_callback=None):
        self.device = torch.device(device) if device else torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        self.metrics = metrics or METRICS
        self.epoch_callback = epoch_callback
        self.batch_callback = batch_callback

        self.train_pb = None
        self.epoch_train_pb = None
        self.epoch_val_pb = None

    def train(self, model, optimizer, loss_criterion, train_data_loader, val_data_loader, path, scheduler=None,
              epochs=10):
        models_path = os.path.join(path, "models")
        os.mkdir(models_path)
        # Print log
        print(f"=============================== Training NN ===============================")
        print(f"== Epochs:              {epochs:6d}")
        print(f"== Train batch size:    {train_data_loader.batch_size:6d}")
        print(f"== Train batches:       {len(train_data_loader):6d}")
        print(f"== Validate batch size: {val_data_loader.batch_size:6d}")
        print(f"== Validate batches:    {len(val_data_loader):6d}")
        print(f"===========================================================================")
        # Initialize progress bars
        self.train_pb = TrainerProgressBar(desc=f'== Epoch {1}', total=epochs, unit='epoch', position=0)
        self.epoch_train_pb = TrainerProgressBar(desc=f'== Train {1}', total=len(train_data_loader),
                                                 unit='batch', position=1)
        self.epoch_val_pb = TrainerProgressBar(desc=f'== Val {1}', total=len(val_data_loader), unit='batch',
                                               position=2)
        # Reset progress bars
        self.train_pb.reset(total=epochs)
        self.epoch_train_pb.reset(total=len(train_data_loader))
        self.epoch_val_pb.reset(total=len(val_data_loader))
        for epoch in range(epochs):
            # Train batches
            train_loss, train_metrics_dict = self.forward_batches(model, optimizer, loss_criterion,
                                                                  train_data_loader, epoch, train=True)
            # Val batches
            val_loss, val_metrics_dict = self.forward_batches(model, optimizer, loss_criterion,
                                                              val_data_loader, epoch, train=False)
            # Save model
            torch.save(model.state_dict(), os.path.join(models_path, f"model_epoch_{epoch+1}.pt"))
            # Make scheduler step
            if scheduler:
                if isinstance(scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                    scheduler.step(train_loss, epoch=epoch)
                else:
                    scheduler.step(epoch=epoch)
            # Update progress bar
            lr = optimizer.param_groups[0]['lr']
            if self.epoch_callback:
                self.epoch_callback(model, epoch, lr, train_loss, val_loss, train_metrics_dict, val_metrics_dict)
            metrics_dict = {'lr': lr}
            for metric_name in train_metrics_dict:
                metrics_dict[f"train_{metric_name}"] = train_metrics_dict[metric_name]
                metrics_dict[f"val_{metric_name}"] = val_metrics_dict[metric_name]
            metrics_dict.update({'train_loss': train_loss, 'val_loss': val_loss})
            self.train_pb.update(desc=f'== Epoch {epoch+1}', ordered_dict=metrics_dict)
        # Close progress bars
        self.train_pb.close()
        self.epoch_train_pb.close()
        self.epoch_val_pb.close()
        # Print log
        print(f"===========================================================================")

    def forward_batches(self, model, optimizer, loss_criterion, data_loader, epoch, train=True):
        # Set model train or eval due to current phase
        if train:
            model.train()
        else:
            model.eval()
        # Preset variables
        avg_loss_value = 0
        avg_metrics_dict = None
        batches = len(data_loader)
        # Reset progress bar
        if train:
            self.epoch_train_pb.reset(batches, f"== Train {epoch+1}")
        else:
            self.epoch_val_pb.reset(batches, f"== Val {epoch+1}")
        for batch_i, data in enumerate(data_loader, 1):
            # Forward batch
            loss_value, predictions, targets = self.forward_batch(model, optimizer, loss_criterion, data,
                                                                  train=train)
            metrics_dict = self.metrics_dict(predictions, targets)
            # Update variables
            avg_loss_value += loss_value
            if avg_metrics_dict is None:
                avg_metrics_dict = metrics_dict.copy()
            else:
                for metric_name in avg_metrics_dict:
                    avg_metrics_dict[metric_name] += metrics_dict[metric_name]
            # Update progress bar
            if self.batch_callback:
                self.batch_callback(train, epoch, batch_i, batches, loss_value, metrics_dict)
            metrics_dict.update({'loss': avg_loss_value/batch_i})
            if train:
                self.epoch_train_pb.update(ordered_dict=metrics_dict)
            else:
                self.epoch_val_pb.update(ordered_dict=metrics_dict)
        # Update variables
        avg_loss_value /= batches
        for metric_name in avg_metrics_dict:
            avg_metrics_dict[metric_name] /= batches
        # Return
        return avg_loss_value, avg_metrics_dict

    def forward_batch(self, model, optimizer, loss_criterion, batch_data, train=True):
        # Get Inputs and Targets and put them to device
        inputs, targets = batch_data
        _inputs = inputs.to(self.device)
        _targets = targets.to(self.device)
        with torch.set_grad_enabled(train):
            # Forward model to get outputs
            _outputs = model.forward(_inputs)
            # Calculate Loss Criterion
            loss = loss_criterion(_outputs, _targets)
        if train:
            # Zero optimizer gradients
            optimizer.zero_grad()
            # Calculate new gradients
            loss.backward()
            # Make optimizer step
            optimizer.step()
        # Variables
        loss_value = loss.item()
        predictions = _outputs.argmax(dim=1).data.cpu()
        targets = targets.data
        return loss_value, predictions, targets

    def metrics_dict(self, predictions, targets):
        d = {}
        for metric_name in self.metrics:
            metric_value = self.metrics[metric_name]['f'](predictions, targets, **self.metrics[metric_name]['args'])
            d[metric_name] = metric_value

        return d
