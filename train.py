import torch
import argparse
import torch.nn as nn
import torch.optim as optim
import os
import numpy as np
import random
from presents import VideoClassificationPresetTrain
from tqdm.auto import tqdm
from model import build_model

from dataset import VideoClassificationDataset
from utils import save_model, save_plots, SaveBestModel
from class_names import class_names
from torchvision.datasets.samplers import (
    RandomClipSampler, UniformClipSampler
)
from torch.utils.data.dataloader import default_collate

seed = 42
np.random.seed(seed)
random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
torch.backends.cudnn.deterministic = True
torch.cuda.empty_cache()
import os


os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'

# Construct the argument parser.
seed = 42
np.random.seed(seed)
random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
torch.backends.cudnn.deterministic = True

# Construct the argument parser.
parser = argparse.ArgumentParser()
parser.add_argument(
    '-e', '--epochs',
    type=int,
    default=10,
    help='Number of epochs to train our network for'
)
parser.add_argument(
    '-lr', '--learning-rate',
    type=float,
    dest='learning_rate',
    default=0.001,
    help='Learning rate for training the model'
)
parser.add_argument(
    '-model_type', '--model_type',
    type=str,
    dest='model_type',
    default='class',
    help='what type of model: reg or class?'
)
parser.add_argument(
    '-b', '--batch-size',
    dest='batch_size',
    default=32,
    type=int
)
parser.add_argument(
    '-ft', '--fine-tune',
    dest='fine_tune',
    action='store_true',
    help='pass this to fine tune all layers'
)
parser.add_argument(
    '--save-name',
    dest='save_name',
    default='model',
    help='file name of the final model to save'
)
parser.add_argument(
    '--scheduler',
    action='store_true',
    help='use learning rate scheduler if passed'
)
parser.add_argument(
    '--workers',
    default=4,
    help='number of parallel workers for data loader',
    type=int
)
parser.add_argument(
    '--clip-len',
    dest='clip_len',
    default=16,
    help='number of frames per clip',
    type=int
)
parser.add_argument(
    '--clips-per-video',
    dest='clips_per_video',
    default=5,
    help='maximum number of clips per video',
    type=int
)
parser.add_argument(
    '--frame-rate',
    dest='frame_rate',
    default=15,
    help='the frame rate of each clip',
    type=int
)
parser.add_argument(
    '--imgsz',
    default=(128, 171),
    nargs='+',
    type=int,
    help='image resize resolution'
)
parser.add_argument(
    '--crop-size',
    dest='crop_size',
    default=(112, 112),
    nargs='+',
    type=int,
    help='image cropping resolution'
)
args = parser.parse_args()


def collate_fn(batch):
    batch = [(d[0], d[1]) for d in batch]
    return default_collate(batch)


# Training function.
def train(model, trainloader, optimizer, criterion, model_type='class'):
    model.train()
    print('Training')
    train_running_loss = 0.0
    train_running_correct = 0
    bs_accumuator = 0
    counter = 0
    prog_bar = tqdm(
        trainloader,
        total=len(trainloader),
        bar_format='{l_bar}{bar:20}{r_bar}{bar:-20b}'
    )
    for i, data in enumerate(prog_bar):
        counter += 1
        image, labels = data
        image = image.to(device)
        labels = labels.to(device)
        optimizer.zero_grad()
        # Forward pass.
        outputs = model(image)
        bs_accumuator += outputs.shape[0]
        # Calculate the loss.
        loss = criterion(outputs, labels)
        train_running_loss += loss.item()

        if model_type == 'class':
            # Calculate the accuracy.
            _, preds = torch.max(outputs.data, 1)

            train_running_correct += (preds == labels).sum().item()
        else:
            mape = torch.mean(torch.abs((labels - outputs) / labels)) * 100
            train_running_correct += mape.item()
        # Backpropagation.
        loss.backward()
        # Update the weights.
        optimizer.step()
    # Loss and accuracy for the complete epoch.
    epoch_loss = train_running_loss / counter
    if model_type == 'class':
        epoch_output = 100. * (train_running_correct / bs_accumuator)
    else:
        epoch_output = train_running_correct / counter  # Average MAPE
    return epoch_loss, epoch_output


# Validation function.
def validate(model, testloader, criterion, model_type="class"):
    model.eval()
    print('Validation')
    valid_running_loss = 0.0
    valid_running_correct = 0
    bs_accumuator = 0
    counter = 0
    prog_bar = tqdm(
        testloader,
        total=len(testloader),
        bar_format='{l_bar}{bar:20}{r_bar}{bar:-20b}'
    )
    with torch.no_grad():
        for i, data in enumerate(prog_bar):
            counter += 1

            image, labels = data
            image = image.to(device)
            labels = labels.to(device)
            # Forward pass.
            outputs = model(image)
            bs_accumuator += outputs.shape[0]
            # Calculate the loss.
            loss = criterion(outputs, labels)
            valid_running_loss += loss.item()

            if model_type == 'class':
                # Calculate the accuracy.
                _, preds = torch.max(outputs.data, 1)
                valid_running_correct += (preds == labels).sum().item()
            else:
                mape = torch.mean(torch.abs((labels - outputs) / labels)) * 100
                valid_running_correct += mape.item()


    # Loss and accuracy for the complete epoch.
    epoch_loss = valid_running_loss / counter
    if model_type == 'class':
        epoch_output = 100. * (valid_running_correct / bs_accumuator)
    else:
        epoch_output = valid_running_correct / counter  # Average MAPE
    return epoch_loss, epoch_output


if __name__ == '__main__':
    # Create a directory with the model name for outputs.
    out_dir = os.path.join('..', 'outputs')
    os.makedirs(out_dir, exist_ok=True)

    ## Data Loading.
    train_crop_size = tuple(args.crop_size)
    train_resize_size = tuple(args.imgsz)


    transform_train = VideoClassificationPresetTrain(
        crop_size=train_crop_size,
        resize_size=train_resize_size
    )
    transform_valid = VideoClassificationPresetTrain(
        crop_size=train_crop_size,
        resize_size=train_resize_size,
        hflip_prob=0.0
    )

    # Load the training and validation datasets.
    dataset_train = VideoClassificationDataset(
        'train_val_test',
        frames_per_clip=args.clip_len,
        frame_rate=args.frame_rate,
        split="train",
        transform=transform_train,
        extensions=(
            "mp4",
            'avi',
            'mov'
        ),
        output_format="TCHW",
        num_workers=args.workers
    )
    dataset_valid = VideoClassificationDataset(
        'train_val_test',
        frames_per_clip=args.clip_len,
        frame_rate=args.frame_rate,
        split="valid",
        transform=transform_valid,
        extensions=(
            "mp4",
            'avi',
            'mov'
        ),
        output_format="TCHW",
        num_workers=args.workers
    )
    print(f"[INFO]: Number of training images: {len(dataset_train)}")
    print(f"[INFO]: Number of validation images: {len(dataset_valid)}")
    if args.model_type == 'class':

        print(f"[INFO]: Classes: {class_names}")
    else:
        print("Regression Model")

    # Load the training and validation data loaders.
    train_sampler = RandomClipSampler(
        dataset_train.video_clips, max_clips_per_video=args.clips_per_video
    )
    test_sampler = UniformClipSampler(
        dataset_valid.video_clips, num_clips_per_video=args.clips_per_video
    )
    train_loader = torch.utils.data.DataLoader(
        dataset_train,
        batch_size=args.batch_size,
        sampler=train_sampler,
        num_workers=args.workers,
        pin_memory=True,
        collate_fn=collate_fn,
    )
    valid_loader = torch.utils.data.DataLoader(
        dataset_valid,
        batch_size=args.batch_size,
        sampler=test_sampler,
        num_workers=args.workers,
        pin_memory=True,
        collate_fn=collate_fn,
    )

    # Learning_parameters.
    lr = args.learning_rate
    epochs = args.epochs
    device = ('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Computation device: {device}")
    print(f"Learning rate: {lr}")
    print(f"Epochs to train for: {epochs}\n")

    # Load the model.
    model = build_model(
        fine_tune=args.fine_tune,
        num_classes=len(class_names),
        model_type=args.model_type
    ).to(device)
    print(model)

    # Total parameters and trainable parameters.
    total_params = sum(p.numel() for p in model.parameters())
    print(f"{total_params:,} total parameters.")
    total_trainable_params = sum(
        p.numel() for p in model.parameters() if p.requires_grad)
    print(f"{total_trainable_params:,} training parameters.")

    # Optimizer.
    # optimizer = torch.optim.SGD(
    # model.parameters(),
    # lr=lr,
    # momentum=0.9,
    # )
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    # Loss function.
    if args.model_type == 'class':
        criterion = nn.CrossEntropyLoss()
    else:
        criterion = nn.MSELoss()

    # Initialize `SaveBestModel` class.
    save_best_model = SaveBestModel()

    # LR scheduler.
    scheduler = optim.lr_scheduler.MultiStepLR(
        optimizer, milestones=[25], gamma=0.1, verbose=True
    )

    # Lists to keep track of losses and accuracies.
    train_loss, valid_loss = [], []
    train_output, valid_output = [], []
    # Start the training.
    for epoch in range(epochs):
        print(f"[INFO]: Epoch {epoch + 1} of {epochs}")
        train_epoch_loss, train_epoch_output = train(
            model, train_loader, optimizer, criterion, model_type=args.model_type
        )
        valid_epoch_loss, valid_epoch_output = validate(
            model, valid_loader, criterion, model_type=args.model_type
        )
        train_loss.append(train_epoch_loss)
        valid_loss.append(valid_epoch_loss)
        train_output.append(train_epoch_output)
        valid_output.append(valid_epoch_output)
        print(f"Training loss: {train_epoch_loss:.3f}, training output: {train_epoch_output:.3f}")
        print(f"Validation loss: {valid_epoch_loss:.3f}, validation output: {valid_epoch_output:.3f}")
        save_best_model(
            valid_epoch_loss, epoch, model, out_dir, args.save_name
        )
        if args.scheduler:
            scheduler.step()
        print('-' * 50)

    # Save the trained model weights.
    save_model(epochs, model, optimizer, criterion, out_dir, args.save_name)
    # Save the loss and accuracy plots.
    save_plots(train_output, valid_output, train_loss, valid_loss, out_dir)
    print('TRAINING COMPLETE')