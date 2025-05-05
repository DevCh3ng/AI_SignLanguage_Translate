import os
import argparse

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import transforms
import videotransforms

import numpy as np

from configs import Config
from pytorch_i3d import InceptionI3d
from datasets.nslt_dataset import NSLT as Dataset

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = '0'

parser = argparse.ArgumentParser()
parser.add_argument('-mode', type=str, help='rgb or flow')
parser.add_argument('-save_model', type=str)
parser.add_argument('-root', type=str)
parser.add_argument('--num_class', type=int)

args = parser.parse_args()

torch.manual_seed(0)
np.random.seed(0)

torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

def run(configs,
        mode='rgb',
        root='/ssd/Charades_v1_rgb',
        train_split='charades/charades.json',
        save_model='',
        weights=None):
    print(configs)

    train_loss_history = []
    train_acc_history = []
    val_loss_history = []
    val_acc_history = []

    # Video pixel manipulations
    train_transforms = transforms.Compose([
        videotransforms.RandomCrop(224),
        videotransforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1)
    ])
    test_transforms = transforms.Compose([videotransforms.CenterCrop(224)])

    dataset = Dataset(train_split, 'train', root, mode, train_transforms)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=configs.batch_size, shuffle=True, num_workers=0,
                                             pin_memory=True)

    val_dataset = Dataset(train_split, 'test', root, mode, test_transforms)
    val_dataloader = torch.utils.data.DataLoader(val_dataset, batch_size=configs.batch_size, shuffle=True, num_workers=2,
                                                 pin_memory=False)

    dataloaders = {'train': dataloader, 'test': val_dataloader}
    datasets = {'train': dataset, 'test': val_dataset}

    # Setup the model
    if mode == 'flow':
        i3d = InceptionI3d(400, in_channels=2)
        i3d.load_state_dict(torch.load('weights/flow_imagenet.pt'))
    else:
        i3d = InceptionI3d(400, in_channels=3)
        i3d.load_state_dict(torch.load('weights/rgb_imagenet.pt'))

    num_classes = dataset.num_classes
    i3d.replace_logits(num_classes)

    if weights:
        print('Loading weights {}'.format(weights))
        i3d.load_state_dict(torch.load(weights))

    i3d.cuda()
    i3d = nn.DataParallel(i3d)

    lr = configs.init_lr
    weight_decay = configs.adam_weight_decay
    optimizer = optim.Adam(i3d.parameters(), lr=lr, weight_decay=weight_decay)

    num_steps_per_update = configs.update_per_step
    steps = 0
    epoch = 0

    best_val_score = 0
    best_val_loss = float('inf')  # assume infinite loss
    patience_counter = 0  # cntr for early stopping
    early_stopping_patience = 10  # Num of continuos same epochs

    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=5, factor=0.3)

    # Training loop
    while steps < configs.max_steps and epoch < 400:
        print('Step {}/{}'.format(steps, configs.max_steps))
        print('-' * 10)

        epoch += 1
        for phase in ['train', 'test']:
            collected_vids = []

            if phase == 'train':
                i3d.train(True)
            else:
                i3d.train(False)

            tot_loss = 0.0
            tot_loc_loss = 0.0
            tot_cls_loss = 0.0
            num_iter = 0
            optimizer.zero_grad()

            confusion_matrix = np.zeros((num_classes, num_classes), dtype=np.int_)
            for data in dataloaders[phase]:
                num_iter += 1
                if data == -1:
                    continue

                inputs, labels, vid = data
                inputs = inputs.cuda()
                t = inputs.size(2)
                labels = labels.cuda()

                per_frame_logits = i3d(inputs, pretrained=False)
                per_frame_logits = F.upsample(per_frame_logits, t, mode='linear')

                loc_loss = F.binary_cross_entropy_with_logits(per_frame_logits, labels)
                tot_loc_loss += loc_loss.data.item()

                predictions = torch.max(per_frame_logits, dim=2)[0]
                gt = torch.max(labels, dim=2)[0]

                cls_loss = F.binary_cross_entropy_with_logits(torch.max(per_frame_logits, dim=2)[0],
                                                              torch.max(labels, dim=2)[0])
                tot_cls_loss += cls_loss.data.item()

                for i in range(per_frame_logits.shape[0]):
                    confusion_matrix[torch.argmax(gt[i]).item(), torch.argmax(predictions[i]).item()] += 1

                loss = (0.5 * loc_loss + 0.5 * cls_loss) / num_steps_per_update
                tot_loss += loss.data.item()
                loss.backward()

                if num_iter == num_steps_per_update and phase == 'train':
                    steps += 1
                    num_iter = 0
                    optimizer.step()
                    optimizer.zero_grad()
                    if steps % 10 == 0:
                        acc = float(np.trace(confusion_matrix)) / np.sum(confusion_matrix)
                        print(f"Epoch {epoch}, Step {steps}: Loc Loss: {tot_loc_loss / (10 * num_steps_per_update):.4f} "
                            f"Cls Loss: {tot_cls_loss / (10 * num_steps_per_update):.4f} "
                            f"Tot Loss: {tot_loss / 10:.4f} Accu: {acc:.4f}",flush=True)
                        tot_loss = tot_loc_loss = tot_cls_loss = 0.

            if phase == 'train':
                train_loss_history.append(tot_loss / len(dataloaders['train']))
                train_acc = float(np.trace(confusion_matrix)) / np.sum(confusion_matrix)
                train_acc_history.append(train_acc)

            if phase == 'test':
                val_score = float(np.trace(confusion_matrix)) / np.sum(confusion_matrix)
                val_loss_history.append(tot_loss / len(dataloaders['test']))
                val_acc_history.append(val_score)

                if tot_loss < best_val_loss:
                    best_val_loss = tot_loss
                    patience_counter = 0  # Reset the counter if the model improves
                    if val_score > best_val_score or epoch % 2 == 0:
                        best_val_score = val_score
                        model_name = save_model + "nslt_" + str(num_classes) + "_" + str(steps).zfill(
                            6) + '_%3f.pt' % val_score
                        torch.save(i3d.module.state_dict(), model_name)
                        print(f"Model saved: {model_name}")
                else:
                    patience_counter += 1
                    if patience_counter >= early_stopping_patience:
                        print("Early stopping triggered!")
                        return  

                print(f"VALIDATION - Epoch {epoch}: Loc Loss: {tot_loc_loss / num_iter:.4f} "
                      f"Cls Loss: {tot_cls_loss / num_iter:.4f} "
                      f"Tot Loss: {(tot_loss * num_steps_per_update) / num_iter:.4f} "
                      f"Accuracy: {val_score:.4f}",flush=True)

                scheduler.step(tot_loss * num_steps_per_update / num_iter)


if __name__ == '__main__':
    mode = 'rgb'
    root = {'word': 'data/WLASL2000'}

    save_model = 'checkpoints/'
    train_split = 'preprocess/nslt_2000.json'

    weights = '' # starts with no weights, but would use weights from /checkpoints if the training process is interrupted
    config_file = 'configfiles/asl2000.ini'

    configs = Config(config_file)
    print(root, train_split)
    run(configs=configs, mode=mode, root=root, save_model=save_model, train_split=train_split, weights=weights)
