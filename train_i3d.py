import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import dataset_helper.videotransforms as vt
import numpy as np

from configfiles.configs import Config
from pytorch_i3d import InceptionI3d
from dataset_helper.dataset_utils import Utils as Dataset
from torchvision import transforms

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = '0'

torch.manual_seed(0)
np.random.seed(0)

torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

IMG_SIZE = 224
CLASSES = 400
mode = 'rgb'
root = {'word': 'data/WLASL2000'}
save_model = 'checkpoints/'
train_split = 'configfiles/data_split.json' 
weights = 'weights/nslt_2000_002276_0.469771.pt' # starts with no weights, but would use weights from /checkpoints if training process is interrupted
config_file = 'configfiles/conf.ini'
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
configs = Config(config_file)

os.makedirs(save_model, exist_ok=True)
print(root, train_split)
print(configs)

train_transforms = transforms.Compose([
    vt.RandomCrop(IMG_SIZE),
    vt.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1)
])
test_transforms = transforms.Compose([vt.CenterCrop(IMG_SIZE)])

dataset = Dataset(train_split, 'train', root, mode, train_transforms)
val_dataset = Dataset(train_split, 'test', root, mode, test_transforms)

dataloader = torch.utils.data.DataLoader(dataset,
                                        batch_size=configs.batch_size,
                                        shuffle=True,
                                        num_workers=0,
                                        pin_memory=True)

val_dataloader = torch.utils.data.DataLoader(val_dataset,
                                            batch_size=configs.batch_size,
                                            shuffle=True,
                                            num_workers=2,
                                            pin_memory=False)

dataloaders = {'train': dataloader, 'test': val_dataloader}
datasets = {'train': dataset, 'test': val_dataset}
num_classes = dataset.num_classes
i3d = InceptionI3d(CLASSES, in_channels=3)
i3d.load_state_dict(torch.load('weights/rgb_imagenet.pt', map_location=device))
i3d.replace_logits(num_classes)

if weights:
    print('Loading weights {}'.format(weights))
    i3d.load_state_dict(torch.load(weights, map_location=device))

i3d.to(device)
i3d = nn.DataParallel(i3d)

lr = configs.init_lr
weight_decay = configs.adam_weight_decay
optimizer = optim.Adam(i3d.parameters(), lr=lr, weight_decay=weight_decay)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', patience=5, factor=0.3, verbose=True)

num_steps_per_update = configs.update_per_step
steps = 0
epoch = 0
max_epochs = 400

best_val_score = 0.0
best_val_loss = float('inf')
early_stopping_patience = 10
patience_counter = 0

loss_hist = []
train_acc_history = []
val_loss_history = []
val_acc_history = []

while steps < configs.max_steps and epoch < max_epochs:
    print('Step {}/{}'.format(steps, configs.max_steps))
    print('-' * 10)

    epoch += 1
    for phase in ['train', 'test']:

        if phase == 'train':
            i3d.train(True)
        else:
            i3d.train(False)

        tot_loss_accum = 0.0
        tot_loc_loss_accum = 0.0
        tot_cls_loss_accum = 0.0
        num_iter = 0

        running_loc_loss = 0.0
        running_cls_loss = 0.0
        running_corrects = 0
        total_samples_in_epoch = 0

        optimizer.zero_grad()
        confusion_matrix = np.zeros((num_classes, num_classes), dtype=np.int64)

        for data in dataloaders[phase]:
            num_iter += 1
            if data == -1:
                continue

            inputs, labels, vid = data

            inputs = inputs.to(device, non_blocking=True)
            t = inputs.size(2)
            labels = labels.to(device, non_blocking=True)
            current_batch_size = inputs.shape[0]
            total_samples_in_epoch += current_batch_size

            with torch.set_grad_enabled(phase == 'train'):
                per_frame_logits = i3d(inputs, pretrained=False)
                per_frame_logits = F.interpolate(per_frame_logits, t, mode='linear', align_corners=False)

                loc_loss = F.binary_cross_entropy_with_logits(per_frame_logits, labels)
                running_loc_loss += loc_loss.item() * current_batch_size

                max_logits = torch.max(per_frame_logits, dim=2)[0]
                max_labels = torch.max(labels, dim=2)[0]

                cls_loss = F.binary_cross_entropy_with_logits(max_logits, max_labels)
                running_cls_loss += cls_loss.item() * current_batch_size

                pred_indices_cm = torch.argmax(max_logits, dim=1)
                true_indices_cm = torch.argmax(max_labels, dim=1)
                for i in range(current_batch_size):
                        confusion_matrix[true_indices_cm[i].item(), pred_indices_cm[i].item()] += 1
                running_corrects += torch.sum(pred_indices_cm == true_indices_cm).item()

                loss = (0.5 * loc_loss + 0.5 * cls_loss)

                if phase == 'train':
                    scaled_loss = loss / num_steps_per_update
                    scaled_loss.backward()

                    tot_loss_accum += loss.item() / num_steps_per_update
                    tot_loc_loss_accum += loc_loss.item()
                    tot_cls_loss_accum += cls_loss.item()

                    if num_iter == num_steps_per_update:
                        steps += 1
                        optimizer.step()
                        optimizer.zero_grad()

                        if steps % 10 == 0:
                            current_cm_sum = np.sum(confusion_matrix)
                            acc = float(np.trace(confusion_matrix)) / current_cm_sum if current_cm_sum > 0 else 0.0
                            print(f"Epoch {epoch}, Step {steps}: Loc Loss: {tot_loc_loss_accum / num_steps_per_update:.4f} "
                                f"Cls Loss: {tot_cls_loss_accum / num_steps_per_update:.4f} "
                                f"Tot Loss: {tot_loss_accum:.4f} Accu: {acc:.4f}", flush=True)
                            tot_loss_accum = tot_loc_loss_accum = tot_cls_loss_accum = 0.
                        num_iter = 0

        epoch_loss_val = (running_loc_loss + running_cls_loss) * 0.5 / total_samples_in_epoch if total_samples_in_epoch > 0 else 0
        epoch_loc_loss_val = running_loc_loss / total_samples_in_epoch if total_samples_in_epoch > 0 else 0
        epoch_cls_loss_val = running_cls_loss / total_samples_in_epoch if total_samples_in_epoch > 0 else 0
        epoch_acc_val = float(running_corrects) / total_samples_in_epoch if total_samples_in_epoch > 0 else 0.0
        epoch_cm_sum = np.sum(confusion_matrix)
        epoch_acc_cm_val = float(np.trace(confusion_matrix)) / epoch_cm_sum if epoch_cm_sum > 0 else 0.0

        if phase == 'train':
            loss_hist.append(epoch_loss_val)
            train_acc_history.append(epoch_acc_val)

        if phase == 'test':
            val_score = epoch_acc_cm_val
            val_loss = epoch_loss_val
            val_loss_history.append(val_loss)
            val_acc_history.append(val_score)

            print(f"VALIDATION - Epoch {epoch}: Loc Loss: {epoch_loc_loss_val:.4f} "
                    f"Cls Loss: {epoch_cls_loss_val:.4f} "
                    f"Tot Loss: {val_loss:.4f} "
                    f"Accuracy: {val_score:.4f}", flush=True)

            scheduler.step(val_loss)

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0

                if val_score > best_val_score or epoch % 2 == 0:
                        if val_score > best_val_score:
                            best_val_score = val_score
                        model_name = os.path.join(save_model, f"nslt_{str(num_classes)}_{str(steps).zfill(6)}_{val_score:.4f}.pt")
                        torch.save(i3d.module.state_dict() if isinstance(i3d, nn.DataParallel) else i3d.state_dict(), model_name)
                        print(f"Model saved: {model_name}")
            else:
                patience_counter += 1
                if patience_counter >= early_stopping_patience:
                    print("Early stopping triggered!")
                    steps = configs.max_steps
                    break

    if steps >= configs.max_steps:
        break

print("Training finished or stopped.")