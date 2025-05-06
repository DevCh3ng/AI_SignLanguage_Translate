import os
import torch
import torch.nn as nn
import dataset_helper.videotransforms as vt
import numpy as np

from pytorch_i3d import InceptionI3d
from dataset_helper.dataset_utils1 import Utils as Dataset
from torchvision import transforms

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = '0'
IMG_SIZE = 224
CLASSES = 400
DEFAULT_SEGMENT_LENGTH = 64
num_classes = 2000

def Setup(mode, weights_path, num_class, device):
    model = InceptionI3d(CLASSES, in_channels=3)
    model.replace_logits(num_class)
    model.load_state_dict(torch.load(weights_path, map_location=device))
    model = model.to(device)
    if torch.cuda.device_count() > 1:
        model = nn.DataParallel(model)
    model.eval()
    return model
def Eval(model, dataloader, num_class, device):
    correct_top1 = 0
    correct_top5 = 0
    correct_top10 = 0

    tp_top1 = np.zeros(num_class, dtype=np.int64)
    tp_top5 = np.zeros(num_class, dtype=np.int64)
    tp_top10 = np.zeros(num_class, dtype=np.int64)

    fp_top1 = np.zeros(num_class, dtype=np.int64)
    fp_top5 = np.zeros(num_class, dtype=np.int64)
    fp_top10 = np.zeros(num_class, dtype=np.int64)

    processed_samples = 0
    with torch.no_grad():
        for i, data in enumerate(dataloader):
            inputs, labels, video_ids = data
            inputs = inputs.to(device)
            labels = labels.to(device)
            batch_size, _, _, _, _ = inputs.shape

            per_frame_logits = model(inputs)
            predictions = torch.mean(per_frame_logits, dim=2)


            video_id = video_ids[0]
            label_item = labels[0].item()

            _, out_labels = torch.sort(predictions[0], descending=True)
            out_labels = out_labels.cpu().numpy()

            correct_top1 = (out_labels[0] == label_item)
            correct_top5 = label_item in out_labels[:5]
            correct_top10 = label_item in out_labels[:10]

            if correct_top1:
                correct_top1 += 1
                tp_top1[label_item] += 1
            else:
                fp_top1[label_item] += 1

            if correct_top5:
                correct_top5 += 1
                tp_top5[label_item] += 1
            else:
                fp_top5[label_item] += 1

            if correct_top10:
                correct_top10 += 1
                tp_top10[label_item] += 1
            else:
                fp_top10[label_item] += 1

            processed_samples += batch_size

            print(f"{video_id} {float(correct_top1) / processed_samples:.8f} {float(correct_top5) / processed_samples:.8f} {float(correct_top10) / processed_samples:.8f}")

    denom_1 = tp_top1 + fp_top1
    denom_5 = tp_top5 + fp_top5
    denom_10 = tp_top10 + fp_top10

    mca_top1 = np.mean(np.divide(tp_top1, denom_1, where=denom_1 > 0, out=np.zeros_like(tp_top1, dtype=float)))
    mca_top5 = np.mean(np.divide(tp_top5, denom_5, where=denom_5 > 0, out=np.zeros_like(tp_top5, dtype=float)))
    mca_top10 = np.mean(np.divide(tp_top10, denom_10, where=denom_10 > 0, out=np.zeros_like(tp_top10, dtype=float)))

    print(f'top-k average per class acc: {mca_top1:.8f}, {mca_top5:.8f}, {mca_top10:.8f}')

def run(mode='', root='', train_split='', weights=''):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    test_transforms = transforms.Compose([vt.CenterCrop(IMG_SIZE)])
    val_dataset = Dataset(split_file=train_split, split='test', root=root, mode=mode, transforms=test_transforms)

    val_dataloader = torch.utils.data.DataLoader(val_dataset, batch_size=1,
                                                 shuffle=False, num_workers=2,
                                                 pin_memory=False)

    model = Setup(mode=mode, weights_path=weights, num_class=num_classes, device=device)
    if model is None:
        print("Model setup failed.")
        return

    Eval(model=model,
        dataloader=val_dataloader,
        num_class=num_classes,
        device=device)

if __name__ == '__main__':
    mode = 'rgb'
    root = 'data/WLASL2000'
    train_split = 'configfiles/data_split.json'
    weights = 'weights/SLT_2000_0.7622.pt' #best 76.22%
    run(mode=mode, root=root, train_split=train_split, weights=weights)