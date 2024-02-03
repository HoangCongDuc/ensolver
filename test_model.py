import argparse
import os
import numpy as np
from tqdm import tqdm
import mmcv
from mmdet.apis import inference_detector, init_detector
from torch.utils.data import Dataset, DataLoader


def parse_args():
    parser = argparse.ArgumentParser("Run model on all images from a folder")
    parser.add_argument('img_dir')
    parser.add_argument('config_path')
    parser.add_argument('--ckpt-path')
    parser.add_argument('--out-path', default='out.txt')
    parser.add_argument('--batch-size', type=int, default=1)
    args = parser.parse_args()
    return args

def decode(result):
    cls = []
    for i, res in enumerate(result):
        cls.extend([i] * res.shape[0])

    result_arr = np.concatenate(result)

    boxes, indices = mmcv.ops.nms(result_arr[:,:4], result_arr[:, -1], 0.5)
    centers = (boxes[:, 0] + boxes[:, 2]) / 2
    permute = np.argsort(centers)
    boxes = boxes[permute]
    indices = indices[permute]

    highscore = boxes[:, -1] > 0.1
    boxes = boxes[highscore]
    indices = indices[highscore]

    dictionary = 'abcdefghijklmnopqrstuvwxyz0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ'
    texts = ''.join(dictionary[cls[i]] for i in indices)
    return boxes, texts

def get_score(boxes):
    if boxes.size == 0:
        return 0
    else:
        return boxes[:,-1].min()

def predict(model, dataloader):
    results = []
    with tqdm(total=len(dataloader.dataset), unit='samples') as pbar:
        for batch in dataloader:
            imgs, labels = zip(*batch)
            outputs = inference_detector(model, imgs)
            for output, label in zip(outputs, labels):
                bboxes, text = decode(output)
                score = get_score(bboxes)
                results.append((label, text, score))
                pbar.update()
    return results
    
class ImageDataset(Dataset):
    def __init__(self, img_dir):
        super().__init__()
        self.img_dir = img_dir
        self.img_files = os.listdir(img_dir)

    def __len__(self):
        return len(self.img_files)

    def __getitem__(self, index):
        path = os.path.join(self.img_dir, self.img_files[index])
        img = mmcv.imread(path)
        label = self.img_files[index].split('.')[0]
        return img, label

if __name__ == '__main__':
    args = parse_args()
    model = init_detector(args.config_path, args.ckpt_path, device='cuda')
    dataset = ImageDataset(args.img_dir)
    dataloader = DataLoader(dataset, batch_size=args.batch_size, collate_fn=lambda b: b, num_workers=1)
    
    results = predict(model, dataloader)
    correct = 0
    with open(args.out_path, 'w') as f:
        for label, pred, _ in results:
            if label != pred:
                f.write(f"{label} {pred}\n")
            else:
                correct += 1
    print("Accuracy: {:.1%}".format(correct / len(results)))