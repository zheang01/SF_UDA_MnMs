import os
import glob
import numpy as np
from tqdm import tqdm

from utils import read_list, read_nifti, config


base_dir = config.base_dir


def write_txt(data, path):
    with open(path, 'w') as f:
        for val in data:
            f.writelines(val + '\n')


def process_npy():
    for tag in ['training','validation','testing']:
        img_ids = []
        for path in tqdm(glob.glob(os.path.join(base_dir, 'B', tag, 'data','*.nii.gz'))):
            img_id = path.split('/')[-1].split('_')[0]
            img_ids.append(img_id)

            image_path = os.path.join(base_dir, 'B', tag, 'data', f'{img_id}_0000.nii.gz')
            label_path = os.path.join(base_dir, 'B', tag, 'seg', f'{img_id}.nii.gz')

            image = read_nifti(image_path)
            label = read_nifti(label_path)

            # <-- 1. [optional]
            image = (image - image.mean()) / (image.std() + 1e-8)
            image = image.astype(np.float32)

            # <-- 2.
            np.save(
                os.path.join(base_dir, 'npy', f'{img_id}_image.npy'),
                image
            )
            np.save(
                os.path.join(base_dir, 'npy', f'{img_id}_label.npy'),
                label
            )


def process_split_fully(train_ratio=0.8):
    for tag in ['training','validation','testing']:
        img_ids = []
        for path in tqdm(glob.glob(os.path.join(base_dir, 'B', tag, 'data', '*.nii.gz'))):
            img_id = path.split('/')[-1].split('_')[0]
            img_ids.append(img_id)
        '''
        if tag == 'Tr':
            img_ids = np.random.permutation(img_ids)
            split_idx = int(len(img_ids) * train_ratio)
            train_ids = sorted(img_ids[:split_idx])
            eval_ids = sorted(img_ids[split_idx:])
            write_txt(
                train_ids,
                os.path.join(base_dir, 'splits/train.txt')
            )
            write_txt(
                eval_ids,
                os.path.join(base_dir, 'splits/eval.txt')
            )
        '''
        
        img_ids = sorted(img_ids)
        rename={'training':'train','validation':'eval','testing':'test'}
        write_txt(
            img_ids,
            os.path.join(base_dir, 'splits/'+rename[tag]+'.txt')
        )


def process_split_semi(split='train', labeled_ratio=0.05):
    ids_list = read_list(split)
    ids_list = np.random.permutation(ids_list)

    split_idx = int(len(ids_list) * labeled_ratio)
    labeled_ids = sorted(ids_list[:split_idx])
    unlabeled_ids = sorted(ids_list[split_idx:])
    
    write_txt(
        labeled_ids,
        os.path.join(base_dir, 'splits/labeled.txt')
    )
    write_txt(
        unlabeled_ids,
        os.path.join(base_dir, 'splits/unlabeled.txt')
    )


if __name__ == '__main__':
    process_npy()
    process_split_fully()
    # process_split_semi()
