import os
import sys
sys.path.append('cocoapi/PythonAPI')
from pycocotools.coco import COCO
import nltk
nltk.data.path.append('nltk_data/tokenizers')
from nltk.tokenize import word_tokenize
import json
from PIL import Image
import numpy as np
import torch
import torch.utils.data as data
from torchvision import transforms
from captioning.vocabulary import Vocabulary
from tqdm import tqdm


def get_loader(transform,
               mode='train',
               batch_size=1,
               vocab_threshold=None,
               vocab_file='vocab.pkl',
               start_word="<start>",
               end_word="<end>",
               unk_word="<unk>",
               vocab_from_file=True,
               num_workers=0):
    """Returns the data loader for COCO captions.
    Args:
        transform: torchvision transforms for preprocessing images.
        mode: 'train' or 'test'.
        batch_size: batch size (must be 1 for test mode).
        vocab_threshold: minimum word frequency to include in vocab.
        vocab_file: path to save or load the vocabulary pickle.
        start_word, end_word, unk_word: special tokens.
        vocab_from_file: if False, build vocab anew (train mode only).
        num_workers: number of data loader workers.
    Returns:
        torch.utils.data.DataLoader
    """
    assert mode in ['train', 'test'], "mode must be 'train' or 'test'"
    if not vocab_from_file:
        assert mode == 'train', "To build vocab, must be in training mode."

    # Set paths based on mode
    if mode == 'train':
        img_folder = os.path.join('cocoapi', 'train2014')
        annotations_file = os.path.join('cocoapi', 'annotations', 'captions_train2014.json')
        if vocab_from_file:
            assert os.path.exists(vocab_file), f"vocab file {vocab_file} not found."
    else:
        assert batch_size == 1, "batch_size must be 1 in test mode"
        img_folder = os.path.join('cocoapi', 'test2014')
        annotations_file = os.path.join('cocoapi', 'annotations', 'image_info_test2014.json')
        assert os.path.exists(vocab_file), f"vocab file {vocab_file} not found."
        assert vocab_from_file, "vocab_from_file must be True in test mode"

    # Create dataset
    dataset = CoCoDataset(
        transform=transform,
        mode=mode,
        batch_size=batch_size,
        vocab_threshold=vocab_threshold,
        vocab_file=vocab_file,
        start_word=start_word,
        end_word=end_word,
        unk_word=unk_word,
        annotations_file=annotations_file,
        vocab_from_file=vocab_from_file,
        img_folder=img_folder
    )

    # Create sampler and DataLoader
    if mode == 'train':
        indices = dataset.get_train_indices()
        sampler = data.sampler.SubsetRandomSampler(indices)
        batch_sampler = data.sampler.BatchSampler(sampler, batch_size=dataset.batch_size, drop_last=False)
        loader = data.DataLoader(dataset, batch_sampler=batch_sampler, num_workers=num_workers)
    else:
        loader = data.DataLoader(dataset, batch_size=dataset.batch_size, shuffle=True, num_workers=num_workers)

    return loader


class CoCoDataset(data.Dataset):
    def __init__(self, transform, mode, batch_size, vocab_threshold,
                 vocab_file, start_word, end_word, unk_word,
                 annotations_file, vocab_from_file, img_folder):
        self.transform = transform
        self.mode = mode
        self.batch_size = batch_size
        # build or load vocabulary
        self.vocab = Vocabulary(vocab_threshold, vocab_file, start_word,
                                 end_word, unk_word, annotations_file,
                                 vocab_from_file)
        self.img_folder = img_folder

        if mode == 'train':
            self.coco = COCO(annotations_file)
            self.ids = list(self.coco.anns.keys())
            # precompute caption lengths for sampler
            print('Obtaining caption lengths...')
            captions = [self.coco.anns[i]['caption'] for i in self.ids]
            tokens = [nltk.tokenize.word_tokenize(c.lower()) for c in tqdm(captions)]
            self.caption_lengths = [len(t) for t in tokens]
        else:
            info = json.load(open(annotations_file))
            self.paths = [img['file_name'] for img in info['images']]

    def __getitem__(self, index):
        if self.mode == 'train':
            ann_id = self.ids[index]
            caption = self.coco.anns[ann_id]['caption']
            img_id = self.coco.anns[ann_id]['image_id']
            path = self.coco.loadImgs(img_id)[0]['file_name']
            image = Image.open(os.path.join(self.img_folder, path)).convert('RGB')
            image = self.transform(image)

            tokens = nltk.tokenize.word_tokenize(caption.lower())
            seq = [self.vocab(self.vocab.start_word)] + [self.vocab(t) for t in tokens] + [self.vocab(self.vocab.end_word)]
            caption_tensor = torch.tensor(seq, dtype=torch.long)
            return image, caption_tensor
        else:
            path = self.paths[index]
            pil = Image.open(os.path.join(self.img_folder, path)).convert('RGB')
            orig = np.array(pil)
            img = self.transform(pil)
            return orig, img

    def __len__(self):
        return len(self.ids) if self.mode=='train' else len(self.paths)

    def get_train_indices(self):
        sel_len = np.random.choice(self.caption_lengths)
        idxs = np.where([l==sel_len for l in self.caption_lengths])[0]
        return list(np.random.choice(idxs, size=self.batch_size, replace=True))
