import torch
from torch.utils.data import DataLoader
from torchvision import transforms
from PIL import Image


from .vqa_dataset import vqa_dataset
from .pretrain_dataset import pretrain_dataset
from .randaugment import RandomAugment


def create_dataset(dataset, config):
    # remote sensing image-text pretraining datasets
    if dataset == 'pretrain':
        normalize = transforms.Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711))
        pretrain_transform = transforms.Compose([
            transforms.Resize((config['image_res'], config['image_res']), interpolation=Image.BICUBIC),
            # transforms.RandomHorizontalFlip(),
            RandomAugment(2, 7, isPIL=True, augs=['Identity', 'AutoContrast', 'Equalize', 'Brightness', 'Sharpness']),
            # RandomAugment(2, 7, isPIL=True,
            #               augs=['Identity', 'Contrast', 'Equalize', 'Brightness', 'Sharpness', 'Color']),
            transforms.ToTensor(),
            normalize,
        ])
        dataset = pretrain_dataset(config['train_file'], pretrain_transform, image_root=config['image_root'], imgcap_root=config['imgcap_root'])
        return dataset

    elif dataset == 'VRS':
        normalize = transforms.Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711))
        train_transform = transforms.Compose([
            transforms.RandomResizedCrop(config['image_res'], scale=(0.9, 1.0), interpolation=Image.BICUBIC),
            # transforms.RandomHorizontalFlip(),
            RandomAugment(2, 7, isPIL=True, augs=['Identity','AutoContrast', 'Equalize', 'Brightness', 'Sharpness', 'Color','TranslateX','TranslateY']),
            transforms.ToTensor(),
            normalize,
        ])
        test_transform = transforms.Compose([
            transforms.Resize((config['image_res'], config['image_res']), interpolation=Image.BICUBIC),
            transforms.ToTensor(),
            normalize,
        ])
        train_dataset = vqa_dataset(config['VRS']['train_file'], train_transform, config['VRS']['vqa_root'], split='train', sample_fraction=1.0)
        test_dataset = vqa_dataset(config['VRS']['test_file'], test_transform, config['VRS']['vqa_root'], split='test',
                                   answer_list=config['VRS']['answer_list'])
        return train_dataset, test_dataset

    elif dataset == 'LR':
        normalize = transforms.Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711))
        train_transform = transforms.Compose([
            transforms.RandomResizedCrop(config['image_res'], scale=(0.8, 1.0), interpolation=Image.BICUBIC),
            transforms.RandomHorizontalFlip(),
            RandomAugment(2, 7, isPIL=True, augs=['Identity', 'AutoContrast', 'Equalize', 'Brightness', 'Sharpness','Rotate']),
            transforms.ToTensor(),
            normalize,
        ])
        test_transform = transforms.Compose([
            transforms.Resize((config['image_res'], config['image_res']), interpolation=Image.BICUBIC),
            transforms.ToTensor(),
            normalize,
        ])
        train_dataset = vqa_dataset(config['LR']['train_file'], train_transform, config['LR']['vqa_root'], split='train', sample_fraction=1.0)
        test_dataset = vqa_dataset(config['LR']['test_file'], test_transform, config['LR']['vqa_root'], split='test',
                                   answer_list=config['LR']['answer_list_test'])
        val_dataset = vqa_dataset(config['LR']['val_file'], test_transform, config['LR']['vqa_root'], split='test',
                                   answer_list=config['LR']['answer_list_val'])
        return train_dataset, test_dataset, val_dataset

    elif dataset == 'HR':
        normalize = transforms.Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711))
        train_transform = transforms.Compose([
            transforms.RandomResizedCrop(config['image_res'], scale=(0.8, 1.0), interpolation=Image.BICUBIC),
            transforms.RandomHorizontalFlip(),
            RandomAugment(2, 7, isPIL=True, augs=['Identity', 'AutoContrast', 'Equalize', 'Brightness', 'Sharpness','Rotate']),
            transforms.ToTensor(),
            normalize,
        ])
        test_transform = transforms.Compose([
            transforms.Resize((config['image_res'], config['image_res']), interpolation=Image.BICUBIC),
            transforms.ToTensor(),
            normalize,
        ])
        train_dataset = vqa_dataset(config['HR']['train_file'], train_transform, config['HR']['vqa_root'], split='train', sample_fraction=1.0)
        test1_dataset = vqa_dataset(config['HR']['test1_file'], test_transform, config['HR']['vqa_root'], split='test',
                                   answer_list=config['HR']['answer_list1'])
        test2_dataset = vqa_dataset(config['HR']['test2_file'], test_transform, config['HR']['vqa_root'], split='test',
                                   answer_list=config['HR']['answer_list2'])
        return train_dataset, test1_dataset, test2_dataset

    elif dataset == 'HRVQA':
        normalize = transforms.Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711))
        train_transform = transforms.Compose([
            transforms.RandomResizedCrop(config['image_res'], scale=(0.9, 1.0), interpolation=Image.BICUBIC),
            # transforms.RandomHorizontalFlip(),
            RandomAugment(2, 7, isPIL=True, augs=['Identity','AutoContrast', 'Equalize', 'Brightness', 'Sharpness', 'Color','TranslateX','TranslateY']),
            transforms.ToTensor(),
            normalize,
        ])
        test_transform = transforms.Compose([
            transforms.Resize((config['image_res'], config['image_res']), interpolation=Image.BICUBIC),
            transforms.ToTensor(),
            normalize,
        ])
        train_dataset = vqa_dataset(config['HRVQA']['train_file'], train_transform, config['HRVQA']['vqa_root'], split='train', sample_fraction=1.0)
        test_dataset = vqa_dataset(config['HRVQA']['test_file'], test_transform, config['HRVQA']['vqa_root'], split='test',
                                   answer_list=config['HRVQA']['answer_list'])
        return train_dataset, test_dataset



def vqa_collate_fn(batch):
    image_list, question_list, answer_list = [], [], []
    for image, question, answer in batch:
        image_list.append(image)
        question_list.append(question)
        answer_list += answer

    return torch.stack(image_list, dim=0), question_list, answer_list


def create_sampler(datasets, shuffles, num_tasks, global_rank):
    samplers = []
    for dataset, shuffle in zip(datasets, shuffles):
        sampler = torch.utils.data.DistributedSampler(dataset, num_replicas=num_tasks, rank=global_rank,
                                                      shuffle=shuffle)
        samplers.append(sampler)
    return samplers


def create_loader(datasets, samplers, batch_size, num_workers, is_trains, collate_fns):
    loaders = []
    for dataset, sampler, bs, n_worker, is_train, collate_fn in zip(datasets, samplers, batch_size, num_workers,
                                                                    is_trains, collate_fns):
        if is_train:
            shuffle = (sampler is None)
            drop_last = True
        else:
            shuffle = False
            drop_last = False
        loader = DataLoader(
            dataset,
            batch_size=bs,
            num_workers=n_worker,
            pin_memory=True,
            sampler=sampler,
            shuffle=shuffle,
            collate_fn=collate_fn,
            drop_last=drop_last,
        )
        loaders.append(loader)
    return loaders

def pretrain_collate_fn(batch):
    image_list, caption_list, box_list, region_text_list = [], [], [], []
    for image, caption, region_boxes, region_texts in batch:
        image_list.append(image)
        caption_list.append(caption)
        box_list.append(region_boxes)
        region_text_list.append(region_texts)

    return torch.stack(image_list, dim=0), caption_list, box_list, region_text_list
