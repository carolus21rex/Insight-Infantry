from torch.utils.data import DataLoader


def collate_fn(batch):
    images, targets = zip(*batch)  # separates images and targets
    images = list(images)
    targets = list(targets)
    return images, targets


def get_dataloader(dataset, batch_size, num_workers, shuffle):
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers,
                                       collate_fn=collate_fn)
