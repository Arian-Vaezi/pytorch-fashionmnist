from torchvision import datasets, transforms
from torch.utils.data import DataLoader

MEAN, STD = (0.2860,), (0.3530,)

def make_dataloaders(batch_size=128, num_workers=1):
    tf_train=transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(MEAN,STD)
    ])

    tf_test = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(MEAN, STD),
        ])
    train_ds= datasets.FashionMNIST(root='./data',train=True, download=True, transform=tf_train)
    test_ds= datasets.FashionMNIST(root='./data',train=False, download=True, transform=tf_test)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,     num_workers=num_workers)
    test_loader=DataLoader(test_ds,batch_size=batch_size, shuffle=True, num_workers=num_workers)
    return train_loader, test_loader
    
