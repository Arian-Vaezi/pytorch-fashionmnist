from src.data import make_dataloaders
import torch

def main():
    train_loader, test_loader = make_dataloaders(batch_size=64, num_workers=4)

    x, y = next(iter(train_loader))

    print("x shape:", x.shape)          
    print("y shape:", y.shape)          
    print("x min/max:", float(x.min()), float(x.max()))
    print("labels sample:", y[:10].tolist())
    print("dtype:", x.dtype, "device:", x.device)
if __name__ == "__main__":


    main()