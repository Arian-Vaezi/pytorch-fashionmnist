import argparse
import time
import torch
import torch.nn as nn
from torch.optim import Adam
from data import make_dataloaders
from models import MLP
from pathlib import Path

def get_args():
    p = argparse.ArgumentParser(description='Train Fashion-MNIST (MLP)')
    p.add_argument('--epochs', type=int, default=5,help='number of training epochs')
    p.add_argument("--batch_size", type=int, default=128, help="mini-batch size")
    p.add_argument("--lr", type=float, default=1e-3, help="learning rate")
    p.add_argument("--num_workers", type=int, default=0, help="DataLoader workers (0 on Windows)")
    p.add_argument("--cpu", action="store_true", help="force CPU even if CUDA is available")
    return p.parse_args()

def accuracy(logits, y):
    preds = logits.argmax(dim=1)
    return (preds == y).float().mean().item()

def train_one_epoch(model, loader, device, criterion, optimizer):
    model.train()
    losses, accs=[],[]
    for x,y in loader:
        x,y=x.to(device), y.to(device)

        logits=model(x)
        loss=criterion(logits,y)

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()

        losses.append(loss.item())
        accs.append(accuracy(logits, y))
    return sum(losses)/len(losses), sum(accs)/len(accs)  

@torch.no_grad()
def evaluate(model, loader, device, criterion):
    model.eval()
    losses, accs = [], []
    for x,y in loader:
        x,y=x.to(device), y.to(device)
        logits=model(x)
        loss=criterion(logits,y)

        losses.append(loss.item())
        accs.append(accuracy(logits,y))
    return sum(losses)/len(losses), sum(accs)/len(accs)


        


def main(args):
    Path("runs").mkdir(parents=True, exist_ok=True)
    device=torch.device("cuda" if torch.cuda.is_available() and not args.cpu else "cpu")
    print('Device:', device)
    print(f"epochs={args.epochs} batch_size={args.batch_size} lr={args.lr} workers={args.num_workers}")
    train_loader, test_loader=make_dataloaders(batch_size=args.batch_size, num_workers=args.num_workers)
    print(f"Train set size: {len(train_loader.dataset)}")
    print(f"Test set size: {len(test_loader.dataset)}")

    model =MLP(hidden=256,dropout=0.1).to(device)
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = Adam(model.parameters(), lr=args.lr)
    best_acc=0.0

    for epoch in range(1,args.epochs+1):
        train_loss,train_acc=train_one_epoch(model,train_loader, device, criterion, optimizer)
        test_loss,test_acc=evaluate(model,test_loader, device, criterion)

        if test_acc>best_acc:
            best_acc=test_acc
            checkpoint={
                'epoch':epoch,
                'model_state':model.state_dict(),
                "optimizer_state": optimizer.state_dict(),
                'best_acc':best_acc,
                "args":vars(args)
            }
            torch.save(checkpoint,'runs/best.pt')
            torch.save(checkpoint, f'runs/epoch_{epoch:02d}.pt')
        print(f"Epoch {epoch:02d} | tr_loss={train_loss:.4f} tr_acc={train_acc*100:.2f}% | test_loss={test_loss:.4f} test_acc={test_acc*100:.2f}% | best={best_acc*100:.2f}%")
        
    # run_dir = Path("runs") / time.strftime("%Y%m%d-%H%M%S")
    # run_dir.mkdir(parents=True, exist_ok=True)
    # torch.save(checkpoint, run_dir / "best.pt")

# FOR LOADING LATER  #data = torch.load("runs/best.pt", map_location=device)
# model.load_state_dict(data["model_state"])
# print("Best Test acc:", data["best_acc"])






if __name__ == "__main__":
    args=get_args()
    main(args)