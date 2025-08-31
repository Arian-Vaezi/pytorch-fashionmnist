import torch
from pathlib import Path
from models import MLP
from data import make_dataloaders
import numpy as np
from sklearn.metrics import classification_report 
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import os

def main():
    device=torch.device("cuda" if torch.cuda.is_available() else 'cpu')
    print("Device:", device)

    if not Path("runs/best.pt").exists():
        raise FileNotFoundError("runs/best.pt not found. Train first.")
    best=torch.load("runs/best.pt", map_location=device, weights_only=False)
    epoch=best.get('epoch')
    acc = best.get('best_acc')
    print(f"Epoch: {epoch}, accuracy: {acc:.4f}")


    model = MLP(hidden=256, dropout=0.1).to(device)
    model.load_state_dict(best['model_state'])
    model.eval()
    _, test_loader= make_dataloaders(batch_size=256, num_workers=0)
    print("Test loader ready, batches:", len(test_loader))

    all_y, all_p=[],[]
    with torch.no_grad():
        for x,y in test_loader:
            x,y=x.to(device), y.to(device)
            logits=model(x)
            preds=logits.argmax(dim=1)

            all_p.append(preds.cpu().numpy())
            all_y.append(y.cpu().numpy())
        y_true=np.concatenate(all_y)
        y_pred=np.concatenate(all_p)
        acc = (y_true == y_pred).mean()
        print(f"Overall accuracy: {acc*100:.2f}%")


        CLASSES = ["T-shirt/top","Trouser","Pullover","Dress","Coat",
           "Sandal","Shirt","Sneaker","Bag","Ankle boot"]
        print(classification_report(y_true, y_pred, target_names=CLASSES, digits=4))


        cm = confusion_matrix(y_true, y_pred, labels=list(range(10)), normalize="true")
        fig, ax=plt.subplots(figsize=(6,5))
        im=ax.imshow(cm, cmap="Blues", interpolation="nearest")
        fig.colorbar(im,ax=ax)


        ax.set(
            xticks=np.arange(10),
            yticks=np.arange(10),
            xticklabels=CLASSES,
            yticklabels=CLASSES,xlabel="Predicted", 
            ylabel="True",
            title="Fashion-MNIST Confusion Matrix (normalized)")
        plt.setp(ax.get_xticklabels(),rotation=45, ha='right')


        fmt='.2f'
        thresh = cm.max() / 2.0
        for i in range(cm.shape[0]):
            for j in range(cm.shape[1]):
                ax.text(
                    j, i, format(cm[i, j], fmt),
                    ha="center", va="center",
                    color="white" if cm[i, j] > thresh else "black",
                )

        os.makedirs("runs", exist_ok=True)
        plt.savefig("runs/confmat_mlp.png", dpi=150, bbox_inches="tight")
        plt.close(fig)
        print('Saved confustion matrix to runs/confmat_mlp.png')










    

if __name__ == "__main__":
    main()