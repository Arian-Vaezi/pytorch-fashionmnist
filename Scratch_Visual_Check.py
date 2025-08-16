import matplotlib.pyplot as plt
from src.data import make_dataloaders

train_loader,_=make_dataloaders(batch_size=16, num_workers=0)
images, labels=next(iter(train_loader))

class_names=["T-shirt/top", "Trouser", "Pullover", "Dress", "Coat", 
               "Sandal", "Shirt", "Sneaker", "Bag", "Ankle boot"]

mean, std= 0.2860,0.3530
images=images*std+mean #unnormalize

fig,axes=plt.subplots(4,4,figsize=(6,6))
for i, ax in enumerate(axes.flat):
    img=images[i].squeeze(0)
    ax.imshow(img,cmap="gray")
    ax.set_title(class_names[labels[i]])
    ax.axis('off')

plt.tight_layout()
plt.savefig('runs/samples.png')
plt.show()


