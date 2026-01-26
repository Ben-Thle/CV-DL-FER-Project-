from torch.utils.data import DataLoader
from data_prep import train_dataset

dl = DataLoader(train_dataset, batch_size=4, shuffle=True)
x, y = next(iter(dl))

print("x shape:", x.shape)
print("y shape:", y.shape)
