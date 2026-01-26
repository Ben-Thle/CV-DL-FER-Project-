import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import numpy as np
from pathlib import Path

from src.models.model import build_model
from src.data_prep import train_dataset, val_dataset, train_transform, test_transform
from src.evaluation import calculate_macro_f1_score, print_evaluation_summary

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Check if datasets have data
if len(train_dataset) == 0:
    raise ValueError("No training data found! Make sure the pipeline has been run.")
if len(val_dataset) == 0:
    raise ValueError("No validation data found! Make sure the pipeline has been run.")

# Get number of classes and class names
num_classes = len(train_dataset.class_names)
class_names = train_dataset.class_names
print(f"Number of classes: {num_classes}")
print(f"Classes: {class_names}")
print(f"Train samples: {len(train_dataset)}")
print(f"Validation samples: {len(val_dataset)}")

# Build model 
model = build_model("resnet18", num_classes=num_classes, input_channels=1, small_input=True)
model = model.to(device)

criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True, num_workers=2)
val_loader = DataLoader(val_dataset, batch_size=64, shuffle=False, num_workers=2)

print("\nStarting training...\n")

# Create save 
save_dir = Path('experiments')
save_dir.mkdir(exist_ok=True)
checkpoints_dir = save_dir / 'checkpoints'
checkpoints_dir.mkdir(exist_ok=True)

num_epochs = 100  # Maximum epochs 
best_f1 = 0.0
best_epoch = 0
patience = 10  # Stop if no improvement for this many epochs
patience_counter = 0

for epoch in range(num_epochs):
    
    model.train()
    total_loss, correct, total = 0.0, 0, 0
    train_preds = []
    train_labels = []

    for x, y in train_loader:
        x, y = x.to(device), y.to(device)

        optimizer.zero_grad()
        logits = model(x)
        loss = criterion(logits, y)
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * x.size(0)
        pred = logits.argmax(dim=1)
        correct += (pred == y).sum().item()
        total += y.size(0)
        
        train_preds.extend(pred.cpu().numpy())
        train_labels.extend(y.cpu().numpy())

    train_loss = total_loss / total
    train_acc = correct / total
    train_f1 = calculate_macro_f1_score(train_labels, train_preds)

    # Validation 
    model.eval()
    val_loss, val_correct, val_total = 0.0, 0, 0
    val_preds = []
    val_labels = []

    with torch.no_grad():
        for x, y in val_loader:
            x, y = x.to(device), y.to(device)
            logits = model(x)
            loss = criterion(logits, y)

            val_loss += loss.item() * x.size(0)
            pred = logits.argmax(dim=1)
            val_correct += (pred == y).sum().item()
            val_total += y.size(0)
            
            val_preds.extend(pred.cpu().numpy())
            val_labels.extend(y.cpu().numpy())

    val_loss = val_loss / val_total
    val_acc = val_correct / val_total
    val_f1 = calculate_macro_f1_score(val_labels, val_preds)

    print(f"\nEpoch {epoch+1}/{num_epochs}:")
    print(f"Train - Loss: {train_loss:.4f}, Acc: {train_acc:.4f}, Macro F1: {train_f1:.4f}")
    print(f"Val   - Loss: {val_loss:.4f}, Acc: {val_acc:.4f}, Macro F1: {val_f1:.4f}")

    # Print detailed evaluation
    print_detailed_summary = (epoch + 1) % 5 == 0 or epoch == 0 or epoch == num_epochs - 1
    
    if print_detailed_summary:
        print("\n" + "="*60)
        print(f"Training Set Evaluation - Epoch {epoch+1}")
        print("="*60)
        print_evaluation_summary(
            y_true=np.array(train_labels),
            y_pred=np.array(train_preds),
            class_names=class_names
        )
        
        print("\n" + "="*60)
        print(f"Validation Set Evaluation - Epoch {epoch+1}")
        print("="*60)
        print_evaluation_summary(
            y_true=np.array(val_labels),
            y_pred=np.array(val_preds),
            class_names=class_names
        )
        print()
    else:
        print() 

    # Save checkpoint
    checkpoint = {
        'epoch': epoch + 1,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'train_loss': train_loss,
        'train_acc': train_acc,
        'train_f1': train_f1,
        'val_loss': val_loss,
        'val_acc': val_acc,
        'val_f1': val_f1,
        'num_classes': num_classes,
        'class_names': class_names,
    }
    
    
    checkpoint_path = checkpoints_dir / f'checkpoint_epoch_{epoch+1}.pt'
    torch.save(checkpoint, checkpoint_path)
    
    # Save best model separately if this is the best F1 score
    if val_f1 > best_f1:
        best_f1 = val_f1
        best_epoch = epoch + 1
        patience_counter = 0 
        best_checkpoint_path = save_dir / 'best_model.pt'
        torch.save(checkpoint, best_checkpoint_path)
        print(f"  → Saved best model (Macro F1: {best_f1:.4f}) to {best_checkpoint_path}")
    else:
        patience_counter += 1
        print(f"  → Saved checkpoint to {checkpoint_path}")
    
    # Early stopping: stop if no improvement for 'patience' epochs
    if patience_counter >= patience:
        print(f"\nEarly stopping triggered! No improvement for {patience} epochs.")
        print(f"Best validation Macro F1 was {best_f1:.4f} at epoch {best_epoch}")
        break

print(f"\nTraining complete!")
print(f"Best Validation Macro F1: {best_f1:.4f}")
print(f"Best model saved to: {save_dir / 'best_model.pt'}")
print(f"All checkpoints saved to: {checkpoints_dir}")
