import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import numpy as np
from pathlib import Path
from torch.cuda.amp import autocast, GradScaler
import time
import traceback

from src.models.model import build_model
from src.data_prep import train_dataset, val_dataset, train_transform, test_transform
from src.evaluation import calculate_macro_f1_score, print_evaluation_summary


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Enable cuDNN benchmark for faster training on fixed-size inputs
    if device.type == "cuda":
        torch.backends.cudnn.benchmark = True

    # AMP scaler (only used when CUDA is available)
    scaler = GradScaler() if device.type == "cuda" else None


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

    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=50)

    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True, num_workers=4, pin_memory=True) 
    val_loader = DataLoader(val_dataset, batch_size=64, shuffle=False, num_workers=4, pin_memory=True) #64 #4

    print("\nStarting training...\n")

    # Create save 
    save_dir = Path('experiments')
    save_dir.mkdir(exist_ok=True)
    checkpoints_dir = save_dir / 'checkpoints'
    checkpoints_dir.mkdir(exist_ok=True)

    num_epochs = 100  # Maximum epochs  #100
    best_f1 = 0.0
    best_epoch = 0
    patience = 10  # Stop if no improvement for this many epochs
    patience_counter = 0

    for epoch in range(num_epochs):
        
        model.train()
        total_loss, correct, total = 0.0, 0, 0
        train_preds = []
        train_labels = []

        for i, (x,y) in enumerate(train_loader):#
            if i % 20 == 0:#
                print(f"Epoch {epoch+1} batch {i}/{len(train_loader)}")#

        batch_times = []
        for i, (x, y) in enumerate(train_loader, 1):
            start_batch = time.time()
            if i % 100 == 0 or i > len(train_loader) - 5:
                print(f"Epoch {epoch+1} - processing batch {i}/{len(train_loader)} (time: {time.strftime('%H:%M:%S')})", flush=True)

            try:
                t0 = time.time()
                x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
                t_after_move = time.time()

                optimizer.zero_grad(set_to_none=True)
                if scaler is not None:
                    with autocast():
                        logits = model(x)
                        loss = criterion(logits, y)
                    t_after_forward = time.time()
                    scaler.scale(loss).backward()
                    t_after_backward = time.time()
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    logits = model(x)
                    t_after_forward = time.time()
                    loss = criterion(logits, y)
                    loss.backward()
                    t_after_backward = time.time()
                    optimizer.step()

                batch_time = time.time() - start_batch
                batch_times.append(batch_time)

                # if a batch takes longer than 10s, print detailed timing info and inspect
                if batch_time > 10.0:
                    print(f"Long batch detected: batch {i}/{len(train_loader)} took {batch_time:.1f}s", flush=True)
                    print(f"  move: {t_after_move - t0:.3f}s, forward: {t_after_forward - t_after_move:.3f}s, backward: {t_after_backward - t_after_forward:.3f}s", flush=True)

                    # try to inspect individual samples if possible
                    try:
                        dataset = train_loader.dataset
                        if hasattr(dataset, 'samples'):
                            start_idx = (i - 1) * train_loader.batch_size
                            end_idx = start_idx + x.size(0)
                            print(f"  dataset sample paths (first 3): {dataset.samples[start_idx:min(end_idx, start_idx+3)]}", flush=True)
                    except Exception:
                        print("  Could not inspect dataset samples", flush=True)

                total_loss += loss.item() * x.size(0)
                pred = logits.argmax(dim=1)
                correct += (pred == y).sum().item()
                total += y.size(0)

                train_preds.extend(pred.cpu().numpy())
                train_labels.extend(y.cpu().numpy())

            except Exception as e:
                print(f"Exception during training batch {i}: {e}", flush=True)
                traceback.print_exc()
                # continue so the loop doesn't stop entirely
                continue

        if len(batch_times) > 0:
            avg_batch = sum(batch_times) / len(batch_times)
            print(f"Average batch time this epoch: {avg_batch:.3f}s (est epoch: {avg_batch * len(train_loader) / 60:.1f} min)", flush=True)

        train_loss = total_loss / total
        train_acc = correct / total
        train_f1 = calculate_macro_f1_score(train_labels, train_preds)

        # Validation 
        model.eval()
        val_loss, val_correct, val_total = 0.0, 0, 0
        val_preds = []
        val_labels = []

        with torch.no_grad():
            val_batch_times = []
            for i, (x, y) in enumerate(val_loader, 1):
                t0 = time.time()
                x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
                if scaler is not None:
                    with autocast():
                        logits = model(x)
                        loss = criterion(logits, y)
                else:
                    logits = model(x)
                    loss = criterion(logits, y)

                val_batch_times.append(time.time() - t0)

                val_loss += loss.item() * x.size(0)
                pred = logits.argmax(dim=1)
                val_correct += (pred == y).sum().item()
                val_total += y.size(0)
                
                val_preds.extend(pred.cpu().numpy())
                val_labels.extend(y.cpu().numpy())

            if len(val_batch_times) > 0:
                avg_val_batch = sum(val_batch_times) / len(val_batch_times)
                print(f"Avg val batch time: {avg_val_batch:.3f}s", flush=True)

        val_loss = val_loss / val_total
        val_acc = val_correct / val_total
        val_f1 = calculate_macro_f1_score(val_labels, val_preds)

        # Step LR scheduler (per epoch)
        if scheduler is not None:
            scheduler.step()

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


if __name__ == '__main__':
    from multiprocessing import freeze_support
    freeze_support()
    main()