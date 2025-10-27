
from args import get_args
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import balanced_accuracy_score, roc_auc_score, average_precision_score
import numpy as np
import torch
import os
import matplotlib.pyplot as plt
import torch.nn.functional as F


def train_model(model, train_loader, val_loader, fold=0):
    args = get_args()

    # Collect all unique labels from the dataset
    all_classes = sorted(set(int(item['label']) for item in train_loader.dataset))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    #Defining the loss function and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=args.lr)

    best_balanced_acc = 0.0


        # For plotting
    train_losses, val_losses = [], []
    train_bal_accs, val_bal_accs = [], []
    train_roc_aucs, val_roc_aucs = [], []
    train_avg_precs, val_avg_precs = [], []

    for epoch in range(args.epochs):
        training_loss = 0.0
        # starting the training  -> setting the model to training mode
        model.train()
        
        all_train_targets = []
        all_train_outputs = []

        for batch in train_loader:
            inputs = batch['img'].to(device)
            targets = batch['label'].to(device)

            #Resetting the gradients
            optimizer.zero_grad()
            outputs = model(inputs)

            loss = criterion(outputs, targets.long())
            loss.backward()
            optimizer.step()

            training_loss += loss.item()

            # Collect predictions and labels for the full epoch
            all_train_targets.extend(targets.cpu().numpy())
            all_train_outputs.extend(F.softmax(outputs, dim=1).detach().cpu().numpy())


        #convert to array once per epoch
        all_train_targets = np.array(all_train_targets)
        all_train_outputs = np.array(all_train_outputs)

        train_loss = training_loss / len(train_loader)
        train_losses.append(train_loss)

        train_preds = np.argmax(all_train_outputs, axis=1)

        # Compute metrics
        train_bal_acc = balanced_accuracy_score(all_train_targets, train_preds)
        train_bal_accs.append(train_bal_acc)

        try:
            train_roc_auc = roc_auc_score(all_train_targets, all_train_outputs, multi_class='ovr')
        except ValueError:
            train_roc_auc = float('nan')
        train_roc_aucs.append(train_roc_auc)

        train_avg_prec = average_precision_score(all_train_targets, all_train_outputs, average='macro')
        train_avg_precs.append(train_avg_prec)


        print('Epoch-{}: {}'.format((epoch +1), training_loss/ len(train_loader)))
        
        #validation
        val_loss, balanced_acc, roc_auc, avg_precision = validate_model(
            model, val_loader, criterion, device, all_classes)
        val_losses.append(val_loss)
        val_bal_accs.append(balanced_acc)
        val_roc_aucs.append(roc_auc)
        val_avg_precs.append(avg_precision)

        print(f"Epoch {epoch + 1}/{args.epochs}")
        print(f"Training Loss: {training_loss / len(train_loader):.4f}")
        print(f"Validation Loss: {val_loss:.4f}")
        print(f"Balanced Accuracy: {balanced_acc:.4f}")
        print(f"ROC-AUC Score: {roc_auc:.4f}")
        print(f"Average Precision Score: {avg_precision:.4f}")
        print("-" * 40)

        if balanced_acc > best_balanced_acc:
            best_balanced_acc = balanced_acc
            os.makedirs(args.out_dir, exist_ok=True)
            save_path = os.path.join(args.out_dir, f"best_model_fold_{fold}.pth")
            torch.save(model.state_dict(), save_path)

    # Generate and save plots
    os.makedirs(args.out_dir, exist_ok=True)

    epochs = range(1, args.epochs + 1)

    # 1. Loss Plot
    plt.figure(figsize=(8, 6))
    plt.plot(epochs, train_losses, label='Training Loss')
    plt.plot(epochs, val_losses, label='Validation Loss')
    plt.title('Training and Validation Loss over Epochs')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(args.out_dir, f'loss_plot_fold_{fold}.png'))
    plt.close()

    # 2. Balanced Accuracy Plot
    plt.figure(figsize=(8, 6))
    plt.plot(epochs, train_bal_accs, label='Training Balanced Accuracy')
    plt.plot(epochs, val_bal_accs, label='Validation Balanced Accuracy')
    plt.title('Balanced Accuracy over Epochs')
    plt.xlabel('Epochs')
    plt.ylabel('Balanced Accuracy')
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(args.out_dir, f'balanced_accuracy_plot_fold_{fold}.png'))
    plt.close()

    # 3. ROC-AUC Plot
    plt.figure(figsize=(8, 6))
    plt.plot(epochs, train_roc_aucs, label='Training ROC-AUC')
    plt.plot(epochs, val_roc_aucs, label='Validation ROC-AUC')
    plt.title('ROC-AUC over Epochs')
    plt.xlabel('Epochs')
    plt.ylabel('ROC-AUC')
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(args.out_dir, f'roc_auc_plot_fold_{fold}.png'))
    plt.close()

    # 4. Average Precision Plot
    plt.figure(figsize=(8, 6))
    plt.plot(epochs, train_avg_precs, label='Training Average Precision')
    plt.plot(epochs, val_avg_precs, label='Validation Average Precision')
    plt.title('Average Precision over Epochs')
    plt.xlabel('Epochs')
    plt.ylabel('Average Precision')
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(args.out_dir, f'avg_precision_plot_fold_{fold}.png'))
    plt.close()

    print(f"Training complete. Plots saved in: {args.out_dir}")


def validate_model(model, val_loader, criterion,device, all_classes):
    model.eval()
    val_loss = 0.0

    all_targets = []
    all_outputs = []

    with torch.no_grad():
        for batch in val_loader:
            inputs = batch['img'].to(device)
            targets = batch['label'].to(device)

            outputs = model(inputs)
            loss = criterion(outputs, targets.long())

            val_loss += loss.item()

            # Collect outputs and targets for metrics
            all_targets.extend(targets.cpu().numpy())
            all_outputs.extend(F.softmax(outputs, dim=1).cpu().numpy())  # probabilities for multi-class


    all_targets = np.array(all_targets)
    all_outputs = np.array(all_outputs)

    # Predicted labels for balanced accuracy
    preds = np.argmax(all_outputs, axis=1)

    # Calculate metrics
    balanced_acc = balanced_accuracy_score(all_targets, preds)
    
    # ROC-AUC: For multi-class, use 'ovr' (one-vs-rest)
    try:
        roc_auc = roc_auc_score(all_targets, all_outputs, multi_class='ovr', labels=all_classes)
    except ValueError:
        roc_auc = float('nan')  # handle cases where ROC-AUC cannot be computed

    try:
        avg_precision = average_precision_score(all_targets, all_outputs, average='macro')
    except ValueError:
        avg_precision = float('nan')

    return val_loss / len(val_loader), balanced_acc, roc_auc, avg_precision, all_targets, preds
