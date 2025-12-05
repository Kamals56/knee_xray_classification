from args import get_args
import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from sklearn.metrics import balanced_accuracy_score, accuracy_score, precision_score, recall_score, f1_score
from utils import save_checkpoint, save_confusion_matrix
import json

def train_model(model, train_loader, val_loader, cur_fold):
    args = get_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    # Defining the loss function and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=args.lr)

    best_val_metric = None
    best_model_path = None
    epoch_losses = []
    loss_file = f'training_loss_fold{cur_fold}.json'

    for epoch in range(args.epochs):
        training_loss = 0
        # starting the training -> setting the model to training mode
        model.train()

        for batch in train_loader:
            inputs = batch['img'].to(device).float()
            targets = batch['label'].to(device).long()

            # Resetting the gradients
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()

            training_loss += loss.item()       # for entire batch of specific epoch

        avg_loss = training_loss/ len(train_loader)
        epoch_losses.append(avg_loss)

        with open(loss_file, 'w') as f:
            json.dump(epoch_losses, f)
        print('Epoch-{}: {}'.format((epoch + 1), avg_loss))

        ba, y_true, y_pred = validate_model(model, val_loader, criterion, device, cur_fold)

        best_val_metric, best_model_path = save_checkpoint(cur_fold,
                        epoch,
                        model,
                        y_true,
                        y_pred,
                        ba,
                        best_val_metric = best_val_metric,
                        prev_model_path = best_model_path,
                        comparator='gt',
                        save_dir=args.out_dir)


def validate_model(model, val_loader, criterion, device, cur_fold=0, epoch = 0):
    model.eval()
    val_loss = 0
    all_preds = []
    all_targets = []

    for batch in val_loader:
        inputs = batch['img'].to(device).float()
        targets = batch['label'].to(device).long()

        outputs = model(inputs)
        loss = criterion(outputs, targets)

        val_loss += loss.item()

        predictions = F.softmax(outputs, dim=1)
        pred_targets = predictions.max(dim=1)[1]

        all_preds.append(pred_targets.cpu().numpy())
        all_targets.append(targets.cpu().numpy())

    all_preds = np.concatenate(all_preds)
    all_targets = np.concatenate(all_targets)

    ba = balanced_accuracy_score(all_targets, all_preds)
    acc = accuracy_score(all_targets, all_preds)
    precision = precision_score(all_targets, all_preds, average='macro')
    recall = recall_score(all_targets, all_preds, average='macro')
    f1 = f1_score(all_targets, all_preds, average='macro')
    
    print(f'Validation Metrics - BA: {ba:.4f} | Acc: {acc:.4f} | '
          f'Precision: {precision:.4f} | Recall: {recall:.4f} | F1: {f1:.4f}')
    
    class_names = ['0','1','2','3','4']  # KL grades
    save_confusion_matrix(
        all_targets,
        all_preds,
        f'confusion_matrix_fold{cur_fold}_epoch{epoch+1}.png',
        class_names=class_names
    )


    return ba, all_targets, all_preds


