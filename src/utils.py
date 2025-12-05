import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
from args import get_args
import os
import pickle
import operator
from termcolor import colored
from colorama import init
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, roc_auc_score, roc_curve, auc
import seaborn as sns
import numpy as np

init(autoreset=True)

# Checkpointing and Metrics
def save_checkpoint(cur_fold, cur_epoch, model, y_true, y_pred, val_metric,
                    best_val_metric=None, prev_model_path=None,
                    comparator='gt', save_dir=None):
    """
    Saves model checkpoint if validation metric improves.
    Also saves confusion matrix and ROC-AUC score.
    """
    args = get_args()

    if save_dir is None:
        save_dir = args.out_dir
    os.makedirs(save_dir, exist_ok=True)

    comparator_fn = getattr(operator, comparator)
    checkpoint_name = os.path.join(save_dir, f'fold_{cur_fold}_epoch_{cur_epoch+1}.pth')

    if best_val_metric is None or comparator_fn(val_metric, best_val_metric):
        # Remove previous checkpoint
        if prev_model_path and os.path.exists(prev_model_path):
            os.remove(prev_model_path)

        # Save model weights
        torch.save(model.state_dict(), checkpoint_name)
        print(colored(f'====> Snapshot saved to: {checkpoint_name}', 'green'))
        print(colored(f'Validation metric improved to: {val_metric:.4f}', 'cyan'))

        # Save session info
        session_info = {
            'fold': cur_fold,
            'epoch': cur_epoch,
            'best_val_metric': val_metric,
            'best_val_preds': y_pred,
            'best_val_targets': y_true,
            'model_path': checkpoint_name
        }

        session_file = os.path.join(save_dir, f'session_fold_{cur_fold}.pkl')
        with open(session_file, 'wb') as f:
            pickle.dump(session_info, f)

        # Save confusion matrix
        cm = confusion_matrix(y_true, y_pred)
        cm_file = os.path.join(save_dir, f'confusion_fold_{cur_fold}_epoch_{cur_epoch+1}.png')
        plt.figure(figsize=(6,6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
        plt.title(f'Confusion Matrix Fold {cur_fold} Epoch {cur_epoch+1}')
        plt.ylabel('True')
        plt.xlabel('Pred')
        plt.savefig(cm_file)
        plt.close()

        # Save ROC-AUC score if multi-class
        try:
            roc_file = os.path.join(save_dir, f'roc_fold_{cur_fold}_epoch_{cur_epoch+1}.png')
            y_true_onehot = np.eye(len(np.unique(y_true)))[y_true]
            y_pred_prob = np.array(y_pred)
            if y_pred_prob.ndim == 1 or y_pred_prob.shape[1] == 1:
                y_pred_prob = y_pred_prob.reshape(-1,1)
            n_classes = y_true_onehot.shape[1]
            fpr = dict()
            tpr = dict()
            roc_auc = dict()
            for i in range(n_classes):
                fpr[i], tpr[i], _ = roc_curve(y_true_onehot[:, i], y_pred_prob[:, i])
                roc_auc[i] = auc(fpr[i], tpr[i])
            plt.figure()
            for i in range(n_classes):
                plt.plot(fpr[i], tpr[i], label=f'Class {i} (area={roc_auc[i]:.2f})')
            plt.plot([0,1], [0,1], 'k--')
            plt.xlabel('False Positive Rate')
            plt.ylabel('True Positive Rate')
            plt.title(f'ROC Curve Fold {cur_fold} Epoch {cur_epoch+1}')
            plt.legend()
            plt.savefig(roc_file)
            plt.close()
        except:
            pass

        return val_metric, checkpoint_name

    else:
        return best_val_metric, prev_model_path

# Plotting metrics

def plot_metrics(train_losses, val_losses, val_metrics, save_dir, fold):
    """
    Plot training and validation loss & validation metric over epochs
    """
    os.makedirs(save_dir, exist_ok=True)
    epochs = range(1, len(train_losses)+1)

    plt.figure(figsize=(12,5))

    # Loss
    plt.subplot(1,2,1)
    plt.plot(epochs, train_losses, 'b-', label='Training Loss')
    plt.plot(epochs, val_losses, 'r-', label='Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Loss per Epoch')
    plt.legend()

    # Validation Metric (BA)
    plt.subplot(1,2,2)
    plt.plot(epochs, val_metrics, 'g-', label='Balanced Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('BA')
    plt.title('Balanced Accuracy per Epoch')
    plt.legend()

    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, f'metrics_fold_{fold}.png'))
    plt.close()

# Compute Dataset Mean & Std

def estimate_mean_std(dataset):
    """
    Compute channel-wise mean and std of a dataset
    """
    args = get_args()
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False,
                        num_workers=4, pin_memory=torch.cuda.is_available())
    mean = 0.0
    std = 0.0
    n_samples = 0

    for batch in tqdm(loader, desc="Computing mean/std"):
        imgs = batch['img'].float()
        batch_size = imgs.size(0)
        mean += imgs.mean(dim=[0,2,3]).sum()
        std += imgs.std(dim=[0,2,3]).sum()
        n_samples += batch_size

    mean /= n_samples
    std /= n_samples
    return mean.item(), std.item()
