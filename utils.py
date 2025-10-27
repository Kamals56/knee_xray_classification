import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
from args import get_args

import os
from termcolor import colored
from colorama import init  # linux
import pickle
import operator


def save_checkpoint(cur_fold, cur_epoch, model, y_true, y_pred, val_metric,
                    best_val_metric=None, prev_model_path=None,
                    comparator='gt', save_dir='sessions'):
   # init(autoreset=True)  # linux


    os.makedirs(save_dir, exist_ok=True)
    comparator = getattr(operator, comparator)
    cur_snapshot_name = os.path.join(save_dir, f'fold_{cur_fold}_epoch_{cur_epoch + 1}.pth')

    if best_val_metric is None or comparator(val_metric, best_val_metric):
        if prev_model_path and os.path.exists(prev_model_path):
            os.remove(prev_model_path)

        torch.save(model.state_dict(), cur_snapshot_name)
        print(colored('====> Snapshot saved to: ', 'green') + colored(cur_snapshot_name, 'yellow'))
        print(colored(f'Validation metric improved to: {val_metric:.4f}', 'cyan'))

        session_info = {
            'fold_id': cur_fold,
            'epoch': cur_epoch,
            'best_val_metric': val_metric,
            'best_val_preds': y_pred,
            'best_val_target': y_true,
            'model_path': cur_snapshot_name
        }

        filename = os.path.join(save_dir, f'session_fold_{cur_fold}.pkl')
        with open(filename, 'wb') as f:
            pickle.dump(session_info, f)

        return val_metric, cur_snapshot_name
    else:
        return best_val_metric, prev_model_path



def estimate_mean_std(dataset):
    args = get_args()
    mean_std_loader = DataLoader(dataset, batch_size=args.batch_size, num_workers=8, pin_memory=torch.cuda.is_available())
    len_inputs = len(mean_std_loader.sampler)
    mean = 0
    std = 0
    for sample in tqdm(mean_std_loader, desc='Computing mean and std values:'):
        local_batch, local_labels = sample['img'], sample['target']

        for j in range(local_batch.shape[0]):
            mean += local_batch.float()[j, :, :, :].mean()
            std += local_batch.float()[j, :, :, :].std()

    mean /= len_inputs
    std /= len_inputs

    return mean, std