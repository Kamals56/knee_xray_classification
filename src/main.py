import os
from args import get_args
import pandas as pd
from datasets import Knee_Xray_Dataset
from torch.utils.data import DataLoader
from models import MyModel
from trainer import train_model
import torch

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
def main():
    #1, we need some arguments 

    args = get_args()

    # 2 - iterate among the folds

    for fold in range(args.n_splits):
        print(f"Training on fold:{fold + 1}")
        train_set = pd.read_csv(os.path.join(args.csv_dir, "train_fold{}.csv".format(str(fold))))
        val_set = pd.read_csv(os.path.join(args.csv_dir, "val_fold{}.csv".format(str(fold))))

        #3. Preparing datasets
        train_dataset = Knee_Xray_Dataset(train_set)
        val_dataset = Knee_Xray_Dataset(val_set)

        #4. Creating data loaders
        train_loader = DataLoader(train_dataset, batch_size= args.batch_size, shuffle=True,
                                  num_workers = 2, pin_memory= torch.cuda.is_available())
        val_loader = DataLoader(val_dataset, batch_size = args.batch_size, shuffle = True,
                                num_workers = 0, pin_memory= torch.cuda.is_available())

        #5. Initializing the model
        model = MyModel(args.backbone).to(device)

        #6.
        train_model(model, train_loader, val_loader, fold)
        
    print("All folds finished.")


if __name__ =="__main__":
    main()

