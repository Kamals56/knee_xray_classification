import argparse
import os

def get_args():
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))

    parser = argparse.ArgumentParser(description="Model training options")

    parser.add_argument("-backbone", type = str, default = "resnet18", 
                        choices=["resnet18", "resnet34", "resnet50"])
    
    parser.add_argument("-csv_dir", type = str, default=os.path.join(BASE_DIR, "../data/csv_datasets"))
    
    parser.add_argument("-batch_size", type= int, default = "32", 
                        choices = [16, 32, 64])

    parser.add_argument("-lr", type = float, default = 1e-3)
    
    parser.add_argument("-epochs" , default = 10)

    parser.add_argument("-out_dir", type = str, default=os.path.join(BASE_DIR, "../session"))

    parser.add_argument("-n_splits", type=int, default=5, help="Number of folds for cross-validation")

    args = parser.parse_args()
    
    return args
    
