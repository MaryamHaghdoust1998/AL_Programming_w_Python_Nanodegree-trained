import argparse

from functions import dataloader, labels, train_model, test_model, save_checkpoint

parser = argparse.ArgumentParser('Train.py')

parser.add_argument('--data_dir', dest='data_dir' , action="store", default="./flowers/", help="Define the directory for data ")

parser.add_argument('--arch', dest="arch", action="store", default="vgg", type=str,
                    metavar='', help="CNN model architecture: vgg19")

args = parser.parse_args()


data_loads = dataloader(args.data_dir)
cat_to_name = labels()
train_model(args.arch)
test_model(args.arch)
save_checkpoint(args.arch)

print("The Model is trained and saved")