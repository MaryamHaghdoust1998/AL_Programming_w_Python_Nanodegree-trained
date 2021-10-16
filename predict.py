import argparse
from functions import predict_label

parser = argparse.ArgumentParser(description="Predict Image Class")

parser.add_argument('--image_path', dest='image_path' , default='flowers/test/1/image_06752.jpg', 
                    action="store", type=str, help="Define the directory for the Image")
parser.add_argument('--arch', dest="arch", action="store", default="vgg", type=str,
                    metavar='', help="CNN model architecture: vgg19")
parser.add_argument('--checkpoint', dest='checkpoint', default='checkpoint_vgg19.pth',
                    action="store", type=str, help="Define the directory to the PTH file")
parser.add_argument('--top_k', dest='top_k' , default=5, action="store", type=int, help="To show the top_k Prediction")
parser.add_argument('--gpu', default="gpu", action="store", help="To use gpu power")
args = parser.parse_args()

predict_label(args.image_path, args.checkpoint, args.arch, args.top_k)