import numpy
import matplotlib.pyplot as plt
import torch
from torch import nn, optim
from torchvision import datasets, transforms
import torchvision.models as models
import json
from PIL import Image
import os
from torch.autograd import Variable

def transformations(root):

    data_dir = root
    train_dir = data_dir + '/train'
    valid_dir = data_dir + '/valid'
    test_dir = data_dir + '/test'

    data_transforms = { 
        'train': transforms.Compose([
            transforms.RandomResizedCrop(size=224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize([0.485,0.456,0.406],
            [0.229,0.224,0.225])
            ]),
        'valid': transforms.Compose([
            transforms.Resize(size=256),
            transforms.CenterCrop(size=224),
            transforms.ToTensor(),
            transforms.Normalize([0.485,0.456,0.406],
            [0.229,0.224,0.225])
            ]),
        'test': transforms.Compose([
            transforms.Resize(size=256),
            transforms.CenterCrop(size=224),
            transforms.ToTensor(),
            transforms.Normalize([0.485,0.456,0.406],
             [0.229,0.224,0.225])
             ])
             }
    dict_name = ['train', 'valid', 'test']

    image_datasets = {x: datasets.ImageFolder(os.path.join(data_dir, x), transform=data_transforms[x]) 
    for x in dict_name}
    
    return image_datasets

def dataloader(root):
    
    data_dir = root    
    images = transformations(data_dir)
    
    dict_name = ['train', 'valid', 'test']

    dataloaders = {x: torch.utils.data.DataLoader(images[x], batch_size=64, shuffle=True) 
    for x in dict_name}
   
    return dataloaders

def labels():
    with open('cat_to_name.json', 'r') as f:
        cat_to_name = json.load(f)

    return cat_to_name


def build_features(name, dropout=0.5, device='gpu'):

    if name == 'vgg':

        the_model = models.vgg19_bn(pretrained=True)
        
        the_model.classifier[-1] = nn.Linear(4096, 1000)
        the_model.classifier.add_module('7', nn.ReLU())
        the_model.classifier.add_module('8', nn.Dropout(dropout))
        the_model.classifier.add_module('9', nn.Linear(1000, 102))
        the_model.classifier.add_module('10', nn.LogSoftmax(dim=1))
        
        for param in the_model.features.parameters():
            param.requires_grad = False
        
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(the_model.classifier.parameters(), lr=0.001 )

    elif name == 'resnet':

        the_model = models.resnet50(pretrained=True)

        for param in the_model.parameters():
            param.requires_grad = False
        
        classifier = nn.Sequential(
            nn.Linear(the_model.fc.in_features, 1024),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(1024, 1024),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(1024, 102),
            nn.LogSoftmax(dim=1))
        
        the_model.fc = classifier

        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(the_model.fc.parameters(), lr=0.001 )

    elif name == 'densenet':
        the_model = models.densenet121(pretrained = True)

        classifier = nn.Sequential(
            nn.Linear(1024,1000),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(1000, 1000),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(1000, 102),
            nn.LogSoftmax(dim=1))
        
        the_model.classifier = classifier
        
        for param in the_model.features.parameters():
            param.requires_grad = False
    else:
        the_model = 'Model Not Found'

    return the_model, criterion, optimizer




device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def train_model(a_name):
    print('\n')
    print('Training process of ' , a_name)
    our_model, our_criterion, our_optimizer = build_features(a_name)
        
    our_model.to(device)
    epochs = 6
    step = 0
    print_every = 20
    
    data_load = dataloader('./flowers/')
    for epoch in range(epochs):
        for inputs, labels in data_load['train']:
            running_loss = 0
            train_accuracy = 0
            step += 1

            our_optimizer.zero_grad()

            inputs, labels = inputs.to(device), labels.to(device)

            output = our_model(inputs)
            loss = our_criterion(output, labels)
            loss.backward()
            our_optimizer.step()

            running_loss += loss.item()

            ps = torch.exp(output)
            equality = (labels.data == ps.max(dim = 1)[1])
            train_accuracy += equality.type(torch.FloatTensor).mean()

            if step % print_every == 0:
                our_model.eval()
                valid_loss = 0
                valid_accuracy = 0
                    
                with torch.no_grad():
                    for inputs, labels in data_load['valid']:
                        inputs, labels = inputs.to(device), labels.to(device)
                        output = our_model(inputs)
                        valid_loss += our_criterion(output, labels).item()
                        
                        ps = torch.exp(output)
                        equality = (labels.data == ps.max(dim=1)[1])
                        valid_accuracy += equality.type(torch.FloatTensor).mean()
                        
                    print("Epoch: {}/{}.. ".format(epoch + 1, epochs),"Training Loss: {:.3f}.. ".format(running_loss/print_every),
                          "Valid Loss: {:.3f}.. ".format(valid_loss/len(data_load['valid'])),
                          "Valid Accuracy: {:.3f}%".format(valid_accuracy/len(data_load['valid'])*100))
                    
                    running_loss = 0
                    train_accuracy = 0
                    our_model.train()

                    
                    
def test_model(the_name):
    dataloaders = dataloader('./flowers/')
    testing_model, t_criterion, t_optimizer = build_features(the_name)
    m_n = the_name
    testing_model.to('cuda')
    valid_p = 0
    with torch.no_grad():
        for images, labels in dataloaders['test']:
            images, labels = images.to('cuda'), labels.to('cuda')
                
            #Calculating accuracy
            outputs = testing_model(images)
            _, predicted = torch.max(outputs.data, 1)
            #total += labels.size(0)
            valid_p += torch.sum(predicted == labels.data).item()
        print(m_n, 'Accuracy of the network on test data set: {}%'.format((valid_p / len(dataloaders['test'])) * 100))


def save_checkpoint(model_name):

    image_dataset = transformations('./flowers/')
    save_model = build_features(model_name)
    #save_model.class_to_idx = image_dataset['train'].class_to_idx
    save_model.cpu()

    if model_name == 'vgg':
        checkpoint = {
            'arch': 'vgg19',
            'class_to_idx': save_model.class_to_idx,
            'opt': optim.Adam(save_model.classifier.parameters(), lr=0.001 ),
            'classifier': save_model.classifier,
            'num_epochs': 6}
        torch.save(checkpoint, 'checkpoint_vgg19.pth')
    
    elif model_name == 'resnet':
        checkpoint = {
            'arch': 'resnet50',
            'class_to_idx': save_model.class_to_idx,
            'opt': optim.Adam(save_model.fc.parameters(), lr=0.001 ),
            'in_f': 2048,
            'classifier': save_model.fc,
            'num_epochs': 6}
        torch.save(checkpoint, 'checkpoint_resnet50.pth')
    
    elif model_name == 'densenet':
        checkpoint = {
            'arch': 'densenet121',
            'class_to_idx': save_model.class_to_idx,
            'opt': optim.Adam(save_model.classifier.parameters(), lr=0.001 ),
            'classifier': save_model.classifier,
            'num_epochs': 6}
        torch.save(checkpoint, 'checkpoint_densenet121.pth')
    
    return checkpoint


def load_checkpoint(path):

    checkpoint = torch.load(path)

    if 'vgg' in path:
        trained_model = build_features('vgg')
        structure = checkpoint['arch']
        trained_model.class_to_idx = checkpoint['class_to_idx']
        optimizer = checkpoint['opt']
        trained_model.classifier = checkpoint['classifier'] 
        epoches = checkpoint['num_epochs']
    
    elif 'resnet' in path:
        trained_model = build_features('resnet')
        structure = checkpoint['arch']
        trained_model.class_to_idx = checkpoint['class_to_idx']
        optimizer = checkpoint['opt']
        trained_model.classifier = checkpoint['classifier'] 
        epoches = checkpoint['num_epochs']
    elif 'densenet' in path:
        trained_model = build_features('densenet')
        structure = checkpoint['arch']
        trained_model.class_to_idx = checkpoint['class_to_idx']
        optimizer = checkpoint['opt']
        trained_model.classifier = checkpoint['classifier'] 
        epoches = checkpoint['num_epochs']
    
    return trained_model


def process_image(image_path):

    img = Image.open(image_path)
    img_process = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    the_image = img_process(img)
    return the_image


def imshow(image, ax=None, title=None):

    if ax is None:
        fig, ax = plt.subplots()
    
    image = image.numpy().transpose((1, 2, 0))
    
    mean = numpy.array([0.485, 0.456, 0.406])
    std = numpy.array([0.229, 0.224, 0.225])
    image = std * image + mean
    
    image = numpy.clip(image, 0, 1)
    
    ax.imshow(image)
    
    return ax



def predict_label(image_path, check_path, n_model, topk=5):

    selected_model = load_checkpoint(check_path, n_model)

    image = process_image(image_path)
    image = image.unsqueeze_(0)
    
    selected_model.to('cuda')
        
    selected_model.eval()

    tensor = image.float().cuda()
    output = selected_model.forward(tensor)  
    ps = torch.exp(output).data.topk(topk)

    probabilities = ps[0].cpu()
    classes = ps[1].cpu()
    
    class_to_idx_inverted = {selected_model.class_to_idx[key]: key for key in selected_model.class_to_idx}
    associated_classes = [class_to_idx_inverted[label] for label in classes.numpy()[0]]
    probabilities = probabilities.numpy()[0]
    return probabilities, associated_classes

def check_sanity(img_path, mmodell):
    probabilities, classes = predict_label(img_path, mmodell)
    max_index = numpy.argmax(probabilities)
    max_probability = probabilities[max_index]
    label = classes[max_index]

    plt.figure(figsize=[40, 10])
    flower_plot = plt.subplot2grid((15,9), (0,0), colspan=9, rowspan=9)

    plt.xlabel('X')
    plt.ylabel('Y')
    plt.yticks(rotation = 180)
    image = Image.open(img_path)
    labels_list = labels()
    flower_plot.set_title(labels_list[label])
    flower_plot.imshow(image)