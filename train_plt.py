import argparse
import os
import torch.nn as nn
import numpy as np
import torch
import torchvision
from torch.utils.data import DataLoader
import torch.optim as optim
from torch.optim import lr_scheduler
import torch.nn.functional as F
# from model import *
from transform import mesonet_data_transforms
from tqdm import tqdm
import matplotlib.pyplot as plt
from sklearn import metrics
from efficientnet_pytorch import EfficientNet
from sklearn.metrics import precision_score,recall_score,f1_score

def main():
    print(torch.__version__)
    print(torch.cuda.is_available( ))

    args = parse.parse_args( )  
    name = args.name
    train_path = args.train_path
    test_path = args.test_path
    continue_train = args.continue_train
    epoches = args.epoches  
    batch_size = args.batch_size
    model_name = args.model_name
    model_path = args.model_path
    output_path = os.path.join('./output', name)
    if not os.path.exists(output_path):
        os.mkdir(output_path)  
    torch.backends.cudnn.benchmark = True  

    # creat train and val dataloader 
    train_dataset = torchvision.datasets.ImageFolder(train_path, transform = mesonet_data_transforms['train'])
    test_dataset = torchvision.datasets.ImageFolder(test_path, transform = mesonet_data_transforms['val'])
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size = batch_size, shuffle = True,
                                               drop_last = False, num_workers = 16)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size = batch_size, shuffle = True, drop_last = False,
                                             num_workers = 16)
    train_dataset_size = len(train_dataset)
    test_dataset_size = len(test_dataset)

    # Creat the model
    # model = EfficientNet.from_pretrained('efficientnet-b4',weights_path= 'efficientnet-b4-6ed6700e.pth')
    model = EfficientNet.from_pretrained('efficientnet-b4', weights_path='efficientnet-b4-6ed6700e.pth')
    # model = efficientnet_b4()
    pretrained_model_path = 'efficientnet-b4-6ed6700e.pth' 


    if continue_train:
        # model.load_state_dict()  
        # pretrained_model_path = torch.load(model_path)
        model.load_state_dict(torch.load(pretrained_model_path))
        # model.load_state_dict(torch.load(model_path))
    model = model.cuda( )
    criterion = nn.CrossEntropyLoss( ) 

    # optimizer = optim.SGD(model.parameters( ), lr = 0.001, momentum = 0.9, weight_decay = 0.001)
    optimizer = optim.Adam(model.parameters( ), lr = 0.001, betas = (0.9, 0.999), eps = 1e-08)  
    scheduler = lr_scheduler.StepLR(optimizer, step_size = 5, gamma = 0.5)  

    """rain the model using multiple GPUs
	model = nn.DataParallel(model)调用多个GPU，加速训练"""
    model = nn.DataParallel(model)
    best_model_wts = model.state_dict( )  
    best_auc = 0.0  
    iteration = 0  
    for epoch in range(epoches):
        print('Epoch {}/{}'.format(epoch + 1, epoches))
        print('-' * 10)
        model = model.train( )
        train_loss = 0.0
        train_corrects = 0.0  
        test_loss = 0.0
        test_corrects = 0.0
        train_probs = []  # add
        train_labels = []  # add
        for (image, labels) in train_loader:
            iter_loss = 0.0
            iter_corrects = 0.0
            image = image.cuda( )
            labels = labels.cuda( )
            optimizer.zero_grad( ) 
            outputs = model(image)
            probs = F.softmax(outputs, dim=1)  # add
            train_probs.extend(probs[:, 1].cpu().detach().numpy())  # add
            train_labels.extend(labels.cpu().detach().numpy())  # add
            _, preds = torch.max(outputs.data, 1)  
            loss = criterion(outputs, labels)  
            loss.backward( )  
            optimizer.step( )  
            iter_loss = loss.data.item( )
            train_loss += iter_loss
            iter_corrects = torch.sum(preds == labels.data).to(torch.float32)
            train_corrects += iter_corrects
            iteration += 1
            train_auc = metrics.roc_auc_score(train_labels,train_probs)  # add
            # train_precision = precision_score(train_labels,preds,average='binary')
            # train_recall = recall_score(train_labels,preds,average='binary')
            # train_f1 = f1_score(train_labels,preds,average='binary')
            if not (iteration % 500):
                print('iteration {} train loss: {:.4f} ACC: {:.4f} AUC: {:.4f} '.format(iteration, iter_loss / batch_size,
                                                                           iter_corrects / batch_size, train_auc))

        epoch_loss = train_loss / train_dataset_size
        epoch_acc = train_corrects / train_dataset_size
        train_pred = [1 if p > 0.5 else 0 for p in train_probs]
        train_precision = precision_score(train_labels,train_pred)
        train_recall = recall_score(train_labels,train_pred)
        train_f1 = f1_score(train_labels,train_pred)
        print('epoch train loss: {:.4f} ACC: {:.4f} AUC: {:.4f}'.format(epoch_loss, epoch_acc, train_auc))
        print('epoch train pred: {:.4f} recall: {:.4f} f1: {:.4f}'.format(train_precision, train_recall, train_f1))

        model.eval( )
        with torch.no_grad( ):  
            test_probs = []  # add
            test_labels = []  # add
            for (image, labels) in test_loader:
                image = image.cuda( )
                labels = labels.cuda( )
                outputs = model(image)
                probs = F.softmax(outputs, dim=1)  # add
                test_probs.extend(probs[:, 1].cpu().detach().numpy())  # add
                test_labels.extend(labels.cpu().detach().numpy())  # add
                _, preds = torch.max(outputs.data, 1)
                loss = criterion(outputs, labels)
                test_loss += loss.data.item( )
                test_corrects += torch.sum(preds == labels.data).to(torch.float32)
            epoch_loss = test_loss / test_dataset_size
            epoch_acc = test_corrects / test_dataset_size
            epoch_auc = metrics.roc_auc_score(test_labels, test_probs)
            test_auc = metrics.roc_auc_score(test_labels, test_probs)  # add
            test_pred = [1 if p > 0.5 else 0 for p in test_probs]
            test_precision = precision_score(test_labels, test_pred)
            test_recall = recall_score(test_labels, test_pred)
            test_f1 = f1_score(test_labels, test_pred)
            print('epoch test loss: {:.4f} ACC: {:.4f} AUC: {:.4f}'.format(epoch_loss, epoch_acc, test_auc))
            print('epoch test pred: {:.4f} recall: {:.4f} f1: {:.4f}'.format(test_precision, test_recall, test_f1))
            # test_precision = precision_score(test_labels, preds, average='binary')
            # test_recall = recall_score(test_labels, preds, average='binary')
            # test_f1 = f1_score(test_labels, preds, average='binary')
            print('epoch test loss: {:.4f} ACC: {:.4f}  AUC: {:.4f} '.format(epoch_loss, epoch_acc, epoch_auc))
            if epoch_auc > best_auc:
                best_auc = epoch_auc
                best_model_wts = model.state_dict( )
        scheduler.step( )
        if not (epoch % 10):
            # Save the model trained with multiple gpu
            # torch.save(model.module.state_dict(), os.path.join(output_path, str(epoch) + '_' + model_name))
            torch.save(model.state_dict( ), os.path.join(output_path, str(epoch) + '_' + model_name))
    print('Best test AUC: {:.4f}'.format(best_auc))
    model.load_state_dict(best_model_wts)
    # torch.save(model.module.state_dict(), os.path.join(output_path, "best.pkl"))
    torch.save(model.state_dict( ), os.path.join(output_path, "DF_F1_best_32.pkl")) 



    plt.figure( )
    plt.plot(range(1, epoches + 1), train_loss, label = 'Training Loss')
    plt.plot(range(1, epoches + 1), test_loss, label = 'test Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend( )
    plt.savefig(os.path.join(output_path, 'loss.png'))
    plt.show()

    plt.figure( )
    plt.plot(range(1, epoches + 1), epoch_acc, label = 'Training Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend( )
    plt.savefig(os.path.join(output_path, 'accuracy.png'))
    plt.show()


if __name__ == '__main__':
    parse = argparse.ArgumentParser(  
        formatter_class = argparse.ArgumentDefaultsHelpFormatter) 
    parse.add_argument('--name', '-n', type = str, default = 'Efficientnet')
    parse.add_argument('--batch_size', '-bz', type=int, default='32')
    parse.add_argument('--epoches', '-e', type=int, default='20')
    parse.add_argument('--train_path', '-tp', type = str, default = '/media/image/Hanberg/Dataset/FF++_dataset_by_ls/jpg/c23/faceswap/train')
    parse.add_argument('--test_path', '-ep', type = str, default = '/media/image/Hanberg/Dataset/FF++_dataset_by_ls/jpg/c23/faceswap/test')
    # parse.add_argument('--train_path', '-tp', type = str, default = '/media/terminator/Hanberg/Dataset/FF++_dataset_by_ls/jpg/c23/face2face/train')
    # parse.add_argument('--test_path', '-vp', type = str, default = '/media/terminator/Hanberg/Dataset/FF++_dataset_by_ls/jpg/c23/nerual_texture/test')
    parse.add_argument('--model_name', '-mn', type = str, default = 'efficientnet_FS_F1_40.pkl')
    parse.add_argument('--continue_train', type = bool, default = True)
    parse.add_argument('--model_path', '-mp', type=str, default='best_FS_F1.pkl')
    main( )
