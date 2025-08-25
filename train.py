#!/usr/bin/env python
import argparse
import os
import csv
import time
import math
import torch
import glob
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision.models as models
from autoattack.autopgd_base import APGDAttack
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

torch.backends.cudnn.benchmark = True

class NormalizedModel(nn.Module):
    def __init__(self, model, norm_mean, norm_std):
        super().__init__()
        self.model = model
        # Create buffers for mean and std, reshaped for broadcasting (N, C, H, W)
        self.register_buffer("mean", torch.tensor(norm_mean).view(1, -1, 1, 1))
        self.register_buffer("std", torch.tensor(norm_std).view(1, -1, 1, 1))

    def forward(self, x):
        # Perform vectorized normalization across the entire batch
        x = (x - self.mean) / self.std
        return self.model(x)
    

class MaxoutUnit(nn.Module):
    def __init__(self, in_features, out_features, pool_size):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.pool_size = pool_size
        self.linear = nn.Linear(in_features, out_features * pool_size)

    def forward(self, x):
        linear_output = self.linear(x)
        output_reshape = linear_output.view(-1, self.out_features, self.pool_size)
        return torch.max(output_reshape, dim=2).values


class MaxoutLayer(nn.Module):
    def __init__(self, in_features, out_features, pool_size):
        super().__init__()
        self.maxout_unit_1 = MaxoutUnit(in_features, out_features, pool_size)
        self.maxout_unit_2 = MaxoutUnit(in_features, out_features, pool_size)

    def forward(self, x):
        h_1 = self.maxout_unit_1(x)
        h_2 = self.maxout_unit_2(x)
        return h_2 - h_1


class TropicalLayer(nn.Module):
    """Custom layer implementing Tropical Embedding."""
    def __init__(self, in_features, out_features):
        super(TropicalLayer, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = nn.Parameter(torch.randn(out_features, in_features))
        nn.init.normal_(self.weight)

    def forward(self, x):
        """Computes tropical distance."""
        result_addition = x.unsqueeze(1) - self.weight # [B, 1, in] - [out, in] -> [B, out, in]
        return torch.min(result_addition, dim=-1).values - torch.max(result_addition, dim=-1).values  # [B, out, in] -> [B, out]
   

class ModelWithResize(nn.Module):
    def __init__(self, base_model, target_size=224):
        super(ModelWithResize, self).__init__()
        self.base_model = base_model
        self.target_size = target_size

    def forward(self, x):
        # x is 32x32; upsample to 224x224
        x = F.interpolate(x, size=(self.target_size, self.target_size), mode='bilinear', align_corners=False)
        return self.base_model(x)
    

def get_model(model_name):
    """Loads a pre-trained model"""
    if model_name == "resnet50":
        return ModelWithResize(models.resnet50(weights=models.ResNet50_Weights.DEFAULT)), "fc"
    elif model_name == "vgg16":
        return ModelWithResize(models.vgg16(weights=models.VGG16_Weights.DEFAULT)), "c_seq"
    elif model_name == "mobilenet_v2":
        return ModelWithResize(models.mobilenet_v2(weights=models.MobileNet_V2_Weights.DEFAULT)), "c_seq"
    elif model_name == "mobilenet_v3_small":
        return ModelWithResize(models.mobilenet_v3_small(weights=models.MobileNet_V3_Small_Weights.DEFAULT)), "c_seq"
    elif model_name == "alexnet":
        return ModelWithResize(models.alexnet(weights=models.AlexNet_Weights.DEFAULT)), "c_seq"
    elif model_name == "densenet121":
        return ModelWithResize(models.densenet121(weights=models.DenseNet121_Weights.DEFAULT)), "c"
    elif model_name == "convnext_tiny":
        return ModelWithResize(models.convnext_tiny(weights=models.ConvNeXt_Tiny_Weights.DEFAULT)), "c_seq"
    elif model_name == "efficientnet_b0":
        return ModelWithResize(models.efficientnet_b0(weights=models.EfficientNet_B0_Weights.DEFAULT)), "c_seq"
    elif model_name == "efficientnet_v2_s":
        return ModelWithResize(models.efficientnet_v2_s(weights=models.EfficientNet_V2_S_Weights.DEFAULT)), "c_seq"
    elif model_name == "googlenet":
        return ModelWithResize(models.googlenet(weights=models.GoogLeNet_Weights.DEFAULT)), "fc"
    elif model_name == "inception_v3":
        return ModelWithResize(models.inception_v3(weights=models.Inception_V3_Weights.DEFAULT), target_size=299), "fc"
    elif model_name == "maxvit_t":
        return ModelWithResize(models.maxvit_t(weights=models.MaxVit_T_Weights.DEFAULT)), "c_seq"
    elif model_name == "mnasnet0_5":
        return ModelWithResize(models.mnasnet0_5(weights=models.MNASNet0_5_Weights.DEFAULT)), "c_seq"
    elif model_name == "regnet_y_1_6gf":
        return ModelWithResize(models.regnet_y_1_6gf(weights=models.RegNet_Y_1_6GF_Weights.DEFAULT)), "fc"
    elif model_name == "resnext50_32x4d":
        return ModelWithResize(models.resnext50_32x4d(weights=models.ResNeXt50_32X4D_Weights.DEFAULT)), "fc"
    elif model_name == "shufflenet_v2_x0_5":
        return ModelWithResize(models.shufflenet_v2_x0_5(weights=models.ShuffleNet_V2_X0_5_Weights.DEFAULT)), "fc"
    elif model_name == "swin_v2_b":
        return ModelWithResize(models.swin_v2_b(weights=models.Swin_V2_B_Weights.DEFAULT)), "head"
    elif model_name == "vit_b_16":
        return ModelWithResize(models.vit_b_16(weights=models.ViT_B_16_Weights.DEFAULT)), "heads"
    elif model_name == "wide_resnet50_2":
        return ModelWithResize(models.wide_resnet50_2(weights=models.Wide_ResNet50_2_Weights.DEFAULT)), "fc"
    else:
        raise ValueError(f"Unsupported model: {model_name}")


def pick_last_layer(last_layer_type, num_ftrs, num_classes):
    if last_layer_type == "tropical":
        return TropicalLayer(num_ftrs, num_classes)
    elif last_layer_type == "normal":
        return nn.Linear(num_ftrs, num_classes)
    else:
        return MaxoutLayer(in_features=num_ftrs, out_features=num_classes, pool_size=100)


def swap_last_layer(model, model_last_layer_name, last_layer_type, num_classes):  
    # Initialize the last layer based on the architecture
    if model_last_layer_name == "fc":
        num_ftrs = model.base_model.fc.in_features
        model.base_model.fc = pick_last_layer(last_layer_type, num_ftrs, num_classes)
    elif model_last_layer_name == "c":
        num_ftrs = model.base_model.classifier.in_features
        model.base_model.classifier = pick_last_layer(last_layer_type, num_ftrs, num_classes)
    elif model_last_layer_name == "c_seq":
        num_ftrs = model.base_model.classifier[-1].in_features
        model.base_model.classifier[-1] = pick_last_layer(last_layer_type, num_ftrs, num_classes)
    elif model_last_layer_name == "head":
        num_ftrs = model.base_model.head.in_features
        model.base_model.head = pick_last_layer(last_layer_type, num_ftrs, num_classes)
    elif model_last_layer_name == "heads":
        num_ftrs = model.base_model.heads[-1].in_features
        model.base_model.heads[-1] = pick_last_layer(last_layer_type, num_ftrs, num_classes)
    return model


def get_dataloader(dataset_name, batch_size):
    data_dir = "./data"     # Default data directory for downloaded datasets
    split = "train"
    if dataset_name == "imagenet":
        # ImageNet is assumed to be stored locally in a separate directory.
        norm_mean = [0.485, 0.456, 0.406]
        norm_std = [0.229, 0.224, 0.225]
        transform = transforms.Compose([
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            #transforms.Normalize(mean=norm_mean, std=norm_std)
        ])
        dataset = datasets.ImageNet(root="imagenet_files", split=split, transform=transform)
    elif dataset_name == "mnist":
        norm_mean, norm_std = (0.1307,), (0.3081,)
        transform = transforms.Compose([
            transforms.ToTensor(),
            #transforms.Normalize(norm_mean, norm_std)
        ])
        dataset = datasets.MNIST(root=data_dir, train=(split == "train"), download=True, transform=transform)
    elif dataset_name == "fmnist":
        norm_mean, norm_std = (0.2860,), (0.3530,)  # or adjust as needed
        transform = transforms.Compose([
            transforms.ToTensor(),
            #transforms.Normalize(norm_mean, norm_std)
        ])
        dataset = datasets.FashionMNIST(root=data_dir, train=(split == "train"), download=True, transform=transform)
    elif dataset_name == "cifar10":
        norm_mean = (0.4914, 0.4822, 0.4465)
        norm_std = (0.2023, 0.1994, 0.2010)
        transform = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.AutoAugment(policy=transforms.AutoAugmentPolicy.CIFAR10),
            transforms.ToTensor(),
            #transforms.Normalize(norm_mean, norm_std)
        ])
        dataset = datasets.CIFAR10(root=data_dir, train=(split == "train"), download=True, transform=transform)
    elif dataset_name == "svhn":
        # SVHN uses 'train' and 'test' splits (no validation)
        norm_mean = (0.4377, 0.4438, 0.4728)
        norm_std = (0.1980, 0.2010, 0.1970)
        transform = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.AutoAugment(policy=transforms.AutoAugmentPolicy.SVHN),
            transforms.ToTensor(),
            #transforms.Normalize(norm_mean, norm_std)
        ])
        dataset = datasets.SVHN(root=data_dir, split='train', download=True, transform=transform)
    elif dataset_name == "cifar100":
        # CIFAR100: similar to CIFAR10 but with 100 classes.
        norm_mean = (0.5071, 0.4865, 0.4409)
        norm_std = (0.2673, 0.2564, 0.2761)
        transform = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.AutoAugment(policy=transforms.AutoAugmentPolicy.CIFAR10),
            transforms.ToTensor(),
            #transforms.Normalize(norm_mean, norm_std)
        ])
        dataset = datasets.CIFAR100(root=data_dir, train=(split == "train"), download=True, transform=transform)
    else:
        raise ValueError(f"Unsupported dataset: {dataset_name}")   
    return DataLoader(dataset, batch_size=batch_size, shuffle=(split == "train"), num_workers=4, pin_memory=True), norm_mean, norm_std


def train_model(model, dataloader, meta_data):
    """Trains the model on the given dataset."""
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=meta_data['lr'])
    model.to(meta_data['device'])
    

    if meta_data['at']:
        model.eval()
        print(f'starting adversarial training loop')
        at_adversary = APGDAttack(model, n_iter=10, eps=0.0, verbose=False, device=meta_data['device'])
        list_at_params = [('Linf', 4/255),
                          ('L2', 0.5),
                          ('L1', 1.5)]
        #aa_linf_adversary = AutoAttack(model, norm="Linf", eps=4/255, version='standard', device=meta_data['device'])
        #aa_l2_adversary = AutoAttack(model, norm="L2", eps=0.5, version='standard', device=meta_data['device'])
        #aa_l1_adversary = AutoAttack(model, norm="L1", eps=1.5, version='standard', device=meta_data['device'])
        #list_aa = [aa_linf_adversary, aa_l2_adversary, aa_l1_adversary]
    else:
        print(f'not adversarially training')
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=meta_data['num_epochs']) ### increase num_epochs schedule?? for cifar 100
    with open(f"training/training_{get_files_names(meta_data)}.csv", mode="w", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(["epoch", "running_loss", "epoch_accuracy", "time"])

    model.train()
    for epoch in range(meta_data['num_epochs']):
        running_loss = 0.0
        correct = 0
        total = 0
        for images, labels in dataloader:
            #batch_size = images.shape[0]
            if meta_data['at']:
                list_x_advs = []
                model.eval()
                for norm, eps in list_at_params:  
                    at_adversary.eps = eps
                    at_adversary.norm = norm
                    x_adv = at_adversary.perturb(images, labels)
                    list_x_advs.append(x_adv)
                images = torch.cat(list_x_advs, dim=0)
                labels = torch.cat([labels for _ in range(len(list_at_params))], dim=0)
                model.train()
            images, labels = images.to(meta_data['device']), labels.to(meta_data['device'])
            optimizer.zero_grad()
            outputs = model(images)
            #if meta_data['model'] == 'inception_v3':
                #outputs = outputs[0]
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
            _, predicted = torch.max(outputs, 1) 
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            with open(f"training/training_{get_files_names(meta_data)}.csv", mode="a", newline="") as file:
                writer = csv.writer(file)
                writer.writerow([str(epoch)+"_"+str(total), 
                                running_loss / len(dataloader), 
                                100*correct/total,
                                time.time()])
        if not meta_data['at']:
            scheduler.step()
        print(f"Epoch {epoch+1}/{meta_data['num_epochs']}, Loss: {running_loss/len(dataloader):.4f}, Acc: {100*correct/total:.4f}")
    return model


def get_files_names(meta_data):
    return f"{meta_data['at']}_{meta_data['dataset']}_{meta_data['model']}_{meta_data['last_layer']}_{meta_data['lr']}_{meta_data['num_epochs']}_{meta_data['time']}"


if __name__ == "__main__":
    # ---- argument parsings ----
    parser = argparse.ArgumentParser()
    parser.add_argument("--combo", type=int, default=0, help="Combo index to select model, dataset, and layer type")
    parser.add_argument("--num_epochs", type=int, default=200, help="Number of epochs")
    parser.add_argument("--lr", type=float, default=0.001, help="Learning rate")
    parser.add_argument("--at", type=bool, default=False, help="adversarial training")
    args = parser.parse_args()
    
    bs = 50

    # ---- lists for choosing model ----
    list_models = ["alexnet", #0
                    "shufflenet_v2_x0_5", #1
                    "mobilenet_v3_small", #2
                    "googlenet", #3
                    "efficientnet_b0", #4
                    "resnet50"] #5
    #list_datasets = ["cifar10"]
    list_last_layers = ["tropical", "normal"]
    dict_num_classes = {"imagenet":1000,
                        "cifar100":100,
                        "cifar10":10,
                        "svhn":10,
                        "mnist":10,
                        "fmnist":10}
    # ---- setting up metadata ----
    meta_data = {'model':list_models[math.floor(args.combo / 2)],
                 'dataset':'cifar10',#list_datasets[args.combo % 2],
                 'last_layer':list_last_layers[args.combo % 2],
                 'num_epochs':args.num_epochs,
                 'lr':args.lr,
                 'at':args.at,
                 'device':torch.device("cuda" if torch.cuda.is_available() else "cpu"),
                 'time':time.strftime("%Y%m%d_%H%M%S",time.localtime())}

    pattern = os.path.join("models", f"{meta_data['at']}_{meta_data['dataset']}_{meta_data['model']}_{meta_data['last_layer']}_*.pth")
    
    matching_files = glob.glob(pattern)
    if matching_files:
        print("we done did it already", matching_files)
    else:
        if meta_data["model"] == "alexnet" or meta_data["model"] == "vgg16":
            meta_data['lr'] = 0.0001

        print(get_files_names(meta_data))
        print(meta_data['device'])

        print("Loading dataset...")
        train_loader, norm_mean, norm_std = get_dataloader(meta_data["dataset"], batch_size=bs)

        print("Loading model...")
        model, model_last_layer_name = get_model(meta_data["model"])
        if meta_data["dataset"] != "imagenet" or meta_data["last_layer"] == "tropical":
            model = swap_last_layer(model, model_last_layer_name, meta_data["last_layer"], dict_num_classes[meta_data["dataset"]])
        if meta_data['at']:
            no_at_pattern = os.path.join("models", f"False_{meta_data['dataset']}_{meta_data['model']}_{meta_data['last_layer']}_*.pth")
            no_at_matching_files = glob.glob(no_at_pattern)
            state_dict = torch.load(no_at_matching_files[0], map_location=meta_data["device"])
            model.load_state_dict(state_dict) ### This line and line below might have to change on future run since you're starting to wrap all models as "normalized" models.
        model = NormalizedModel(model, norm_mean, norm_std)
        print("Training model...")
        model = train_model(model, train_loader, meta_data)

        # Save results
        model_path = f"models/{get_files_names(meta_data)}.pth"
        torch.save(model.state_dict(), model_path)
        print(f"Model saved to {model_path}")
