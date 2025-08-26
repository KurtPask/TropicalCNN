#!/usr/bin/env python
import argparse
import os
import glob
import torch
import math
import csv
import time
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
import schedulefree
from pcl_regularizer import Proximity, ConProximity
from advertorch.attacks import CarliniWagnerL2Attack, LinfSPSAAttack 
from autoattack import AutoAttack
from autoattack.autopgd_base import APGDAttack
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Subset

class LeNet5(nn.Module):
    def __init__(self, channels=1):
        super(LeNet5, self).__init__()
        # Convolutional layers
        self.conv1 = nn.Conv2d(channels, 6, kernel_size=5)
        self.conv2 = nn.Conv2d(6, 16, kernel_size=5)
        # Fully connected layers
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc = nn.Linear(84, 10)

    def forward(self, x):
        # Input size: 32x32
        # Convolutional and pooling layers
        x = F.tanh(self.conv1(x)) # Output size: 28x28
        x = F.max_pool2d(x, kernel_size=2, stride=2) # Output size: 14x14
        x = F.tanh(self.conv2(x)) # Output size: 10x10
        x = F.max_pool2d(x, kernel_size=2, stride=2) # Output size: 5x5
        # Flatten the output for fully connected layers
        x = x.view(-1, 16 * 5 * 5)
        # Fully connected layers
        x = F.tanh(self.fc1(x))
        x = F.tanh(self.fc2(x))
        x = self.fc(x)
        return x


class CustomSimpleCNN(nn.Module):
    def __init__(self, channels=1):
        super(CustomSimpleCNN, self).__init__()
        self.conv1 = nn.Conv2d(channels, 64, kernel_size=3)
        self.conv2 = nn.Conv2d(64, 64, kernel_size=3)
        self.conv3 = nn.Conv2d(64, 64, kernel_size=3)
        self.flattened_layer_count = 64 * 4 * 4
        self.fc1 = nn.Linear(self.flattened_layer_count, 64)
        self.fc = nn.Linear(64, 10)
    
    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.max_pool2d(x, 2) 
        x = F.relu(self.conv2(x))
        x = F.max_pool2d(x, 2) 
        x = F.relu(self.conv3(x))
        x = x.view(-1, self.flattened_layer_count)
        x = F.relu(self.fc1(x))
        x = self.fc(x)
        return x


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
    

class Experiment:
    def __init__(self,
                 task,
                 at,
                 model_name,
                 dataset,
                 last_layer_type,
                 device,
                 regularizer='none',
                 num_epochs = 50,
                 lr = 0.001,
                 slice = 0,
                 total_slices = 1,
                 batch_size = 100,
                 normalize_model = 'none'):
        # -- assertions --
        assert task in ['attack', 'evaluate', 'train']
        assert last_layer_type in ['tropical', 'normal', 'maxout']
        assert type(at) is bool
        assert regularizer in ['none', 'pcl_l2', 'pcl_tropical']

        # -- essentials --
        self.task = task
        self.at = at
        print(self.at)
        self.model_name = model_name
        self.dataset = dataset
        self.channels = 1 if dataset == 'mnist' else 3
        self.split = 'train' if self.task == 'train' else 'test'
        self.last_layer_type = last_layer_type
        self.normalize_model = normalize_model
        self.regularizer = regularizer
        
        # -- data --
        dict_num_classes = {"cifar100":100, "cifar10":10, "svhn":10, 'mnist':10}
        self.num_classes = dict_num_classes[dataset]
        self.slice = slice
        self.total_slices = total_slices
        self.num_epochs = num_epochs
        self.lr = lr
        self.batch_size = batch_size

        # -- tracking --
        self.device = device
        self.init_time = self._time_string()
        self.base_pattern = f"{self.at}_{self.dataset}_{self.model_name}_{self.last_layer_type}_{self.regularizer}_"
        self.full_file_name =  f"{self.base_pattern}{self.lr}_{self.num_epochs}_{self.init_time}"
        print(self.full_file_name)
        self.cached_last_layer_input = None
        self.hook_handle = None

    def run(self):
        self.set_dataloader()
        self.set_model()
        if self.task == 'evaluate':
            self.evaluate_model()
        elif self.task == 'attack':
            self.attack_model()
        else:
            self.train_model()
            
    
    def _set_base_model(self):
        print(f'...loading pre-trained model...{self._time_string()}')
        if self.model_name == "resnet50":
            self.model = ModelWithResize(models.resnet50(weights=models.ResNet50_Weights.DEFAULT))
            self.model_last_layer_type = "fc"
        elif self.model_name == "vgg16":
            self.model = ModelWithResize(models.vgg16(weights=models.VGG16_Weights.DEFAULT))
            self.model_last_layer_type = "c_seq"
        elif self.model_name == "mobilenet_v3_small":
            self.model = ModelWithResize(models.mobilenet_v3_small(weights=models.MobileNet_V3_Small_Weights.DEFAULT))
            self.model_last_layer_type = "c_seq"
        elif self.model_name == "alexnet":
            self.model = ModelWithResize(models.alexnet(weights=models.AlexNet_Weights.DEFAULT))
            self.model_last_layer_type = "c_seq"
        elif self.model_name == "efficientnet_b0":
            self.model = ModelWithResize(models.efficientnet_b0(weights=models.EfficientNet_B0_Weights.DEFAULT))
            self.model_last_layer_type = "c_seq"
        elif self.model_name == "googlenet":
            self.model = ModelWithResize(models.googlenet(weights=models.GoogLeNet_Weights.DEFAULT))
            self.model_last_layer_type = "fc"
        elif self.model_name == "shufflenet_v2_x0_5":
            self.model = ModelWithResize(models.shufflenet_v2_x0_5(weights=models.ShuffleNet_V2_X0_5_Weights.DEFAULT))
            self.model_last_layer_type = "fc"
        elif self.model_name == 'lenet5':
            self.model = LeNet5(channels=self.channels)
            self.model_last_layer_type = 'fc'
        elif self.model_name == 'custom_lenet':
            self.model = CustomSimpleCNN(channels=self.channels)
            self.model_last_layer_type = 'fc'
        else:
            raise ValueError(f"Unsupported model: {self.model_name}")

    def _set_last_layer(self):
        if self.last_layer_type == "tropical":
            self.last_layer = TropicalLayer(self.num_ftrs, self.num_classes)
        elif self.last_layer_type == "normal":
            self.last_layer = nn.Linear(self.num_ftrs, self.num_classes)
        else:
            self.last_layer = MaxoutLayer(in_features=self.num_ftrs, out_features=self.num_classes, pool_size=100)
    
    def _swap_last_layer(self):
        print(f'...swapping last layer...{self._time_string()}')
        if self.model_name in ['custom_lenet', 'lenet5']:
            if self.model_last_layer_type == "fc":
                self.num_ftrs = self.model.fc.in_features
                self._set_last_layer()
                self.model.fc = self.last_layer
        else: 
            if self.model_last_layer_type == "fc":
                self.num_ftrs = self.model.base_model.fc.in_features
                self._set_last_layer()
                self.model.base_model.fc = self.last_layer
            elif self.model_last_layer_type == "c":
                self.num_ftrs = self.model.base_model.classifier.in_features
                self._set_last_layer()
                self.model.base_model.classifier = self.last_layer
            elif self.model_last_layer_type == "c_seq":
                self.num_ftrs = self.model.base_model.classifier[-1].in_features
                self._set_last_layer()
                self.model.base_model.classifier[-1] = self.last_layer
            elif self.model_last_layer_type == "head":
                self.num_ftrs = self.model.base_model.head.in_features
                self._set_last_layer()
                self.model.base_model.head = self.last_layer
            elif self.model_last_layer_type == "heads":
                self.num_ftrs = self.model.base_model.heads[-1].in_features
                self._set_last_layer()
                self.model.base_model.heads[-1] = self.last_layer

        if self.regularizer != 'none' and self.hook_handle is None:
            self.hook_handle = self.last_layer.register_forward_hook(self._cache_last_layer_input)

    def _cache_last_layer_input(self, module, inputs, output):
        self.cached_last_layer_input = inputs[0]

    def set_model(self):
        print(f'...setting model object...{self._time_string()}')
        self._set_base_model()
        self._swap_last_layer()
        print('here', self.task, self.at)
        if (self.task != 'train') or (self.at and self.task == 'train'):
            print('here', self.task != 'train', self.at)
            if self.task != 'train':
                pattern = os.path.join("models", f"{self.base_pattern}*.pth")     
            elif self.task == 'train' and self.at:
                pattern = os.path.join("models", f"{(self.base_pattern).replace('True','False')}*.pth")
            matching_files = glob.glob(pattern)
            state_dict = torch.load(matching_files[0], map_location=self.device)
            if self.normalize_model == 'before_load':
                self.model = NormalizedModel(self.model, self.norm_mean, self.norm_std)
            self.model.load_state_dict(state_dict)
            if self.normalize_model == 'after_load':
                self.model = NormalizedModel(self.model, self.norm_mean, self.norm_std)
        else:
            if self.normalize_model == 'normalize_wrapper':
                self.model = NormalizedModel(self.model, self.norm_mean, self.norm_std)

    def _save_model(self):
        model_path = f"models/{self.full_file_name}.pth"
        torch.save(self.model.state_dict(), model_path)
        print(f"...model saved to {model_path}...{self._time_string()}")

    def set_dataloader(self):
        print(f'...setting dataloader...{self._time_string()}')
        data_dir = "./data"     # Default data directory for downloaded datasets
        if self.dataset == "cifar10":
            self.norm_mean = (0.4914, 0.4822, 0.4465)
            self.norm_std = (0.2023, 0.1994, 0.2010)
            if self.split == 'train':
                transform = transforms.Compose([transforms.RandomCrop(32, padding=4),
                                                transforms.RandomHorizontalFlip(),
                                                transforms.AutoAugment(policy=transforms.AutoAugmentPolicy.CIFAR10),
                                                transforms.ToTensor()])
            else:
                transform = transforms.Compose([transforms.ToTensor()])
            dataset = datasets.CIFAR10(root=data_dir, train=(self.split == 'train'), download=True, transform=transform)
        elif self.dataset == "svhn":
            self.norm_mean = (0.4377, 0.4438, 0.4728)
            self.norm_std = (0.1980, 0.2010, 0.1970)
            if self.split == 'train':
                transform = transforms.Compose([transforms.RandomCrop(32, padding=4),
                                                transforms.RandomHorizontalFlip(),
                                                transforms.AutoAugment(policy=transforms.AutoAugmentPolicy.SVHN),
                                                transforms.ToTensor()])
            else:
                transform = transforms.Compose([transforms.ToTensor()])
            dataset = datasets.SVHN(root=data_dir, split=self.split, download=True, transform=transform)
        elif self.dataset == "cifar100":
            self.norm_mean = (0.5071, 0.4865, 0.4409)
            self.norm_std = (0.2673, 0.2564, 0.2761)
            if self.split == 'train':
                transform = transforms.Compose([transforms.RandomCrop(32, padding=4),
                                                transforms.RandomHorizontalFlip(),
                                                transforms.AutoAugment(policy=transforms.AutoAugmentPolicy.CIFAR10),
                                                transforms.ToTensor()])
            else:
                transform = transforms.Compose([transforms.ToTensor()])
            dataset = datasets.CIFAR100(root=data_dir, train=(self.split == 'train'), download=True, transform=transform)
        elif self.dataset == 'mnist':
            transform = transforms.Compose([transforms.Resize((32, 32)),transforms.ToTensor()])
            dataset = datasets.MNIST(root=data_dir, train=(self.split == "train"), download=True, transform=transform)
        else:
            raise ValueError(f"Unsupported dataset: {self.dataset}")  
        total_samples = len(dataset)
        slice_size = total_samples // self.total_slices
        start_idx = self.slice * slice_size
        end_idx = total_samples if self.slice == self.total_slices - 1 else start_idx + slice_size
        subset = Subset(dataset, list(range(start_idx, end_idx)))
        print(f'start index: {start_idx}, end index: {end_idx}')
        self.dataloader = DataLoader(subset, batch_size=self.batch_size, shuffle=False, num_workers=1, pin_memory=True)

    def _get_top1_top5_correct(self, outputs, labels):
        top1_preds = torch.max(outputs, 1).indices
        top5_preds = torch.topk(outputs, 5, dim=1).indices

        top1_correct = (top1_preds == labels).sum().item()
        top5_correct = sum(labels[i] in top5_preds[i] for i in range(outputs.size(0)))

        return top1_correct, top5_correct

    def evaluate_model(self):
        print(f'...evaluating model...{self._time_string()}')
        self.model = self.model.to(self.device)
        self.model.eval()

        top1_correct = 0
        top5_correct = 0
        total = 0
        self._write_to_csv('w', 'evaluate', ['clean top 1', 'clean top 5', 'total_evaluated'])
        with torch.no_grad():
            for images, labels in self.dataloader:
                images, labels = images.to(self.device), labels.to(self.device)
                outputs = self.model(images)
                batch_top1_correct, batch_top5_correct = self._get_top1_top5_correct(outputs, labels)
                top1_correct += batch_top1_correct
                top5_correct += batch_top5_correct
                total += labels.size(0)

        self.top1_accuracy = 100 * top1_correct / total
        self.top5_accuracy = 100 * top5_correct / total
        self.total_evaluated = total
        self._write_to_csv('a', 'evaluate', [self.top1_accuracy, self.top5_accuracy, self.total_evaluated])
        print(f"Accuracy: top1-{self.top1_accuracy:.2f}%, top5-{self.top5_accuracy:.2f}%")

    def _time_string(self):
        return time.strftime("%Y%m%d_%H%M%S",time.localtime())
    
    def _update_dict(self, top1, top5, norm, attack_name):
        self.correct[norm][attack_name]['top1'] += top1
        self.correct[norm][attack_name]['top5'] += top5

    def _write_to_csv(self, mode, task, list_data):
        os.makedirs(task, exist_ok=True)
        with open(f"{task}/{task}_{self.full_file_name}.csv", mode=mode, newline="") as file:
            writer = csv.writer(file)
            writer.writerow(list_data)

    def train_model(self):
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(self.model.parameters(), lr=self.lr)
        self.model.to(self.device)

        if self.regularizer != 'none':
            metric = 'l2' if self.regularizer == 'pcl_l2' else 'tropical'
            self.prox = Proximity(
                self.device,
                num_classes=self.num_classes,
                feat_dim=self.num_ftrs,
                distance_metric=metric,
            ).to(self.device)
            self.conprox = ConProximity(
                self.device,
                num_classes=self.num_classes,
                feat_dim=self.num_ftrs,
                distance_metric=metric,
            ).to(self.device)
            prox_optimizer = schedulefree.RAdamScheduleFree(self.prox.parameters(), lr=self.lr)
            conprox_optimizer = schedulefree.RAdamScheduleFree(self.conprox.parameters(), lr=self.lr)
            lambda_conprox = 1e-4

        if self.at:
            self.model.eval()
            print(f'...starting adversarial training loop...{self._time_string()}')
            at_adversary = APGDAttack(self.model, n_iter=10, eps=0.0, verbose=False, device=self.device)
            list_at_params = [('Linf', 4/255), ('L2', 0.5), ('L1', 1.5)]
        else:
            print(f'not adversarially training')
            t_max_input = self.num_epochs*2 if self.dataset == 'cifar100' and self.last_layer_type == 'tropical' else self.num_epochs
            print('scheduler max', t_max_input)
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=t_max_input) ### increase num_epochs schedule for tropical, cifar 100??

        self._write_to_csv('w', 'training', ["epoch", "total", "running_loss", "epoch_accuracy", "time"])

        self.model.train()
        for epoch in range(self.num_epochs):
            running_loss = 0.0
            correct = 0
            total = 0
            for images, labels in self.dataloader:
                if self.at:
                    list_x_advs = []
                    self.model.eval()
                    for norm, eps in list_at_params:
                        at_adversary.eps = eps
                        at_adversary.norm = norm
                        x_adv = at_adversary.perturb(images, labels)
                        list_x_advs.append(x_adv)
                    images = torch.cat(list_x_advs, dim=0)
                    labels = torch.cat([labels for _ in range(len(list_at_params))], dim=0)
                    self.model.train()
                images, labels = images.to(self.device), labels.to(self.device)
                optimizer.zero_grad()
                if self.regularizer != 'none':
                    prox_optimizer.zero_grad()
                    conprox_optimizer.zero_grad()
                outputs = self.model(images)
                if self.regularizer != 'none':
                    feats = self.cached_last_layer_input
                    ce_loss = criterion(outputs, labels)
                    prox_loss = self.prox(feats, labels)
                    conprox_loss = self.conprox(feats, labels)
                    loss = ce_loss + prox_loss - lambda_conprox * conprox_loss
                else:
                    loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()
                if self.regularizer != 'none':
                    prox_optimizer.step()
                    conprox_optimizer.step()
                running_loss += loss.item()
                predicted = torch.max(outputs, 1).indices
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
                self._write_to_csv('a', 'training', [str(epoch), str(total), running_loss / len(self.dataloader), 100*correct/total, time.time()])
            if not self.at:
                scheduler.step()
            print(f"Epoch {epoch+1}/{self.num_epochs}, Loss: {running_loss/len(self.dataloader):.4f}, Acc: {100*correct/total:.4f}")
        self._save_model()

    def attack_model(self):
        self.model = self.model.to(self.device)
        self.model.eval()

        # ---- setup adversaries ----
        aa_linf_adversary = AutoAttack(self.model, norm="Linf", eps=0.1 if self.dataset == 'mnist' else 4/255, version='standard', device=self.device)
        aa_l2_adversary = AutoAttack(self.model, norm="L2", eps=2 if self.dataset == 'mnist' else 0.5, version='standard', device=self.device)
        aa_l1_adversary = AutoAttack(self.model, norm="L1", eps=6 if self.dataset == 'mnist' else 1.5, version='standard', device=self.device)
        tup_list_aa = [('Linf',aa_linf_adversary), ('L2',aa_l2_adversary), ('L1',aa_l1_adversary)]
        cw_adverary = CarliniWagnerL2Attack(self.model,
                                            num_classes=self.num_classes,
                                            confidence=0,
                                            targeted=False,
                                            learning_rate=0.01,
                                            binary_search_steps=5,
                                            max_iterations=500,
                                            abort_early=True)
        spsa_adversary = LinfSPSAAttack(self.model, 
                                        eps=0.1 if self.dataset == 'mnist' else 4/255, 
                                        delta=0.01, 
                                        lr=0.01, 
                                        nb_iter=50, 
                                        nb_sample=64, 
                                        max_batch_size=128)
        
        # ---- setup counters ----
        self.correct = {
        'Linf':
        {
            'spsa':    {"top1": 0, "top5": 0},
            'apgd-ce': {"top1": 0, "top5": 0},
            'apgd-t':  {"top1": 0, "top5": 0},
            'fab-t':   {"top1": 0, "top5": 0},
            'square':  {"top1": 0, "top5": 0},
            'aa_total':   {"top1": 0, "top5": 0}
        },
        'L2':
        {
            'cw':      {"top1": 0, "top5": 0},
            'apgd-ce': {"top1": 0, "top5": 0},
            'apgd-t':  {"top1": 0, "top5": 0},
            'fab-t':   {"top1": 0, "top5": 0},
            'square':  {"top1": 0, "top5": 0},
            'aa_total':   {"top1": 0, "top5": 0}
        },
        'L1':
        {
            'apgd-ce': {"top1": 0, "top5": 0},
            'apgd-t':  {"top1": 0, "top5": 0},
            'fab-t':   {"top1": 0, "top5": 0},
            'square':  {"top1": 0, "top5": 0},
            'aa_total':   {"top1": 0, "top5": 0}
        }}
        total_samples = 0
        attack_order = ['apgd-ce', 'apgd-t', 'fab-t', 'square']
        cw_total_l2 = 0.0
        cw_success_count = 0
        header = (
            ["cw avg_l2", "cw success count", "total_samples"] +
            [f"{norm} {attack} {metric}" for norm in self.correct.keys() for attack in self.correct[norm].keys() for metric in ["top1", "top5"]] + 
            [self._time_string()]
        )
        self._write_to_csv('w', 'attack', header)
        # ---- ATTACK!! ----
        with torch.no_grad():
            for images, labels in self.dataloader:
                # ---- setup ----
                images, labels = images.to(self.device), labels.to(self.device)
                batch_size = labels.size(0)
                total_samples += batch_size

                # ---- SPSA ----
                print(f'...starting SPSA attack...{self._time_string()}')
                x_adv_spsa = spsa_adversary(images, labels)
                spsa_top1, spsa_top5 = self._get_top1_top5_correct(self.model(x_adv_spsa), labels)
                self._update_dict(spsa_top1, spsa_top5, 'Linf', 'spsa') 

                # ---- CW ----
                print(f'...starting Carlini and Wagner Attack...{self._time_string()}')
                # --- accuracy ---
                with torch.enable_grad():
                    x_adv_cw = cw_adverary(images, labels)
                outputs_cw = self.model(x_adv_cw)
                cw_top1, cw_top5 = self._get_top1_top5_correct(outputs_cw, labels)
                self._update_dict(cw_top1, cw_top5, 'L2', 'cw') 
                # --- l2 perturbation needed ---
                preds_cw = torch.max(outputs_cw, 1).indices
                successful_mask = (preds_cw != labels)
                if successful_mask.sum() > 0:
                    perturbations = (x_adv_cw - images).view(batch_size, -1)
                    l2_distances = torch.norm(perturbations, p=2, dim=1)
                    cw_total_l2 += l2_distances[successful_mask].sum().item()
                    cw_success_count += successful_mask.sum().item()

                # ---- AutoAttack ---
                for norm, adversary in tup_list_aa:
                    print(f'...starting {norm} AutoAttack attack...{self._time_string()}')
                    adversary.verbose = True
                    adversary.attacks_to_run = attack_order
                    x_adv_dict = adversary.run_standard_evaluation_individual(images, labels, bs=batch_size, return_labels=False)
                    for attack, x_adv in x_adv_dict.items():
                        aa_top1, aa_top5 = self._get_top1_top5_correct(self.model(x_adv), labels)
                        self._update_dict(aa_top1, aa_top5, norm, attack)
                    x_adv_total = images.clone()
                    outputs_total = self.model(x_adv_total)
                    preds_total = torch.max(outputs_total, 1).indices
                    for attack in attack_order:
                        if attack not in x_adv_dict:
                            print(f"Warning: {attack} not found in adversarial examples dictionary. Skipping attack.")
                            continue
                        robust_mask = (preds_total == labels)
                        if robust_mask.sum() == 0:
                            break  
                        x_adv_candidate = x_adv_dict[attack]
                        outputs_candidate = self.model(x_adv_candidate)
                        preds_candidate = torch.max(outputs_candidate, 1).indices
                        successful_mask = robust_mask & (preds_candidate != labels)
                        if successful_mask.sum() > 0:
                            x_adv_total[successful_mask] = x_adv_candidate[successful_mask]  
                            outputs_total = self.model(x_adv_total)
                            preds_total = torch.max(outputs_total, 1).indices
                    aa_top1, aa_top5 = self._get_top1_top5_correct(self.model(x_adv_total), labels)
                    self._update_dict(aa_top1, aa_top5, norm, 'aa_total')
        
                results = {}
                for norm in self.correct.keys():
                    results[norm] = {}
                    for attack in self.correct[norm].keys():
                        results[norm][attack] = {
                            'top1': 100.0 * self.correct[norm][attack]['top1'] / total_samples,
                            'top5': 100.0 * self.correct[norm][attack]['top5'] / total_samples
                        }
                avg_cw_l2 = cw_total_l2 / cw_success_count if cw_success_count > 0 else 0
                data_row = (
                    [avg_cw_l2, cw_success_count, total_samples] +
                    [results[norm][attack][metric] for norm in results.keys() for attack in results[norm].keys() for metric in ["top1", "top5"]] + 
                    [self._time_string()]
                )
                self._write_to_csv('a', 'attack', data_row)
        
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--job_file", type=str, default='jobs_to_do', help="job file to read")
    parser.add_argument("--job_id", type=int, default=0, help="job_id index to select model, dataset, and layer type")
    parser.add_argument("--slice", type=int, default=0, help="batch index")
    parser.add_argument("--total_slices", type=int, default=1, help="number_batches")
    parser.add_argument("--regularizer", type=str, choices=['none', 'pcl_l2', 'pcl_tropical'], default=None,
                        help="regularizer to apply")
    args = parser.parse_args()

    with open(f'{args.job_file}.csv', 'r') as file:
        csv_reader = csv.reader(file)
        for row in csv_reader:
            if row[0] != 'job_id':
                if int(row[0]) == int(args.job_id):
                    task = row[1]
                    dataset = row[2]
                    model_name = row[3]
                    last_layer_name = row[4]
                    regularizer = row[5]
                    at = True if row[6] == 'yes' else False
                    num_epochs = int(row[7])
                    batch_size = int(row[8])
                    normalize_model = row[9]
                    learning_rate = float(row[10])
                    print(row)

    if args.regularizer is not None:
        regularizer = args.regularizer

    experiment = Experiment(task,
                 at,
                 model_name,
                 dataset,
                 last_layer_name,
                 torch.device("cuda" if torch.cuda.is_available() else "cpu"),
                 regularizer=regularizer,
                 num_epochs = num_epochs,
                 lr = learning_rate,
                 slice = args.slice,
                 total_slices = args.total_slices,
                 batch_size = batch_size,
                 normalize_model = normalize_model)
    experiment.run()
