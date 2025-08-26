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
from autoattack import AutoAttack
from autoattack.autopgd_base import APGDAttack
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Subset
from resnet_model import *
import schedulefree
from pcl_regularizer import Proximity, ConProximity


class SixConvNet(nn.Module):
    def __init__(self, in_channels=1, num_classes=10):
        super().__init__()
        self.block1 = nn.Sequential(
            nn.Conv2d(in_channels,  32, kernel_size=5, padding=2),
            nn.PReLU(num_parameters=32),
            nn.Conv2d(32,            32, kernel_size=5, padding=2),
            nn.PReLU(num_parameters=32),
            nn.MaxPool2d(2),
        )
        self.block2 = nn.Sequential(
            nn.Conv2d(32,            64, kernel_size=5, padding=2),
            nn.PReLU(num_parameters=64),
            nn.Conv2d(64,            64, kernel_size=5, padding=2),
            nn.PReLU(num_parameters=64),
            nn.MaxPool2d(2),
        )
        self.block3 = nn.Sequential(
            nn.Conv2d(64,           128, kernel_size=5, padding=2),
            nn.PReLU(num_parameters=128),
            nn.Conv2d(128,          128, kernel_size=5, padding=2),
            nn.PReLU(num_parameters=128),
            nn.MaxPool2d(2),
        )

        self.regularize = False
        self.gap = nn.AdaptiveAvgPool2d(1)
        self.fc1 = nn.Linear(128, 512)
        self.prelu_fc1 = nn.PReLU()
        self.fc2 = nn.Linear(512,  64)
        self.prelu_fc2 = nn.PReLU()
        self.fc3 = nn.Linear( 64,  num_classes)

    def forward(self, x):
        x = self.block1(x)   # → [B,32,14,14] then → [B,32,7,7]
        x = self.block2(x)   # → [B,64,3,3] after two pools
        x = self.block3(x)   # → [B,128,1,1] after two more pools
        feats_1 = self.gap(x).view(x.size(0), -1)
        feats_2 = self.prelu_fc1(self.fc1(feats_1))
        feats_3 = self.prelu_fc2(self.fc2(feats_2))
        if self.training:
            return feats_1, feats_2, feats_3, self.fc3(feats_3)
        else:
            return self.fc3(feats_3)


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


class Experiment:
    def __init__(self,
                 task,
                 at,
                 model_name,
                 dataset,
                 device,
                 regularizer,
                 warmup_epochs = 50,
                 num_epochs = 50,
                 lr = 0.001,
                 lr_prox = 0.001,
                 lr_conprox = 0.001,
                 slice = 0,
                 total_slices = 1,
                 batch_size = 100):
        # -- assertions -- 
        assert task in ['attack', 'evaluate', 'train']
        assert regularizer in ['none', 'vanilla', 'tropical']
        assert model_name in ['cnn6', 'resnet']
        assert type(at) is bool

        # -- essentials --
        self.task = task
        self.at = at
        print(self.at)
        self.model_name = model_name
        self.feat_nums = [128, 256, 1024] if self.model_name == 'resnet' else [128, 512, 64]
        self.dataset = dataset
        self.regularizer = regularizer
        self.channels = 1 if dataset == 'mnist' or dataset == 'fmnist' else 3
        self.split = 'train' if self.task == 'train' else 'test'
        
        # -- data --
        dict_num_classes = {"cifar100":100, "cifar10":10, "svhn":10, 'mnist':10, 'fmnist':10}
        self.num_classes = dict_num_classes[dataset]
        self.slice = slice
        self.total_slices = total_slices
        self.warmup_epochs = warmup_epochs
        self.num_epochs = num_epochs
        self.lr = lr
        self.lr_prox = lr_prox
        self.lr_conprox = lr_conprox
        self.batch_size = batch_size

        # -- tracking --
        self.device = device
        self.init_time = self._time_string()
        self.base_pattern = f"{self.at}_{self.dataset}_{self.model_name}_{self.regularizer}_"  
        self.full_file_name = f"{self.base_pattern}{self.lr}_{self.num_epochs}_{self.init_time}"
        print(self.full_file_name)

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
        if self.model_name == 'cnn6':
            self.model = SixConvNet(in_channels=self.channels, num_classes=self.num_classes)
        elif self.model_name == "resnet":
            self.model = resnet(num_classes=self.num_classes,depth=110)
        else:
            raise ValueError(f"Unsupported model: {self.model_name}")
    
    def set_model(self):
        print(f'...setting model object...{self._time_string()}')
        self._set_base_model()
        if self.task != 'train':
            self._load_model()

    def _load_model(self):
        pattern = os.path.join("models", f"{self.base_pattern}*.pth")
        matching_files = glob.glob(pattern)
        state_dict = torch.load(matching_files[0], map_location=self.device)
        self.model_being_evaluated = matching_files[0].replace('.pth','').replace('models\\','') # change \\ back to /
        print(self.model_being_evaluated)
        self.model.load_state_dict(state_dict)

    def _save_model(self):
        os.makedirs("models", exist_ok=True)
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
        elif self.dataset == 'fmnist':
            transform = transforms.Compose([transforms.Resize((32, 32)),transforms.ToTensor()])
            dataset = datasets.FashionMNIST(root=data_dir, train=(self.split == "train"), download=True, transform=transform)
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
        c_CE = nn.CrossEntropyLoss()
        o_CE = schedulefree.RAdamScheduleFree(self.model.parameters(), lr=self.lr)
        o_CE.train()
        if self.regularizer != 'none':
            # set loss criterions
            c_prox_1    = Proximity(device = self.device, num_classes=self.num_classes, feat_dim=self.feat_nums[0], distance_metric=self.regularizer if self.regularizer == 'tropical' else 'l2')
            c_conprox_1 = ConProximity(device = self.device, num_classes=self.num_classes, feat_dim=self.feat_nums[0], distance_metric=self.regularizer if self.regularizer == 'tropical' else 'l2')
            c_prox_2    = Proximity(device = self.device, num_classes=self.num_classes, feat_dim=self.feat_nums[1], distance_metric=self.regularizer if self.regularizer == 'tropical' else 'l2')
            c_conprox_2 = ConProximity(device = self.device, num_classes=self.num_classes, feat_dim=self.feat_nums[1], distance_metric=self.regularizer if self.regularizer == 'tropical' else 'l2')
            c_prox_3    = Proximity(device = self.device, num_classes=self.num_classes, feat_dim=self.feat_nums[2], distance_metric=self.regularizer if self.regularizer == 'tropical' else 'l2')
            c_conprox_3 = ConProximity(device = self.device, num_classes=self.num_classes, feat_dim=self.feat_nums[2], distance_metric=self.regularizer if self.regularizer == 'tropical' else 'l2')
        
            # set optimizers
            o_prox_1    = schedulefree.RAdamScheduleFree(c_prox_1.parameters(), lr=self.lr_prox)
            o_conprox_1 = schedulefree.RAdamScheduleFree(c_conprox_1.parameters(), lr=self.lr_conprox)
            o_prox_2    = schedulefree.RAdamScheduleFree(c_prox_2.parameters(), lr=self.lr_prox)
            o_conprox_2 = schedulefree.RAdamScheduleFree(c_conprox_2.parameters(), lr=self.lr_conprox)
            o_prox_3    = schedulefree.RAdamScheduleFree(c_prox_3.parameters(), lr=self.lr_prox)
            o_conprox_3 = schedulefree.RAdamScheduleFree(c_conprox_3.parameters(), lr=self.lr_conprox)

            o_prox_1.train()
            o_conprox_1.train()
            o_prox_2.train()
            o_conprox_2.train()
            o_prox_3.train()
            o_conprox_3.train()
        # place model on device
        self.model.to(self.device)
        
        if self.at:
            self.model.eval()
            print(f'...starting adversarial training loop...{self._time_string()}')
            at_adversary = APGDAttack(self.model, n_iter=10, eps=0.0, verbose=False, device=self.device)
            list_at_params = [('Linf', 8/255), ('L2', 0.5), ('L1', 1.5)]
        else:
            print(f'not adversarially training')
            
        self._write_to_csv('w', 'training', ["epoch", "total", "CE loss (minimize)", "Prox loss (minimize)", "ConProx loss (maximize)", "epoch_accuracy", "time"])

        self.model.train()
        for epoch in range(self.num_epochs + self.warmup_epochs):
            running_loss = 0.0
            l_prox = torch.tensor([0])
            l_conprox = torch.tensor([0])
            correct = 0
            total = 0
            if self.regularizer != 'none' and epoch > self.warmup_epochs:
                self.model.regularize = True
            for i_batch, (images, labels) in enumerate(self.dataloader):
                images, labels = images.to(self.device), labels.to(self.device)
                if self.at and epoch > self.warmup_epochs:
                    list_x_advs = [images]
                    self.model.eval()
                    for norm, eps in list_at_params:  
                        at_adversary.eps = eps
                        at_adversary.norm = norm
                        x_adv = at_adversary.perturb(images, labels)
                        x_adv = x_adv.to(self.device) 
                        list_x_advs.append(x_adv)
                    images = torch.cat(list_x_advs, dim=0)
                    labels = torch.cat([labels for _ in range(len(list_at_params)+1)], dim=0)
                    self.model.train()
                feats_1, feats_2, feats_3, outputs = self.model(images)
                if self.regularizer != 'none' and epoch > self.warmup_epochs:
                    # loss computations
                    l_CE = c_CE(outputs, labels)

                    l_prox_1 = c_prox_1(feats_1, labels) 
                    l_conprox_1 = c_conprox_1(feats_1, labels) 

                    l_prox_2 = c_prox_2(feats_2, labels) 
                    l_conprox_2 = c_conprox_2(feats_2, labels) 

                    l_prox_3 = c_prox_3(feats_3, labels) 
                    l_conprox_3 = c_conprox_3(feats_3, labels) 

                    l_prox = l_prox_1 + l_prox_2 + l_prox_3
                    l_conprox = l_conprox_1 + l_conprox_2 + l_conprox_3
                    loss = l_CE + l_prox - l_conprox * 0.0001

                    # update weights
                    o_CE.zero_grad()
                    o_prox_1.zero_grad()
                    o_conprox_1.zero_grad()
                    o_prox_2.zero_grad()
                    o_conprox_2.zero_grad()
                    o_prox_3.zero_grad()
                    o_conprox_3.zero_grad()
                    loss.backward()
                    o_CE.step()
                    o_prox_1.step()
                    o_conprox_1.step()
                    o_prox_2.step()
                    o_conprox_2.step()
                    o_prox_3.step()
                    o_conprox_3.step()
                else:
                    l_CE = c_CE(outputs, labels)
                    loss = l_CE
                    o_CE.zero_grad()
                    loss.backward()
                    o_CE.step()

                #track data
                running_loss += loss.item()
                predicted = torch.max(outputs, 1).indices
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
                if i_batch % 5 == 0:
                    self._write_to_csv('a', 'training', [str(epoch), str(total), l_CE.item(), l_prox.item(), l_conprox.item(), 100*correct/total, time.time()])
            print(f"Epoch {epoch+1}/{self.num_epochs+self.warmup_epochs}, Loss: {running_loss/len(self.dataloader):.4f}, Acc: {100*correct/total:.4f}")
        self._save_model()

    def attack_model(self):
        self.model = self.model.to(self.device)
        self.model.eval()

        # ---- setup adversaries ----
        aa_linf_adversary = AutoAttack(self.model, norm="Linf", eps=0.1 if self.dataset in ['mnist','fmnist']  else 8/255, version='standard', device=self.device)
        aa_l2_adversary = AutoAttack(self.model, norm="L2", eps=2 if self.dataset in ['mnist','fmnist'] else 0.5, version='standard', device=self.device)
        aa_l1_adversary = AutoAttack(self.model, norm="L1", eps=6 if self.dataset in ['mnist','fmnist'] else 1.5, version='standard', device=self.device)
        tup_list_aa = [('Linf',aa_linf_adversary), ('L2',aa_l2_adversary), ('L1',aa_l1_adversary)]
        
        # ---- setup counters ----
        self.correct = {
        'Linf':
        {
            'apgd-ce': {"top1": 0, "top5": 0},
            'apgd-t':  {"top1": 0, "top5": 0},
            'fab-t':   {"top1": 0, "top5": 0},
            'square':  {"top1": 0, "top5": 0},
            'aa_total':   {"top1": 0, "top5": 0}
        },
        'L2':
        {
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
        header = (
            ["total_samples"] +
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

                # ---- AutoAttack ---
                for norm, adversary in tup_list_aa:
                    print(f'...starting {norm} AutoAttack attack...{self._time_string()}')
                    adversary.verbose = True
                    adversary.attacks_to_run = attack_order
                    x_adv_dict = adversary.run_standard_evaluation_individual(images, labels, bs=batch_size, return_labels=False)
                    for attack, x_adv in x_adv_dict.items():
                        outputs = self.model(x_adv)
                        aa_top1, aa_top5 = self._get_top1_top5_correct(outputs, labels)
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
                    outputs = self.model(x_adv_total)
                    aa_top1, aa_top5 = self._get_top1_top5_correct(outputs, labels)
                    self._update_dict(aa_top1, aa_top5, norm, 'aa_total')
        
                results = {}
                for norm in self.correct.keys():
                    results[norm] = {}
                    for attack in self.correct[norm].keys():
                        results[norm][attack] = {
                            'top1': 100.0 * self.correct[norm][attack]['top1'] / total_samples,
                            'top5': 100.0 * self.correct[norm][attack]['top5'] / total_samples
                        }

                data_row = (
                    [total_samples] +
                    [results[norm][attack][metric] for norm in results.keys() for attack in results[norm].keys() for metric in ["top1", "top5"]] + 
                    [self._time_string()]
                )
                self._write_to_csv('a', 'attack', data_row)
        
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--job_file", type=str, default='jobs_training', help="job file to read")
    parser.add_argument("--job_id", type=int, default=19, help="job_id index to select model, dataset, and layer type")
    parser.add_argument("--slice", type=int, default=0, help="batch index")
    parser.add_argument("--total_slices", type=int, default=1, help="number_batches")
    args = parser.parse_args()
    
    with open(f'{args.job_file}.csv', 'r') as file:
        csv_reader = csv.reader(file)
        for row in csv_reader:
            if row[0] != 'job_id':
                if int(row[0]) == int(args.job_id):
                    task = row[1]
                    dataset = row[2]
                    model_name = row[3]
                    regularizer = row[4]
                    at = True if row[5] == 'yes' else False
                    warmup_epochs = int(row[6])
                    num_epochs = int(row[7])
                    batch_size = int(row[8])
                    learning_rate = float(row[9])
                    learning_rate_prox = float(row[10])
                    learning_rate_conprox = float(row[11])
                    print(row)
    experiment = Experiment(task = task,
                            at = at,
                            model_name = model_name,
                            dataset = dataset,
                            regularizer = regularizer,
                            device = torch.device("cuda" if torch.cuda.is_available() else "cpu"),
                            warmup_epochs = warmup_epochs,
                            num_epochs = num_epochs,
                            lr = learning_rate,
                            lr_prox = learning_rate_prox,
                            lr_conprox = learning_rate_conprox,
                            slice = args.slice,
                            total_slices = args.total_slices,
                            batch_size = batch_size)
    experiment.run()
