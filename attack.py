#!/usr/bin/env python
import argparse
import os
import glob
import torch
import math
import csv
import time
import torch.nn as nn
from advertorch.attacks import CarliniWagnerL2Attack, LinfSPSAAttack 
from autoattack import AutoAttack
from oldish_train import get_model, swap_last_layer, NormalizedModel
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Subset

class DataContainer:
    def __init__(self,
                 model_name,
                 dataset,
                 last_layer,
                 slice,
                 total_slices,
                 device,
                 time):
        self.model_name = model_name
        self.dataset = dataset
        self.last_layer = last_layer
        dict_num_classes = {"imagenet":1000,
                    "cifar100":100,
                    "cifar10":10,
                    "svhn":10,
                    "mnist":10,
                    "fmnist":10}
        self.num_classes = dict_num_classes[dataset]
        self.slice = slice
        self.total_slices = total_slices
        self.device = device
        self.time = time  
        self.base_pattern = f"{self.dataset}_{self.model_name}_{self.last_layer}_"   


def get_dataloader_slice(data):
    data_dir = "./data"     # Default data directory for downloaded datasets
    
    if data.dataset == "imagenet":
        norm_mean = (0.485, 0.456, 0.406)
        norm_std = (0.229, 0.224, 0.225)
        transform = transforms.Compose([transforms.Resize(256),
                                        transforms.CenterCrop(224),
                                        transforms.ToTensor()])
        dataset = datasets.ImageNet(root="imagenet_files", split='val', transform=transform)
    elif data.dataset == "mnist":
        norm_mean, norm_std = (0.1307,), (0.3081,)
        transform = transforms.Compose([transforms.ToTensor()])
        dataset = datasets.MNIST(root=data_dir, train=False, download=True, transform=transform)
    elif data.dataset == "fmnist":
        norm_mean, norm_std = (0.2860,), (0.3530,)
        transform = transforms.Compose([transforms.ToTensor()])
        dataset = datasets.FashionMNIST(root=data_dir, train=False, download=True, transform=transform)
    elif data.dataset == "cifar10":
        norm_mean = (0.4914, 0.4822, 0.4465)
        norm_std = (0.2023, 0.1994, 0.2010)
        transform = transforms.Compose([transforms.ToTensor()])
        dataset = datasets.CIFAR10(root=data_dir, train=False, download=True, transform=transform)
    elif data.dataset == "svhn":
        norm_mean = (0.4377, 0.4438, 0.4728)
        norm_std = (0.1980, 0.2010, 0.1970)
        transform = transforms.Compose([transforms.ToTensor()])
        dataset = datasets.SVHN(root=data_dir, split='test', download=True, transform=transform)
    elif data.dataset == "cifar100":
        norm_mean = (0.5071, 0.4865, 0.4409)
        norm_std = (0.2673, 0.2564, 0.2761)
        transform = transforms.Compose([transforms.ToTensor()])
        dataset = datasets.CIFAR100(root=data_dir, train=False, download=True, transform=transform)
    else:
        raise ValueError(f"Unsupported dataset: {data.dataset}")  
    total_samples = len(dataset)
    slice_size = total_samples // data.total_slices
    start_idx = data.slice * slice_size
    end_idx = total_samples if data.slice == data.total_slices - 1 else start_idx + slice_size
    subset = Subset(dataset, list(range(start_idx, end_idx)))
    print(f'start index: {start_idx}, end index: {end_idx}')
    return DataLoader(subset, batch_size=50, shuffle=False, num_workers=1, pin_memory=True), norm_mean, norm_std


def evaluate_model(model, dataloader, device):
    model = model.to(device)
    model.eval()
    top1_correct = 0
    top5_correct = 0
    total = 0

    with torch.no_grad():
        for images, labels in dataloader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, top1_predicted = torch.max(outputs, 1)
            top1_correct += (top1_predicted == labels).sum().item()
            _, top5_predicted = torch.topk(outputs, 5, dim=1)
            top5_correct += sum(labels[i] in top5_predicted[i] for i in range(labels.size(0)))
            total += labels.size(0)

    top1_accuracy = 100 * top1_correct / total
    top5_accuracy = 100 * top5_correct / total
    print(f"Accuracy: top1-{top1_accuracy:.2f}%, top5-{top5_accuracy:.2f}%")
    return top1_accuracy, top5_accuracy


def update_dict(dict_results, norm, attack_name, outputs, labels):
    top1_preds = torch.max(outputs, 1).indices
    dict_results[norm][attack_name]['top1'] += (top1_preds == labels).sum().item()
    top5_preds = torch.topk(outputs, 5, dim=1).indices
    dict_results[norm][attack_name]['top5'] += sum(labels[i] in top5_preds[i] for i in range(outputs.size(0)))
    return dict_results


def attack_model_indiv_attacks(model, dataloader, device, num_classes, data, epsilon=4/255):
    model = model.to(device)
    model.eval()
    # ---- setup adversaries ----
    aa_linf_adversary = AutoAttack(model, norm="Linf", eps=4/255, version='standard', device=device)
    aa_l2_adversary = AutoAttack(model, norm="L2", eps=0.5, version='standard', device=device)
    aa_l1_adversary = AutoAttack(model, norm="L1", eps=1.5, version='standard', device=device)
    tup_list_aa = [('Linf',aa_linf_adversary),
                   ('L2',aa_l2_adversary),
                   ('L1',aa_l1_adversary)]
    cw_adverary = CarliniWagnerL2Attack(model,
                                        num_classes=num_classes,
                                        confidence=0,
                                        targeted=False,
                                        learning_rate=0.01,
                                        binary_search_steps=9,
                                        max_iterations=1000,
                                        abort_early=True)
    spsa_adversary = LinfSPSAAttack(model, 
                                    eps=epsilon, 
                                    delta=0.01, 
                                    lr=0.01, 
                                    nb_iter=100, 
                                    nb_sample=128, 
                                    max_batch_size=128)
    
    # ---- setup counters ----
    correct = {
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
    csv_filename = f"prints/attacks/attacking_just_aa_{data.base_pattern}.csv"
    with open(csv_filename, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(["Message", "Total_Samples"])  # Define the single column header
    # ---- ATTACK!! ----
    with torch.no_grad():
        for images, labels in dataloader:
            # ---- setup ----
            images, labels = images.to(device), labels.to(device)
            batch_size = labels.size(0)
            total_samples += batch_size

            # ---- SPSA ----
            '''print(f'Starting SPSA attack...{time.strftime("%Y%m%d_%H%M%S",time.localtime())}')
            x_adv_spsa = spsa_adversary(images, labels)
            correct = update_dict(correct, 'Linf', 'spsa' ,model(x_adv_spsa), labels)

            # ---- CW ----
            print(f'Starting Carlini and Wagner Attack...{time.strftime("%Y%m%d_%H%M%S",time.localtime())}')
            with torch.enable_grad():
                x_adv_cw = cw_adverary(images, labels)
            
            outputs_cw = model(x_adv_cw)
            correct = update_dict(correct, 'L2', 'cw', outputs_cw, labels)
            preds_cw = torch.max(outputs_cw, 1).indices
            successful_mask = (preds_cw != labels)
            if successful_mask.sum() > 0:
                perturbations = (x_adv_cw - images).view(batch_size, -1)
                l2_distances = torch.norm(perturbations, p=2, dim=1)
                cw_total_l2 += l2_distances[successful_mask].sum().item()
                cw_success_count += successful_mask.sum().item()'''

            # ---- AutoAttack ---
            for norm, adversary in tup_list_aa:
                text = f'Starting {norm} AutoAttack attack...{time.strftime("%Y%m%d_%H%M%S",time.localtime())}'
                with open(csv_filename, mode='a', newline='') as file:
                    writer = csv.writer(file)
                    writer.writerow([text, total_samples])  # Writing a single column
                print(text)
                adversary.verbose = True
                adversary.attacks_to_run = attack_order
                x_adv_dict = adversary.run_standard_evaluation_individual(images, labels, bs=batch_size, return_labels=False)
                for attack, x_adv in x_adv_dict.items():
                    correct = update_dict(correct, norm, attack, model(x_adv), labels)
                x_adv_total = images.clone()
                outputs_total = model(x_adv_total)
                preds_total = torch.max(outputs_total, 1).indices
                for attack in attack_order:
                    if attack not in x_adv_dict:
                        print(f"Warning: {attack} not found in adversarial examples dictionary. Skipping attack.")
                        continue
                    robust_mask = (preds_total == labels)
                    if robust_mask.sum() == 0:
                        break  
                    x_adv_candidate = x_adv_dict[attack]
                    outputs_candidate = model(x_adv_candidate)
                    preds_candidate = torch.max(outputs_candidate, 1).indices
                    successful_mask = robust_mask & (preds_candidate != labels)
                    if successful_mask.sum() > 0:
                        x_adv_total[successful_mask] = x_adv_candidate[successful_mask]  
                        outputs_total = model(x_adv_total)
                        preds_total = torch.max(outputs_total, 1).indices
                correct = update_dict(correct, norm, 'aa_total', model(x_adv_total), labels)
    results = {}
    for norm in correct.keys():
        results[norm] = {}
        for attack in correct[norm].keys():
            results[norm][attack] = {
                'top1': 100.0 * correct[norm][attack]['top1'] / total_samples,
                'top5': 100.0 * correct[norm][attack]['top5'] / total_samples
            }
            print(f"{attack} Adversarial Accuracy: top1-{results[norm][attack]['top1']:.2f}%, top5-{results[norm][attack]['top5']:.2f}%")
    avg_cw_l2 = cw_total_l2 / cw_success_count if cw_success_count > 0 else 0
    return results, avg_cw_l2, cw_success_count, total_samples

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--combo", type=int, default=1, help="Combo index to select model, dataset, and layer type")
    parser.add_argument("--att_index", type=int, default=0, help="batch index")
    parser.add_argument("--att_total", type=int, default=1, help="number_batches")
    args = parser.parse_args()
    
    list_models = ["mobilenet_v3_small",  #0
                    "vgg16",               #1
                    "mobilenet_v2",        #2
                    "resnet50",            #3
                    "alexnet",             #4
                    "densenet121",         #5
                    "convnext_tiny",       #6
                    "efficientnet_b0",     #7
                    "efficientnet_v2_s",   #8
                    "googlenet",           #9
                    "inception_v3",        #10
                    "maxvit_t",            #11
                    "mnasnet0_5",          #12
                    "regnet_y_1_6gf",      #13
                    "resnext50_32x4d",     #14
                    "shufflenet_v2_x0_5",   #15
                    "swin_v2_b",           #16
                    "vit_b_16",            #17
                    "wide_resnet50_2"]     #18
    list_datasets = ["cifar100", "cifar10", "svhn"]
    list_last_layers = ["tropical", "normal"]

    data = DataContainer(model_name=list_models[math.floor(args.combo / 6)],
                 dataset=list_datasets[args.combo % 3],
                 last_layer=list_last_layers[args.combo % 2],
                 slice=args.att_index,
                 total_slices=args.att_total,
                 device=torch.device("cuda" if torch.cuda.is_available() else "cpu"),
                 time=time.strftime("%Y%m%d_%H%M%S",time.localtime()))
    print(data.time)
    pattern = os.path.join("models", f"{data.base_pattern}*.pth")
    matching_files = glob.glob(pattern)
    if not matching_files:
        print(f"No model file matching the pattern: {pattern}\n--EXITING--")
    elif len(matching_files) > 1:
        print("Warning: More than one matching file found.\n--EXITING--")
    else:
        print(matching_files[0])
        print("Loading dataset...")
        val_loader, norm_mean, norm_std = get_dataloader_slice(data)

        print("Loading model...")
        model, model_last_layer_name = get_model(data.model_name)
        model = swap_last_layer(model, model_last_layer_name, data.last_layer, data.num_classes)
        state_dict = torch.load(matching_files[0], map_location=data.device)
        model.load_state_dict(state_dict)
        model = NormalizedModel(model, norm_mean, norm_std)
        model.eval()

        
        print("Evaluating model...")
        top1_clean, top5_clean = evaluate_model(model, val_loader, data.device)

        dir_path = f"results/{data.base_pattern}"
        os.makedirs(dir_path, exist_ok=True)
        output_filename = os.path.basename(matching_files[0]).replace('.pth', '.csv')
        output_path = os.path.join(dir_path, f"{data.slice}_of_{data.total_slices}_{output_filename}")

        print("Running adversarial attack...")
        dict_results, cw_avg_l2, cw_success_count, total_samples = attack_model_indiv_attacks(model, val_loader, data.device, data.num_classes, data, epsilon=4/255)
            
        header = (
            ["clean top1", "clean top5", "cw avg_l2", "cw success count", "total_samples"] +
            [f"{norm} {attack} {metric}" for norm in dict_results.keys() for attack in dict_results[norm].keys() for metric in ["top1", "top5"]]
        )

        data_row = (
            [top1_clean, top5_clean, cw_avg_l2, cw_success_count, total_samples] +
            [dict_results[norm][attack][metric] for norm in dict_results.keys() for attack in dict_results[norm].keys() for metric in ["top1", "top5"]]
        )

        with open(output_path, mode="w", newline="") as file:
            writer = csv.writer(file)
            writer.writerow(header)
            writer.writerow(data_row)

        print(f"Results saved to {output_path}")
    print(time.strftime("%Y%m%d_%H%M%S",time.localtime()))
