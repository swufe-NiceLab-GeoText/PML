import os
import random
import time
import torch
import torch.nn.functional as F
import numpy as np
import cv2
from PIL import Image
import PIL.ImageOps
from model.UNO_road_map_v5_bj import UNO
from utils_pack.metrics import get_MSE, get_MAE, get_MAPE
from utils_pack.utils import get_dataloader, print_model_parm_nums
from utils_pack.args import get_args
from tqdm import tqdm
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
plt.switch_backend('Agg')
from einops import rearrange
from torch.utils.data import ConcatDataset

args = get_args()


os.environ["CUDA_VISIBLE_DEVICES"] = "0"
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
save_path = f'../pycharm_project_916/experiments/joint-{args.model}-{args.dataset}-{args.n_channels}'
os.makedirs(save_path, exist_ok=True)
def calc_recon_loss(pred, target, mask, patch_size=8):
    B, C, H_pred, W_pred = pred.shape
    num_patches = mask.size(1)

    h_patches_low = int(num_patches ** 0.5)
    assert h_patches_low ** 2 == num_patches, "num_patches must be a square number"
    w_patches_low = h_patches_low



    patch_mask = mask.view(B, h_patches_low, w_patches_low)

    patch_mask = F.interpolate(
        patch_mask.unsqueeze(1).float(),
        size=(H_pred, W_pred),
        mode='nearest'
    ).squeeze(1)
    patch_mask = patch_mask.to(mask.dtype)



    assert pred.shape == target.shape, "pred and target must be the same size"

    pixel_loss = F.mse_loss(pred, target, reduction='none')
    return (pixel_loss * patch_mask.unsqueeze(1)).mean()

def choose_model():
    if args.model == 'PML_bj':
        return UNO(
            height=args.height, width=args.width,
            use_exf=args.use_exf, scale_factor=args.scale_factor,
            channels=args.n_channels, sub_region=args.sub_region,
            scaler_X=args.scaler_X, scaler_Y=args.scaler_Y, args=args
        )


class JointTrainer:
    def __init__(self, args):
        self.args = args
        self.best_metrics = {'val_mse': np.inf, 'epoch': 0}
        self.train_sequence = ["P1", "P2", "P3", "P4"]
        self.road_map = self._load_roadmap()
        self.task_records = {
            task: {"test_results": {"MSE": 0, "MAE": 0, "MAPE": 0}}
            for task in self.train_sequence
        }

    def _load_roadmap(self):
        roadmap_path = f"../pycharm_project_916/datasets/road_map/{args.dataset}.png"
        roadmap = Image.open(roadmap_path).convert('L')
        roadmap = PIL.ImageOps.invert(roadmap)
        roadmap = np.expand_dims(cv2.resize(np.array(roadmap), (128, 128)), 0)
        return torch.FloatTensor(roadmap).to(device)

    def _get_joint_dataloaders(self):

        def load_all(mode):
            datasets = []
            for task_id in range(len(self.train_sequence)):
                loader = get_dataloader(
                    args,
                    datapath='../pycharm_project_916/datasets',
                    dataset=args.dataset,
                    batch_size=args.batch_size if mode == 'train' else 32,
                    mode=mode,
                    task_id=task_id + 1
                )
                datasets.append(loader.dataset)
            return DataLoader(ConcatDataset(datasets),
                              batch_size=args.batch_size,
                              shuffle=(mode == 'train'))

        return {
            'pretrain': load_all('train'),
            'train': load_all('train'),
            'val': load_all('valid'),
            'test': [get_dataloader(args, datapath='../pycharm_project_916/datasets',mode='test', task_id=i + 1) for i in range(4)]
        }

    def _get_optimizer(self, model):
        base_params = [p for n, p in model.named_parameters() if 'decoder' not in n]
        decoder_params = [p for n, p in model.named_parameters() if 'decoder' in n]
        return torch.optim.Adam([
            {'params': base_params, 'lr': args.lr},
            {'params': decoder_params, 'lr': args.lr * 0.1}
        ], betas=(args.b1, args.b1))

    def _pretrain_phase(self, model, optimizer, loader):
        model.train()
        for epoch in range(args.pretrain_epochs):
            total_loss = 0
            for c_map, f_map, exf in tqdm(loader, desc=f"Pretrain Epoch {epoch + 1}"):
                c_map, f_map = c_map.to(device), f_map.to(device)
                exf = exf.to(device)

                optimizer.zero_grad()
                pred, recon, mask = model(c_map, exf, self.road_map, is_pretrain=True)

                task_loss = F.l1_loss(pred, f_map * args.scaler_Y)
                recon_loss = calc_recon_loss(recon, f_map, mask)
                loss = recon_loss + args.lambda_recon * task_loss

                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()

                total_loss += loss.item() * len(c_map)

            print(f"Pretrain Loss: {total_loss / len(loader.dataset):.4f}")

    def _joint_train_phase(self, model, optimizer, train_loader, val_loader):
        scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
            optimizer, T_0=5, T_mult=2, eta_min=args.lr * 0.001)

        for epoch in range(args.joint_epochs):
            # Training
            model.train()
            total_loss = 0
            for c_map, f_map, exf in tqdm(train_loader, desc=f"Train Epoch {epoch + 1}"):
                c_map, f_map = c_map.to(device), f_map.to(device)
                exf = exf.to(device)

                optimizer.zero_grad()
                pred = model(c_map, exf, self.road_map) * args.scaler_Y
                loss = F.smooth_l1_loss(pred, f_map * args.scaler_Y)

                loss.backward()
                optimizer.step()
                total_loss += loss.item() * len(c_map)

            # Validation
            val_mse = self._evaluate(model, val_loader)
            scheduler.step()

            # Save best model
            if val_mse < self.best_metrics['val_mse']:
                self.best_metrics.update({
                    'val_mse': val_mse,
                    'epoch': epoch,
                    'state_dict': model.state_dict()
                })
                torch.save(model.state_dict(), f"{save_path}/best_joint.pth")

            print(f"Epoch {epoch + 1}/{args.joint_epochs} | "
                  f"Train Loss: {total_loss / len(train_loader.dataset):.4f} | "
                  f"Val MSE: {val_mse:.4f} | "
                  f"LR: {scheduler.get_last_lr()[0]:.2e}")

    def _evaluate(self, model, loader):
        model.eval()
        total_mse = 0
        with torch.no_grad():
            for c_map, f_map, exf in loader:
                c_map, f_map = c_map.to(device), f_map.to(device)
                exf = exf.to(device)

                pred = model(c_map, exf, self.road_map)
                total_mse += F.mse_loss(pred, f_map * args.scaler_Y).item() * len(c_map)
        return total_mse / len(loader.dataset)

    def test_all_tasks(self, model):
        model.load_state_dict(torch.load(f"{save_path}/best_joint.pth"))
        model.eval()

        for task_id in range(4):
            task_name = self.train_sequence[task_id]
            loader = get_dataloader(args, datapath='../pycharm_project_916/datasets',mode='test', task_id=task_id + 1)

            metrics = {'MSE': 0, 'MAE': 0, 'MAPE': 0}
            for c_map, f_map, exf in tqdm(loader, desc=f"Testing {task_name}"):
                pred = model(c_map.to(device), exf.to(device), self.road_map).cpu()

                pred = pred.cpu().detach().numpy() * args.scaler_Y
                real = f_map.cpu().detach().numpy() * args.scaler_Y
                metrics['MSE'] += get_MSE(pred, real) * len(c_map)
                metrics['MAE'] += get_MAE(pred, real) * len(c_map)
                metrics['MAPE'] += get_MAPE(pred, real) * len(c_map)

            for k in metrics:
                self.task_records[task_name]['test_results'][k] = metrics[k] / len(loader.dataset)

    def run(self):
        start_time = time.time()
        model = choose_model().to(device)
        optimizer = self._get_optimizer(model)
        print_model_parm_nums(model, args.model)

        dataloaders = self._get_joint_dataloaders()


        self._pretrain_phase(model, optimizer, dataloaders['pretrain'])
        self._joint_train_phase(model, optimizer,
                                dataloaders['train'], dataloaders['val'])

        self.test_all_tasks(model)

        print("\n=== Final Results ===")
        for task in self.train_sequence:
            start_time1 = time.time()
            res = self.task_records[task]['test_results']
            print(f"{task} - MSE: {res['MSE']:.4f} | "
                  f"MAE: {res['MAE']:.4f} | MAPE: {res['MAPE']:.2%}")
            print(f"\nTotal Time: {(time.time() - start_time1) / 3600:.2f} hours")
        print(f"\nTotal Time: {(time.time() - start_time) / 3600:.2f} hours")


if __name__ == "__main__":
    trainer = JointTrainer(args)
    trainer.run()