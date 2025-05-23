import os
import random
import time
import torch
import torch.nn.functional as F
import numpy as np
import cv2
from PIL import Image
import PIL.ImageOps
from model.PML_cd import PML
from utils_pack.metrics import  get_MAE, get_MAPE, get_RMSE
from utils_pack.utils import get_dataloader, print_model_parm_nums
from utils_pack.args_cd import get_args
from tqdm import tqdm
from einops import rearrange
import math

args = get_args()

# ===========================================================

os.environ["CUDA_VISIBLE_DEVICES"] = "0"
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


save_path = f'../experiments/masked-{args.model}-{args.dataset}-{args.n_channels}'
os.makedirs(save_path, exist_ok=True)



def calc_recon_loss(pred, target, mask):

    B, C, H_pred, W_pred = pred.shape
    num_patches = mask.size(1)

    h_patches_low = int(math.sqrt(num_patches))
    assert h_patches_low ** 2 == num_patches, "num_patches should be squared"
    w_patches_low = h_patches_low
    try:
        patch_mask = mask.view(B, h_patches_low, w_patches_low)


        upsampled_size = (H_pred, W_pred)
        patch_mask = F.interpolate(
            patch_mask.unsqueeze(1).float(),
            size=upsampled_size,
            mode='nearest'
        ).expand(-1, C, -1, -1)
    except Exception as e:
        print(f"Error Details:\n"
              f"- Input mask shape: {mask.shape}\n"
              f"- Calculated grid: {h_patches_low}x{w_patches_low}\n"
              f"- Target size: {H_pred}x{W_pred}")
        raise


    assert pred.shape == target.shape, f"Shape mismatch: pred {pred.shape} vs target {target.shape}"
    assert patch_mask.shape == pred.shape, f"Mask shape {patch_mask.shape} should match pred {pred.shape}"


    pixel_loss = F.mse_loss(pred, target, reduction='none')
    return (pixel_loss * patch_mask).mean()

def choose_model():
    if args.model == 'PML_cd':
        model = PML(height=args.height, width=args.width, use_exf=args.use_exf,
                    scale_factor=args.scale_factor, channels=args.n_channels,
                    sub_region=args.sub_region,
                    scaler_X=args.scaler_X, scaler_Y=args.scaler_Y, args=args)
    return model

class MaskedTrainer:
    def __init__(self, args):
        self.args = args
        self.best_metrics = {'train_loss': np.inf, 'val_mse': np.inf}
        self.road_map = self._load_roadmap()
        self.train_sequence = ["P1"]
        self.best_metrics = {task: {"mse": np.inf, "epoch": 0} for task in self.train_sequence}
        self.task_records = {
            task: {
                "train_loss": [],
                "val_mse": [],
                "test_results": {"RMSE": 0, "MAE": 0, "MAPE": 0}
            } for task in self.train_sequence
        }
        model = choose_model().to(device)
        self.optimizer = self._get_optimizer(model)
    def _load_roadmap(self):

        roadmap_path = "../pycharm_project_916/datasets/road_map/{}.png".format(args.dataset)
        roadmap = Image.open(roadmap_path).convert('L')
        roadmap = PIL.ImageOps.invert(roadmap)
        roadmap = np.expand_dims(cv2.resize(np.array(roadmap), (64, 64)), 0)
        return torch.FloatTensor(roadmap).to(device)

    def _get_task_dataloaders(self, task_id):
        total_datapath = '../pycharm_project_916/datasets'
        return {
            'pretrain': get_dataloader(args,
                              datapath=total_datapath, dataset=args.dataset,
                              batch_size=args.batch_size, mode='train', task_id=task_id+1),
            'train': get_dataloader(args,
                              datapath=total_datapath, dataset=args.dataset,
                              batch_size=args.batch_size, mode='train', task_id=task_id+1),
            'val': get_dataloader(args,
                                datapath=total_datapath, dataset=args.dataset,
                                batch_size=32, mode='valid', task_id=task_id+1),
            'test': get_dataloader(args,
                             datapath=total_datapath, dataset=args.dataset,
                             batch_size=32, mode='test', task_id=task_id+1)
        }
    def _get_optimizer(self, model):

        base_params = [p for n, p in model.named_parameters() if 'decoder' not in n]
        decoder_params = [p for n, p in model.named_parameters() if 'decoder' in n]
        return torch.optim.Adam([
            {'params': base_params, 'lr': args.lr},
            {'params': decoder_params, 'lr': args.lr * 0.1}
        ], betas=(args.b1, args.b1))

    def run_phase(self, model, dataloader, phase='train'):

        is_train = phase == 'train'
        model.train() if is_train else model.eval()

        total_loss, total_samples = 0, 0
        with torch.set_grad_enabled(is_train):
            for c_map, f_map, exf in dataloader:
                f_map = f_map.squeeze(1)
                c_map, f_map = c_map.to(device), f_map.to(device)
                f_map = f_map.squeeze(1)
                exf = exf.to(device)


                if phase == 'pretrain':
                    pred, recon, mask = model(c_map, exf, self.road_map, is_pretrain=True)
                    task_loss = F.l1_loss(pred, f_map * args.scaler_Y)
                    recon_loss = calc_recon_loss(recon, f_map, mask)
                    loss = recon_loss + args.lambda_recon * task_loss
                else:
                    pred = model(c_map, exf, self.road_map)
                    loss = F.smooth_l1_loss(pred, f_map * args.scaler_Y)


                if is_train:
                    self.optimizer.zero_grad()
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                    self.optimizer.step()

                total_loss += loss.item() * len(c_map)
                total_samples += len(c_map)

        return total_loss / total_samples

    def train_task(self, task_id):
        task_name = self.train_sequence[task_id]
        print(f"\n=== Training Task {task_name} ===")


        model = choose_model().to(device)
        optimizer = self._get_optimizer(model)

        dataloaders = self._get_task_dataloaders(task_id)


        self._pretrain_phase(model, optimizer, dataloaders['pretrain'], task_name)
        self._joint_train_phase(model, optimizer, dataloaders['train'], dataloaders['val'], task_name)
        self._fine_tune_phase(model, optimizer, dataloaders['train'], task_name)


        self.test_task(model, dataloaders['test'], task_name)

    def _pretrain_phase(self, model, optimizer, dataloader, task_name):
        print(f"\n─── Phase 1: Masked Pretraining ({task_name}) ───")

        for epoch in range(self.args.pretrain_epochs):
            model.train()
            total_loss = 0
            for c_map, f_map, exf in tqdm(dataloader, desc=f"Pretrain Epoch {epoch + 1}"):
                f_map = f_map.squeeze(1)
                c_map, f_map = c_map.to(device), f_map.to(device)
                exf = exf.to(device)

                optimizer.zero_grad()
                pred, recon, mask = model(c_map, exf, self.road_map, is_pretrain=True)

                # 联合损失计算
                task_loss = F.l1_loss(pred, f_map * self.args.scaler_Y)
                recon_loss = calc_recon_loss(recon, f_map, mask)
                loss = recon_loss + self.args.lambda_recon * task_loss

                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()

                total_loss += loss.item() * len(c_map)

            avg_loss = total_loss / len(dataloader.dataset)
            self.task_records[task_name]["train_loss"].append(avg_loss)
            print(f"Pretrain Loss: {avg_loss:.4f}")


    def _joint_train_phase(self, model, optimizer, train_loader, val_loader, task_name):
        print(f"\n─── Phase 2: Joint Training ({task_name}) ───")
        scheduler =torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
            optimizer,
            T_0=5,
            T_mult=3,
            eta_min=args.lr * 0.001
        )

        for epoch in range(self.args.joint_epochs):

            model.train()
            total_loss = 0
            for c_map, f_map, exf in tqdm(train_loader, desc=f"Joint Epoch {epoch + 1}"):
                f_map = f_map.squeeze(1)
                c_map, f_map = c_map.to(device), f_map.to(device)
                exf = exf.to(device)

                optimizer.zero_grad()
                pred = model(c_map, exf, self.road_map)* self.args.scaler_Y

                loss = F.smooth_l1_loss(pred, f_map * self.args.scaler_Y)

                loss.backward()
                optimizer.step()
                total_loss += loss.item() * len(c_map)


            val_mse = self._evaluate(model, val_loader, task_name)
            scheduler.step()

            if val_mse < self.best_metrics[task_name]["mse"]:
                self.best_metrics[task_name].update({
                    "mse": val_mse,
                    "epoch": epoch,
                    "state_dict": model.state_dict()
                })
                torch.save(model.state_dict(),
                           f"{save_path}/best_{task_name}.pth")

            avg_loss = total_loss / len(train_loader.dataset)
            self.task_records[task_name]["train_loss"].append(avg_loss)
            self.task_records[task_name]["val_mse"].append(val_mse)

            print(
                f"Epoch {epoch + 1}: Train Loss={avg_loss:.4f}, Val MSE={val_mse:.4f}, LR={scheduler.get_last_lr()[0]:.1e}")

    def _fine_tune_phase(self, model, optimizer, train_loader, task_name):
        print(f"\n─── Phase 3: Fine-tuning ({task_name}) ───")

        for param in model.decoder.parameters():
            param.requires_grad = False


        ft_optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()),
                                        lr=self.args.lr * 0.1)

        for epoch in range(self.args.finetune_epochs):
            model.train()
            total_loss = 0
            for c_map, f_map, exf in tqdm(train_loader, desc=f"Finetune Epoch {epoch + 1}"):
                f_map = f_map.squeeze(1)
                c_map, f_map = c_map.to(device), f_map.to(device)
                exf = exf.to(device)

                ft_optimizer.zero_grad()
                pred = model(c_map, exf, self.road_map)*self.args.scaler_Y
                loss = F.smooth_l1_loss(pred, f_map * self.args.scaler_Y)

                loss.backward()
                ft_optimizer.step()
                total_loss += loss.item() * len(c_map)

            avg_loss = total_loss / len(train_loader.dataset)
            print(f"Finetune Loss: {avg_loss:.4f}")
    def _save_checkpoint(self, model, epoch, val_loss, phase):

        if val_loss < self.best_metrics['val_mse']:
            self.best_metrics.update({
                'epoch': epoch,
                'val_mse': val_loss,
                'phase': phase
            })
            torch.save(model.state_dict(), f"{save_path}/best_model.pth")

    def _evaluate(self, model, dataloader):
        model.eval()
        total_mse = 0
        with torch.no_grad():
            for c_map, f_map, exf in dataloader:
                f_map = f_map.squeeze(1)
                c_map, f_map = c_map.to(device), f_map.to(device)
                exf = exf.to(device)

                pred = model(c_map, exf, self.road_map)
                pred = pred.cpu().detach().numpy() * args.scaler_Y
                real = f_map.cpu().detach().numpy() * args.scaler_Y

                total_mse += get_RMSE(pred=pred, real=real) * len(c_map)
        return total_mse / len(dataloader.dataset)

    def test_task(self, model, test_loader, task_name):
        print(f"\n=== Testing Task {task_name} ===")
        model.load_state_dict(self.best_metrics[task_name]["state_dict"])
        model.eval()

        metrics = {"RMSE": 0, "MAE": 0, "MAPE": 0}
        with torch.no_grad():
            for c_map, f_map, exf in tqdm(test_loader, desc="Testing"):
                f_map = f_map.squeeze(1)
                pred = model(c_map.to(device), exf.to(device), self.road_map)

                pred = pred.cpu().detach().numpy() * args.scaler_Y
                real = f_map.cpu().detach().numpy() * args.scaler_Y
                metrics["RMSE"] += get_RMSE(pred=pred, real=real) * len(c_map)
                metrics["MAE"] += get_MAE(pred=pred, real=real) * len(c_map)
                metrics["MAPE"] += get_MAPE(pred=pred, real=real) * len(c_map)

        for k in metrics:
            metrics[k] /= len(test_loader.dataset)
            self.task_records[task_name]["test_results"][k] = metrics[k]

        print(f"Test Results - RMSE: {metrics['RMSE']:.4f}, MAE: {metrics['MAE']:.4f}, MAPE: {metrics['MAPE']:.2%}")

    def run(self):
        start_time = time.time()
        for task_id in range(len(self.train_sequence)):
            self.train_task(task_id)


        print("\n=== Final Report ===")
        for task in self.train_sequence:
            print(f"\nTask {task}:")
            print(f"Best Epoch: {self.best_metrics[task]['epoch']}")
            print(f"Best Val MSE: {self.best_metrics[task]['mse']:.4f}")
            print("Test Metrics:")
            for metric, value in self.task_records[task]["test_results"].items():
                print(f"{metric}: {value:.4f}" if metric != "MAPE" else f"{metric}: {value:.2%}")

        print(f"\nTotal Time: {(time.time() - start_time) / 3600:.2f} hours")


if __name__ == "__main__":
    trainer = MaskedTrainer(args)
    start_time = time.time()
    trainer.run()
    print(f"Total time: {(time.time() - start_time) / 3600:.2f} hours")