import argparse
import os
import shutil
import time
import warnings
from pathlib import Path
from typing import Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from einops import rearrange
from scipy.io import loadmat,savemat

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
warnings.simplefilter("ignore")


DEFAULT_DATASETS = [
    #"MUUFL",
    # "abu-urban-1",
     "abu-urban-4",
    # "abu-airport-2",
    # "abu-airport-3",
]

# (s, lambda_2, lambda_1): SSTV, PAG, and Lipschitz-forcing weights.
DEFAULT_LOSS_WEIGHTS = {
    #"MUUFL": (1e-3, 0.2, 1e-1),
    # "abu-urban-1": (1e-3, 0.1, 2.5e-1),
     "abu-urban-4": (8e-4, 0.5, 1e-1),
    # "abu-airport-2": (8e-4, 0.5, 2.5e-1),
    # "abu-airport-3": (1e-3, 0.5, 1e-1),
    }
   
FALLBACK_LOSS_WEIGHTS = (1e-3, 0.3, 2.5e-1)


def set_seed(seed: int) -> None:
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)


def get_rgb(
    hsi: np.ndarray,
    bands: Tuple[int, int, int] = (30, 20, 10),
    channel_last: Optional[bool] = None,
) -> np.ndarray:
    hsi = np.asarray(hsi)
    if hsi.ndim != 3:
        raise ValueError(f"HSI must be 3D, got shape={hsi.shape}.")
    if len(bands) != 3:
        raise ValueError("bands must contain three indices for (R, G, B).")

    if channel_last is None:
        channel_last = hsi.shape[-1] >= 3 and hsi.shape[0] != 3

    band_dim = hsi.shape[-1] if channel_last else hsi.shape[0]
    for band in bands:
        if not 0 <= band < band_dim:
            raise IndexError(f"Band index {band} out of range for B={band_dim}.")

    if channel_last:
        rgb = np.stack([hsi[..., band] for band in bands], axis=-1)
    else:
        rgb = np.stack([hsi[band] for band in bands], axis=-1)

    return np.clip(rgb.astype(np.float32), 0, 1)


def plot_two_hsi_as_rgb(
    hsi_1: np.ndarray,
    hsi_2: np.ndarray,
    bands: Tuple[int, int, int],
    out_path: Path,
    titles: Tuple[str, str] = ("HSI-1", "HSI-2"),
    layout: str = "horizontal",
    dpi: int = 200,
) -> None:
    rgb_1 = get_rgb(hsi_1, bands=bands)
    rgb_2 = get_rgb(hsi_2, bands=bands)

    if layout == "horizontal":
        fig, axes = plt.subplots(1, 2, figsize=(10, 5))
    else:
        fig, axes = plt.subplots(2, 1, figsize=(6, 10))

    axes = np.asarray(axes).reshape(-1)
    axes[0].imshow(rgb_1)
    axes[0].set_title(titles[0])
    axes[0].axis("off")
    axes[1].imshow(rgb_2)
    axes[1].set_title(titles[1])
    axes[1].axis("off")

    fig.suptitle(f"RGB bands = {bands} (R, G, B)", y=0.98)
    fig.tight_layout()
    fig.savefig(out_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out_path}")


class Engine(nn.Module):
    def __init__(
        self,
        s: float,
        lambda_2: float,
        lambda_1: float,
        iteration: int,
        lr: float,
        shift_pixel: int,
        N: int,
        gaussian_sigma: float,
        weight_scale: float,
        scheduler_t_max: int,
        scheduler_eta_min: float,
        data_dir: Path,
        output_dir: Path,
        rgb_bands: Tuple[int, int, int],
        plot_layout: str,
        plot_dpi: int,
    ) -> None:
        super().__init__()
        self.train_dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.Sg = nn.Sigmoid()
        self.lr = lr
        self.N = N
        self.s = s
        self.lambda_2 = lambda_2
        self.lambda_1 = lambda_1
        self.iteration = iteration
        self.shift_pixel = shift_pixel
        self.weight_scale = weight_scale
        self.scheduler_t_max = scheduler_t_max
        self.scheduler_eta_min = scheduler_eta_min
        self.data_dir = data_dir
        self.output_dir = output_dir
        self.rgb_bands = rgb_bands
        self.plot_layout = plot_layout
        self.plot_dpi = plot_dpi
        self.distribution = self.define_gaussian_kernel(sigma=gaussian_sigma)

    @staticmethod
    def normalize(x):
        if isinstance(x, np.ndarray):
            return (x - np.min(x)) / (np.max(x) - np.min(x))
        if isinstance(x, torch.Tensor):
            return (x - torch.min(x)) / (torch.max(x) - torch.min(x))
        raise TypeError(f"Unsupported input type: {type(x)}")

    @staticmethod
    def spe_spa_tv(x: torch.Tensor) -> torch.Tensor:
        # x: (B, C, H, W)
        dc = torch.abs(x[:, 1:, :, :] - x[:, :-1, :, :]).sum(dim=(-3, -2, -1))
        dh = torch.abs(x[:, :, 1:, :] - x[:, :, :-1, :]).sum(dim=(-3, -2, -1))
        dw = torch.abs(x[:, :, :, 1:] - x[:, :, :, :-1]).sum(dim=(-3, -2, -1))
        return 2 * dc + 2 * dh + dw

    def define_weight(self, Y_H: torch.Tensor) -> torch.Tensor:
        # Y_H: (B, C, H, W)
        _, _, h, w = Y_H.shape
        Y_H = rearrange(Y_H, "b c h w -> b c (h w)")
        Y_H = Y_H - Y_H.mean(dim=-1, keepdim=True)
        U_all, _, _ = torch.linalg.svd(Y_H, full_matrices=False)
        U = U_all[:, :, : self.N]
        U_T_Y_H = U.permute(0, 2, 1) @ Y_H
        U_T_Y_H = rearrange(U_T_Y_H, "b c (h w) -> b c h w", h=h, w=w)
        weight = torch.abs(U_T_Y_H - U_T_Y_H.mean(dim=(-2, -1), keepdim=True))
        return self.weight_scale * (weight / weight.max()) ** 2

    def objective_YA(self, Y_H: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        # Y_H: (B, C, H, W); A: (C, H, W)
        _, _, h, w = Y_H.shape
        Y_A = self.Sg(self.A + Y_H)
        Y_A = rearrange(Y_A, "b c h w -> b c (h w)")
        Y_A = Y_A - Y_A.mean(dim=-1, keepdim=True)
        U_all, _, _ = torch.linalg.svd(Y_A, full_matrices=False)
        U = U_all[:, :, : self.N]
        U_T_Y_A = U.permute(0, 2, 1) @ Y_A
        U_T_Y_A_t = rearrange(U_T_Y_A, "b c (h w) -> b c h w", h=h, w=w)
        U_T_Y_A_w = self.D2CM * U_T_Y_A_t

        dh = U_T_Y_A_w[:, :, 1:, :] - U_T_Y_A_w[:, :, :-1, :]
        dw = U_T_Y_A_w[:, :, :, 1:] - U_T_Y_A_w[:, :, :, :-1]

        PAG_Loss = -torch.norm(Y_A - U @ U_T_Y_A, p="fro", dim=(-2, -1))
        Lipschitz_Forcing_Loss = (
            dh.pow(2).sum(dim=(-2, -1)).sqrt()
            + dw.pow(2).sum(dim=(-2, -1)).sqrt()
        ).sum(dim=1)
        return PAG_Loss, Lipschitz_Forcing_Loss

    def objective_P(self, Y_H: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        # Y_H: (B, C, H, W); A: (C, H, W)
        Y_A = self.Sg(self.A + Y_H)
        P = Y_A - Y_H
        SSTV = self.spe_spa_tv(P)
        Energy = P.pow(2).sum(dim=(-3, -2, -1)).sqrt()
        return Energy, SSTV

    def define_gaussian_kernel(self, sigma: float) -> torch.Tensor:
        size = self.shift_pixel * 2 + 1
        ax = torch.arange(-(size // 2), size // 2 + 1, dtype=torch.float32)
        xx, yy = torch.meshgrid(ax, ax, indexing="ij")
        kernel = torch.exp(-(xx**2 + yy**2) / (2 * sigma**2))
        return kernel / kernel.sum()

    def crop(self, data: torch.Tensor, dx: int, dy: int) -> torch.Tensor:
        # data: (C, H, W)
        _, h, w = data.shape
        return data[
            :,
            self.shift_pixel + dx : h - self.shift_pixel + dx,
            self.shift_pixel + dy : w - self.shift_pixel + dy,
        ]

    def loss(self, Y_H: torch.Tensor) -> torch.Tensor:
        Energy, SSTV = self.objective_P(Y_H)
        PAG_Loss, Lipschitz_Forcing_Loss = self.objective_YA(Y_H)
        return (
            Energy
            + self.s * SSTV
            + self.lambda_2 * PAG_Loss
            + self.lambda_1 * Lipschitz_Forcing_Loss
        )

    def init_optimizer(self, all_parameters):
        optimizer = torch.optim.Adam(all_parameters, lr=self.lr)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=self.scheduler_t_max,
            eta_min=self.scheduler_eta_min,
            last_epoch=-1,
        )
        return optimizer, scheduler

    def initial(self, name: str):
        data = loadmat(self.data_dir / f"{name}.mat")
        GT, HSI = data["map"], data["data_cube"]
        HSI = torch.from_numpy(HSI).permute(2, 0, 1).float().to(self.train_dev)
        if HSI.max() > 1 or HSI.min() < 0:
            HSI = self.normalize(HSI)
        GT = torch.from_numpy(GT).float()

        self.A = nn.Parameter(torch.zeros_like(self.crop(HSI, 0, 0))).to(self.train_dev)
        self.optG, self.sch = self.init_optimizer([self.A])

        offsets = torch.arange(-self.shift_pixel, self.shift_pixel + 1)
        all_shifting_HSI = []
        flatten_kernel = []
        for dx in offsets:
            for dy in offsets:
                all_shifting_HSI.append(self.crop(HSI, dx, dy))
                flatten_kernel.append(self.distribution[self.shift_pixel + dx, self.shift_pixel + dy])

        all_shifting_HSI = torch.stack(all_shifting_HSI, dim=0).to(self.train_dev)
        flatten_kernel = torch.stack(flatten_kernel).to(self.train_dev)
        self.D2CM = self.define_weight(all_shifting_HSI)
        return GT, HSI, all_shifting_HSI, flatten_kernel

    def update(self, all_shifting_HSI: torch.Tensor, epoch: int, flatten_kernel: torch.Tensor) -> None:
        self.optG.zero_grad()
        loss = (self.loss(all_shifting_HSI) * flatten_kernel).sum()
        if epoch % 20 == 0:
            progress = int((epoch + 1) / self.iteration * 20)
            bar = "*" * progress + "-" * (20 - progress)
            print(f"\rTraining |{bar}| {epoch + 1}/{self.iteration} | Loss: {loss:.4f}")
        loss.backward()
        self.optG.step()
        self.sch.step()

    def optimization(self, name: str):
        GT, HSI, all_shifting_HSI, flatten_kernel = self.initial(name)
        print("--------------Perturbation Estimation Begin----------------------------")
        start = time.time()
        for epoch in range(self.iteration):
            self.update(all_shifting_HSI, epoch, flatten_kernel)
        elapsed_time = time.time() - start

        with torch.no_grad():
            Y_H = self.crop(HSI, 0, 0)
            Y_A = self.Sg(all_shifting_HSI + self.A)
            P = ((Y_A - all_shifting_HSI) * flatten_kernel.reshape(-1, 1, 1, 1)).sum(dim=0)
        return P, elapsed_time, GT, HSI, Y_H

    def inference(self, name: str) -> None:
        self.output_dir.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(Path(__file__), self.output_dir / Path(__file__).name)
        P, elapsed_time, _, _, Y_H = self.optimization(name)

        Y_H = Y_H.permute(1, 2, 0).detach().cpu().numpy()
        P = P.permute(1, 2, 0).detach().cpu().numpy()
        plot_two_hsi_as_rgb(
            Y_H,
            Y_H + P,
            bands=self.rgb_bands,
            out_path=self.output_dir / f"{name}.svg",
            layout=self.plot_layout,
            dpi=self.plot_dpi,
        )

        mat_path = self.output_dir / f"{name}.mat"
        savemat(
            mat_path,
            {
                "Perturbation": P,   # (H, W, C)               
                "time_generate_attack": elapsed_time,
                "Shifting_pixel_num": self.shift_pixel,
            },
        )
        print(f"Saved: {mat_path}")


def get_loss_weights(dataset: str, args: argparse.Namespace) -> Tuple[float, float, float]:
    s, lambda_2, lambda_1 = DEFAULT_LOSS_WEIGHTS.get(dataset, FALLBACK_LOSS_WEIGHTS)
    if args.s is not None:
        s = args.s
    if args.lambda_2 is not None:
        lambda_2 = args.lambda_2
    if args.lambda_1 is not None:
        lambda_1 = args.lambda_1
    return s, lambda_2, lambda_1


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Robust AHAD perturbation estimation.")
    parser.add_argument("--datasets", nargs="+", default=DEFAULT_DATASETS)
    parser.add_argument("--data-dir", type=Path, default=Path("./datasets"))
    parser.add_argument("--output-dir", type=Path, default=Path("./AHAD_Data"))
    parser.add_argument("--cuda-visible-devices", type=str, default="0")
    parser.add_argument("--seed", type=int, default=0)

    parser.add_argument("--iteration", type=int, default=500)
    parser.add_argument("--lr", type=float, default=0.05)
    parser.add_argument("--shift-pixel", type=int, default=1)
    parser.add_argument("--N", type=int, default=10, help="Number of ASF components.")
    parser.add_argument("--gaussian-sigma", type=float, default=1.0)
    parser.add_argument("--weight-scale", type=float, default=8e2)
    parser.add_argument("--scheduler-t-max", type=int, default=60)
    parser.add_argument("--scheduler-eta-min", type=float, default=8e-4)

    parser.add_argument("--s", type=float, default=None, help="Scale factor for SSTV regularizer.")
    parser.add_argument("--lambda-2", dest="lambda_2", type=float, default=None, help="Trade-off parameter for PAG loss weight.")
    parser.add_argument("--lambda-1", dest="lambda_1", type=float, default=None, help="Trade-off parameter for Lipschitz-forcing loss weight.")

    parser.add_argument("--rgb-bands", nargs=3, type=int, default=(36, 17, 7), metavar=("R", "G", "B"))
    parser.add_argument("--plot-layout", choices=("horizontal", "vertical"), default="horizontal")
    parser.add_argument("--plot-dpi", type=int, default=200)
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = args.cuda_visible_devices
    set_seed(args.seed)

    for dataset in args.datasets:
        print("-" * 20)
        print(dataset, "\n")
        s, lambda_2, lambda_1 = get_loss_weights(dataset, args)
        model = Engine(
            s=s,
            lambda_2=lambda_2,
            lambda_1=lambda_1,
            iteration=args.iteration,
            lr=args.lr,
            shift_pixel=args.shift_pixel,
            N=args.N,
            gaussian_sigma=args.gaussian_sigma,
            weight_scale=args.weight_scale,
            scheduler_t_max=args.scheduler_t_max,
            scheduler_eta_min=args.scheduler_eta_min,
            data_dir=args.data_dir,
            output_dir=args.output_dir,
            rgb_bands=tuple(args.rgb_bands),
            plot_layout=args.plot_layout,
            plot_dpi=args.plot_dpi,
        )
        model.inference(dataset)


if __name__ == "__main__":
    main()
