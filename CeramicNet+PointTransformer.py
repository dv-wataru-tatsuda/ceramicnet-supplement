#!/usr/bin/env python
# SPDX-License-Identifier: MIT

import os

# Get MODE from environment variable, default to CPU if not set
MODE = os.getenv('MODE', 'CPU')  # "GPU" or "CPU"

import concurrent.futures
import os
import random
import pickle
import math
import threading
from datetime import datetime
from io import BytesIO

if MODE == "GPU":
    import cupy as cp

import madgrad
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap, to_rgb
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F

from scipy.cluster.hierarchy import dendrogram, linkage
from scipy.cluster.hierarchy import set_link_color_palette as set_color
from sklearn.decomposition import PCA
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import KFold
from torch.utils.data import Dataset
from torchvision import transforms
from tqdm import tqdm

local_base_dir = "ceramicnet_data"
train_key = 'ceramicnet_train'
test_key = 'ceramicnet_test'
shape_names_file = os.path.join(local_base_dir, "ceramicnet_shape_names.txt")
fold_num = 5

def kfold_sample():
    all_files = []

    for root, subdirs, files in os.walk(local_base_dir):
        relative_path = os.path.relpath(root, local_base_dir)
        if relative_path != ".":
            for file in files:
                if file.endswith('.txt'):
                    all_files.append(os.path.join(root, file))

    print(f"ALL_FILES:{len(all_files)}")
    random.shuffle(all_files)

    train_splits = [""] * fold_num
    test_splits = [""] * fold_num

    kf = KFold(n_splits=fold_num, shuffle=True, random_state=42)
    for i, (train_index, test_index) in enumerate(kf.split(all_files)):
        train_files = [all_files[idx] for idx in train_index]
        test_files = [all_files[idx] for idx in test_index]

        print(f"KFOLD: \n TRAIN: {len(train_files)} TEST: {len(test_files)} INDEX: {i+1}")

        train_splits[i] += "\n".join([os.path.basename(file) for file in train_files]) + "\n"
        test_splits[i] += "\n".join([os.path.basename(file) for file in test_files]) + "\n"

    # Create output directory
    os.makedirs("output", exist_ok=True)

    for i, data in enumerate(train_splits):
        key = f"{train_key}_fold_{i+1}.txt"
        with open(os.path.join(local_base_dir, key), 'w') as f:
            f.write(data)
        # Save a copy to output directory
        with open(os.path.join("output", key), 'w') as f:
            f.write(data)
        print(f"DATA_SPLIT: \n Index: {i+1} \n Key: {key}")

    for i, data in enumerate(test_splits):
        key = f"{test_key}_fold_{i+1}.txt"
        with open(os.path.join(local_base_dir, key), 'w') as f:
            f.write(data)
        # Save a copy to output directory
        with open(os.path.join("output", key), 'w') as f:
            f.write(data)
        print(f"DATA_SPLIT: \n Index: {i+1} \n Key: {key}")

def pc_normalize(pc):
    if MODE == "GPU":
        pc = cp.asarray(pc)
        centroid = cp.mean(pc, axis=0)
        pc = pc - centroid
        m = cp.max(cp.sqrt(cp.sum(pc**2, axis=1)))
        pc = pc / m
        return cp.asnumpy(pc)
    else:
        pc = torch.tensor(pc)
        centroid = pc.mean(dim=0)
        pc = pc - centroid
        m = pc.norm(dim=1).max()
        pc = pc / m
        return pc.numpy()


def farthest_point_sample(point, npoint):
    """
    Input:
        xyz: pointcloud data, [N, D]
        npoint: number of samples
    Return:
        centroids: sampled pointcloud index, [npoint, D]
    """
    if MODE == "GPU":
        point = cp.asarray(point)
        N, D = point.shape
        xyz = point[:, :3]
        centroids = cp.zeros((npoint,))
        distance = cp.ones((N,)) * 1e10
        farthest = cp.random.randint(0, N)
        for i in range(npoint):
            centroids[i] = farthest
            centroid = xyz[farthest, :]
            dist = cp.sum((xyz - centroid) ** 2, -1)
            mask = dist < distance
            distance[mask] = dist[mask]
            farthest = cp.argmax(distance, -1)
        point = point[centroids.astype(cp.int32)]
        return cp.asnumpy(point)
    else:
        N, D = point.shape
        xyz = point[:, :3]
        centroids = np.zeros((npoint,))
        distance = np.ones((N,)) * 1e10
        farthest = np.random.randint(0, N)
        for i in range(npoint):
            centroids[i] = farthest
            centroid = xyz[farthest, :]
            dist = np.sum((xyz - centroid) ** 2, -1)
            mask = dist < distance
            distance[mask] = dist[mask]
            farthest = np.argmax(distance, -1)
        point = point[centroids.astype(np.int32)]
        return point


class RandomRotation_z(object):
    def __call__(self, pointcloud):
        if MODE == "GPU":
            pointcloud = cp.asarray(pointcloud)
            theta = cp.random.rand() * 2.0 * cp.pi
            rot_matrix = cp.array(
                [
                    [cp.cos(theta), -cp.sin(theta), 0],
                    [cp.sin(theta), cp.cos(theta), 0],
                    [0, 0, 1],
                ]
            )
            rot_pointcloud = cp.dot(pointcloud, rot_matrix)
            return cp.asnumpy(rot_pointcloud)
        else:
            pointcloud = torch.tensor(pointcloud)
            theta = torch.rand(1) * 2.0 * np.pi
            rot_matrix = torch.tensor(
                [
                    [torch.cos(theta), -torch.sin(theta), 0],
                    [torch.sin(theta), torch.cos(theta), 0],
                    [0, 0, 1],
                ]
            )
            rot_pointcloud = torch.mm(pointcloud, rot_matrix)
            return rot_pointcloud.numpy()


class RandomNoise(object):
    def __call__(self, pointcloud):
        if MODE == "GPU":
            pointcloud = cp.asarray(pointcloud)
            noise = cp.random.normal(0, 0.02, (pointcloud.shape))
            noisy_pointcloud = pointcloud + noise
            return cp.asnumpy(noisy_pointcloud)
        else:
            pointcloud = torch.tensor(pointcloud)
            noise = torch.normal(0, 0.02, (pointcloud.shape))
            noisy_pointcloud = pointcloud + noise
            return noisy_pointcloud.numpy()


class ShufflePoints(object):
    def __call__(self, pointcloud):
        if MODE == "GPU":
            pointcloud = cp.asarray(pointcloud)
            cp.random.shuffle(pointcloud)
            return cp.asnumpy(pointcloud)
        else:
            pointcloud = torch.tensor(pointcloud)
            torch.randperm(pointcloud)
            return pointcloud.numpy()


def default_transforms():
    return transforms.Compose([RandomRotation_z(), RandomNoise()])


class CeramicNetDataLoader(Dataset):
    def __init__(
        self,
        root,
        num_point=1024,
        transforms=default_transforms(),
        use_uniform_sample=True,
        use_normals=True,
        split="train",
        process_data=False,
        fold=0,
    ):
        self.root = root
        self.npoints = num_point
        self.process_data = process_data
        self.uniform = use_uniform_sample
        self.use_normals = use_normals
        self.transforms = transforms
        self.split = split

        self.catfile = shape_names_file

        # Load class names from local file
        with open(self.catfile, "r") as f:
            self.cat = [line.rstrip() for line in f if line]

        self.classes = dict(zip(self.cat, range(len(self.cat))))

        # Load train/test file lists from local files
        shape_ids = {}
        with open(f"{os.path.join(local_base_dir, train_key)}_fold_{fold}.txt", "r") as f:
            shape_ids["train"] = [line.rstrip() for line in f if line]
        with open(f"{os.path.join(local_base_dir, test_key)}_fold_{fold}.txt", "r") as f:
            shape_ids["test"] = [line.rstrip() for line in f if line]

        assert split == "train" or split == "test"
        shape_names = [
            "_".join(x.split("_")[0:-1]) if "_" in x else x for x in shape_ids[split]
        ]
        self.datapath = [
            (
                shape_names[i],
                self.root + "/" + shape_names[i] + "/" + shape_ids[split][i],
            )
            for i in range(len(shape_ids[split]))
        ]

        print("The size of %s data is %d" % (split, len(self.datapath)))

        if self.uniform:
            self.save_path = self.root + "ceramicnet_%s_%dpts_fps_fold_%d.dat" % (
                split,
                self.npoints,
                fold,
            )
        else:
            self.save_path = self.root + "ceramicnet_%s_%dpts_fold_%d.dat" % (
                split,
                self.npoints,
                fold,
            )

        if self.process_data:
            print(
                "Processing data %s (only running in the first time)..."
                % self.save_path
            )
            # initialize the lists
            self.list_of_points = [None] * len(self.datapath)
            self.list_of_labels = [None] * len(self.datapath)

            with concurrent.futures.ThreadPoolExecutor(max_workers=20) as executor:
                results = list(
                    tqdm(
                        executor.map(
                            self._process_data_point, enumerate(self.datapath)
                        ),
                        total=len(self.datapath),
                        position=0,
                        leave=True,
                    )
                )

            with open(self.save_path, "wb") as f:
                pickle.dump([self.list_of_points, self.list_of_labels], f)
            # with BytesIO() as f:
            #     pickle.dump([self.list_of_points, self.list_of_labels], f)
            #     f.seek(0)
        else:
            if os.path.exists(self.save_path):
                print("Load processed data from %s..." % self.save_path)
                with open(self.save_path, "rb") as f:
                    self.list_of_points, self.list_of_labels = pickle.load(f)
            else:
                print("No data found at %s" % self.save_path)

    def _process_data_point(self, datapath_with_index):
        index, datapath = datapath_with_index
        category_name, file_path = datapath
        category_label = self.classes[category_name]
        category_label_array = np.array([category_label]).astype(np.int32)

        point_set_data = np.loadtxt(file_path, delimiter=" ").astype(np.float32)
        if self.uniform:
            point_set_data = farthest_point_sample(point_set_data, self.npoints)
        else:
            point_set_data = point_set_data[0 : self.npoints, :]

        self.list_of_points[index] = point_set_data
        self.list_of_labels[index] = category_label_array
        return point_set_data, category_label_array

    def __len__(self):
        return len(self.datapath)

    def _get_item(self, index):
        point_set, label = self.list_of_points[index], self.list_of_labels[index]
        point_set[:, 0:3] = pc_normalize(point_set[:, 0:3])
        return point_set, label

    def __getitem__(self, index):
        return self._get_item(index)

    def __reduce__(self):
        return (
            self.__class__,
            (
                self.root,
                self.bucket_name,
                self.npoints,
                self.transforms,
                self.uniform,
                self.use_normals,
                self.split,
                self.process_data,
            ),
        )

def visualize_attention_plain(xyz, attn, batch_idx, counter=0):
    # xyz: b x n x 3
    # attn: b x n x k x d_model

    # Select a random point and its attention weights
    point_idx = np.random.randint(xyz.shape[1])
    point_attn = (
        attn[batch_idx, point_idx, :, :].mean(dim=-1).detach().cpu().numpy()
    )  # k

    # Get the selected point and its k nearest neighbors
    point_xyz = xyz[batch_idx, point_idx, :].detach().cpu().numpy()  # 3
    knn_xyz = (
        xyz[batch_idx, attn[batch_idx, point_idx, :, 0].argsort(descending=True), :]
        .detach()
        .cpu()
        .numpy()
    )  # k x 3

    # Create a 3D figure for attention visualization
    fig = plt.figure()
    ax = fig.add_subplot(111, projection="3d")

    # Plot the original point cloud in blue color
    original_points = xyz[batch_idx].detach().cpu().numpy()
    ax.scatter(
        original_points[:, 0],
        original_points[:, 1],
        original_points[:, 2],
        c="gray",
        s=20,
        alpha=0.5,
    )

    # Plot the selected point in red color
    # ax.scatter(point_xyz[0], point_xyz[1], point_xyz[2], c='red', s=100, marker='o')

    # Plot the k nearest neighbors with attention weights as colors
    # cmap = plt.colormaps.get_cmap('coolwarm')
    # colors = cmap(point_attn / point_attn.max())
    # ax.scatter(knn_xyz[:, 0], knn_xyz[:, 1], knn_xyz[:, 2], c=colors, s=50)

    # Fix the view angle
    ax.view_init(elev=20, azim=30)
    ax.set_xlim([-0.8, 0.8])
    ax.set_ylim([-0.8, 0.8])
    ax.set_zlim([-0.4, 0.4])

    # Remove the grid and axis
    ax.grid(False)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_zticks([])
    ax.set_xticklabels([])
    ax.set_yticklabels([])
    ax.set_zticklabels([])
    ax.set_axis_off()

    ax.grid(False)
    plt.show()
    plt.close()

def visualize_attention(xyz, attn, batch_idx, counter=0):
    # xyz: b x n x 3
    # attn: b x n x k x d_model

    # Select a random point and its attention weights
    point_idx = np.random.randint(xyz.shape[1])
    point_attn = (
        attn[batch_idx, point_idx, :, :].mean(dim=-1).detach().cpu().numpy()
    )  # k

    # Get the selected point and its k nearest neighbors
    point_xyz = xyz[batch_idx, point_idx, :].detach().cpu().numpy()  # 3
    knn_xyz = (
        xyz[batch_idx, attn[batch_idx, point_idx, :, 0].argsort(descending=True), :]
        .detach()
        .cpu()
        .numpy()
    )  # k x 3

    # Create a 3D figure for attention visualization
    fig = plt.figure()
    ax = fig.add_subplot(111, projection="3d")

    # Plot the original point cloud in blue color
    original_points = xyz[batch_idx].detach().cpu().numpy()
    ax.scatter(
        original_points[:, 0],
        original_points[:, 1],
        original_points[:, 2],
        c="gray",
        s=20,
        alpha=0.5,
    )

    # Plot the selected point in red color
    ax.scatter(point_xyz[0], point_xyz[1], point_xyz[2], c="red", s=50, marker="o")

    # Plot the k nearest neighbors with attention weights as colors
    cmap = LinearSegmentedColormap.from_list("gradcam_gray_red", ["gray", "red"])
    colors = cmap(point_attn / point_attn.max())
    ax.scatter(knn_xyz[:, 0], knn_xyz[:, 1], knn_xyz[:, 2], c=colors, s=20)

    # Fix the view angle
    ax.view_init(elev=20, azim=30)
    ax.set_xlim([-0.8, 0.8])
    ax.set_ylim([-0.8, 0.8])
    ax.set_zlim([-0.4, 0.4])

    # Remove the grid and axis
    ax.grid(False)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_zticks([])
    ax.set_xticklabels([])
    ax.set_yticklabels([])
    ax.set_zticklabels([])
    ax.set_axis_off()

    ax.grid(False)

    plt.show()
    plt.close()


# point transformer block
def square_distance(src, dst):
    """
    Input:
        src: source points, [B, N, C]
        dst: target points, [B, M, C]
    Output:
        dist: per-point square distance, [B, N, M]
    """
    return torch.sum((src[:, :, None] - dst[:, None]) ** 2, dim=-1)

def index_points(points, idx):
    """
    Input:
        points: input points data, [B, N, C]
        idx: sample index data, [B, S, K]
    Output:
        new_points:, indexed points data, [B, S, K, C]
    """
    raw_size = idx.size()
    idx = idx.reshape(raw_size[0], -1)
    res = torch.gather(points, 1, idx[..., None].expand(-1, -1, points.size(-1)))
    return res.reshape(*raw_size, -1)

class PointTransformerBlock(nn.Module):
    def __init__(self, d_points, d_model, k) -> None:
        super().__init__()
        self.fc1 = nn.Linear(d_points, d_model)
        self.fc2 = nn.Linear(d_model, d_points)
        self.fc_delta = nn.Sequential(
            nn.Linear(3, d_model), nn.ReLU(), nn.Linear(d_model, d_model)
        )
        self.fc_gamma = nn.Sequential(
            nn.Linear(d_model, d_model), nn.ReLU(), nn.Linear(d_model, d_model)
        )
        self.phi = nn.Linear(d_model, d_model, bias=False)  # queries
        self.psi = nn.Linear(d_model, d_model, bias=False)  # keys
        self.alpha = nn.Linear(d_model, d_model, bias=False)  # values
        self.k = k
        self.lock = threading.Lock()
        self.attn_counter = 0  # Counter for attention visualization
        self.target_batch_index = 0

    # xyz: b x n x 3, features: b x n x f (f=d_points)
    def forward(self, xyz, features):
        dists = square_distance(xyz, xyz)  # b x n x n
        knn_idx = dists.argsort()[:, :, : self.k]  # b x n x k
        knn_xyz = index_points(xyz, knn_idx)  # b x n x k x 3

        pre = features  # b x n x f
        x = self.fc1(features)  # b x n x d_model

        q = self.phi(x)  # b x n x d_model
        k = index_points(self.psi(x), knn_idx)  # b x n x k x d_model
        v = index_points(self.alpha(x), knn_idx)  # b x n x k x d_model

        pos_enc = self.fc_delta(xyz[:, :, None] - knn_xyz)  # b x n x k x d_model

        attn = self.fc_gamma(q[:, :, None] - k + pos_enc)  # b x n x k x d_model
        attn = F.softmax(attn / np.sqrt(k.size(-1)), dim=-2)  # b x n x k x d_model

        res = torch.einsum("bmnf,bmnf->bmf", attn, v + pos_enc)  # b x n x d_model
        res = self.fc2(res) + pre  # b x n x f

        # Visualize attention for the first 50 calculations
        # with self.lock:
        #     if self.attn_counter == 0:
        #         visualize_attention_plain(xyz, attn, self.target_batch_index, self.attn_counter)
        #         visualize_attention(xyz, attn, self.target_batch_index, self.attn_counter)
        #         self.attn_counter += 1

        return res, attn

# Transition down & Transition up
def farthest_point_sample_batch(xyz, npoint):
    """
    Input:
        xyz: pointcloud data, [B, N, 3]
        npoint: number of samples
    Return:
        centroids: sampled pointcloud index, [B, npoint]
    """
    device = xyz.device
    B, N, C = xyz.shape
    centroids = torch.zeros(B, npoint, dtype=torch.long).to(device, non_blocking=True)
    distance = torch.ones(B, N).to(device, non_blocking=True) * 1e10
    farthest = torch.randint(0, N, (B,), dtype=torch.long).to(device, non_blocking=True)
    batch_indices = torch.arange(B, dtype=torch.long).to(device, non_blocking=True)
    for i in range(npoint):
        centroids[:, i] = farthest
        centroid = xyz[batch_indices, farthest, :].view(B, 1, 3)
        dist = torch.sum((xyz - centroid) ** 2, -1)
        distance = torch.min(distance, dist)
        farthest = torch.max(distance, -1)[1]
    return centroids.to(device, non_blocking=True)

class TransitionDown(nn.Module):
    def __init__(self, npoint, k, input_dim, output_dim) -> None:
        """
        npoint: target number of points after transition down
        nneighbor: number of neighbors to max pool the new features from
        input_dim: dimension of input features for each point
        outut_dim: dimension of output features for each point
        """
        super().__init__()
        self.npoint = npoint
        self.k = k
        self.mlp_convs = nn.ModuleList(
            [nn.Conv2d(input_dim, output_dim, 1), nn.Conv2d(output_dim, output_dim, 1)]
        )
        self.mlp_bns = nn.ModuleList(
            [nn.BatchNorm2d(output_dim), nn.BatchNorm2d(output_dim)]
        )

    def forward(self, xyz, features):
        """
        Input:
            xyz: input points position data, [B, N, 3]
            features: input points data, [B, N, D]
        Return:
            new_xyz: sampled points position data, [B, S, 3]
            new_features: new points feature data, [B, S, D']
        """
        fps_idx = farthest_point_sample_batch(xyz, self.npoint)  # B x npoint
        torch.cuda.empty_cache()
        new_xyz = index_points(xyz, fps_idx)  # B x npoint x 3
        torch.cuda.empty_cache()
        dists = square_distance(new_xyz, xyz)  # B x npoint x N
        idx = dists.argsort()[:, :, : self.k]  # B x npoint x k
        torch.cuda.empty_cache()
        index_points(xyz, idx)  # B x npoint x k x 3
        torch.cuda.empty_cache()

        new_features = index_points(features, idx)  # B x npoint x k x D
        new_features = new_features.permute(0, 3, 2, 1)  # B x D x k x npoint
        for i, conv in enumerate(self.mlp_convs):
            bn = self.mlp_bns[i]
            new_features = F.relu(bn(conv(new_features)))  # B x D' x k x npoint
        new_features, _ = torch.max(new_features, 2)  # B x D' x npoint
        new_features = new_features.transpose(1, 2)  # B x npoint x D'

        return new_xyz, new_features

class Config:
    def __init__(self, **kwargs):
        self.__dict__.update(kwargs)


cfg = Config(
    model=Config(nneighbor=16, nblocks=4, transformer_dim=64),
    batch_size=128,
    epochs=200,
    learning_rate=5e-3,
    gpu=5,
    num_point=1024,
    optimizer="SGD",
    weight_decay=1e-4,
    normal=True,
)
cfg.num_class = 5
cfg.input_dim = 6 if cfg.normal else 3

class PointTransformerClassifier(nn.Module):
    def __init__(self, cfg) -> None:
        super().__init__()
        npoints, nblocks, nneighbor, n_c, d_points = (
            cfg.num_point,
            cfg.model.nblocks,
            cfg.model.nneighbor,
            cfg.num_class,
            cfg.input_dim,
        )
        self.fc1 = nn.Sequential(nn.Linear(d_points, 32), nn.ReLU(), nn.Linear(32, 32))
        self.transformer1 = PointTransformerBlock(
            32, cfg.model.transformer_dim, nneighbor
        )
        self.transition_downs = nn.ModuleList()
        self.transformers = nn.ModuleList()
        for i in range(nblocks):
            channel = 32 * 2 ** (i + 1)  # 32(d) * blocks
            self.transition_downs.append(
                TransitionDown(
                    npoints // 4 ** (i + 1), nneighbor, channel // 2, channel
                )
            )
            self.transformers.append(
                PointTransformerBlock(channel, cfg.model.transformer_dim, nneighbor)
            )

        self.fc2 = nn.Sequential(
            nn.Linear(32 * 2**nblocks, 256),
            nn.ReLU(),
            nn.Linear(256, 64),
            nn.ReLU(),
            nn.Linear(64, n_c),
        )
        self.nblocks = nblocks

    def forward(self, x):
        xyz = x[..., :3]
        features = self.transformer1(xyz, self.fc1(x))[0]
        for i in range(self.nblocks):
            xyz, features = self.transition_downs[i](xyz, features)
            features = self.transformers[i](xyz, features)[0]
        res = self.fc2(features.mean(1))
        return res, features

# input: numpy array with [N, 3]
def plot_figure(point, title="Example Ceramic in 3D Space"):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection="3d")
    ax.scatter(point[:, 0], point[:, 1], point[:, 2])
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")
    plt.title(title)
    plt.show()
    plt.close()


def reshape_features(features):
    n, D, _ = features.size()  # (n, D, 4)
    features_reshaped = features.view(n, -1)  # (n, 4D)
    return features_reshaped

def maplabel(n):
    try:
        n = int(n)
    except ValueError:
        if n == "accuracy":
            return n
        if n == "macro avg":
            return n
        if n == "weighted avg":
            return n
        else:
            raise ValueError("out of range")

    if n == 0:
        return "DC"
    elif n == 1:
        return "DBR"
    elif n == 2:
        return "DB"
    elif n == 3:
        return "B"
    elif n == 4:
        return "P"
    else:
        raise ValueError("out of range")

label_order = [
    "DC",
    "DBR",
    "DB",
    "B",
    "P",
]

def cluster_and_plot_dendrogram(features_reshaped, labels, epoch, fold):
    Z = linkage(features_reshaped.cpu().detach().numpy(), "ward")
    plt.figure(figsize=(20, 10))

    v = np.vectorize(maplabel)
    ddata = dendrogram(
        Z,
        labels=v(labels.cpu().detach().numpy()),
        orientation="top",
        leaf_rotation="vertical",
        leaf_font_size=7.0,
        color_threshold=0,
        above_threshold_color="black",
    )

    for i, d, c in zip(ddata["icoord"], ddata["dcoord"], ddata["color_list"]):
        y = d[1]
        x = 0.5 * sum(i[1:3])
        if y > 40:
            plt.plot(x, y, "o", c=c)
            plt.annotate(
                "%.3g" % y,
                (x, y),
                xytext=(0, 5),
                textcoords="offset points",
                va="center",
                ha="left",
            )
    
    title = f"Dendrogram - Epoch {epoch+1} - Fold {fold}"
    plt.title(title)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    os.makedirs("output", exist_ok=True)
    
    filename = f"output/dendrogram_epoch{epoch+1}_fold{fold}_{timestamp}.png"
    plt.savefig(filename, dpi=300, format="png")
    plt.close()


def get_label_color_map(labels, colors):
    unique_labels = labels.unique()
    return {label.item(): colors[i] for i, label in enumerate(unique_labels)}


def get_label_marker_map(labels, markers):
    unique_labels = labels.unique()
    return {label.item(): markers[i] for i, label in enumerate(unique_labels)}


def apply_pca_and_plot(features_reshaped, labels, epoch, fold):
    # Perform PCA
    pca = PCA(n_components=2)
    features_pca = pca.fit_transform(features_reshaped.cpu().detach().numpy())

    # Define color map
    colors = ["#025159", "#04BFBF", "#038C8C", "#BF9A78", "#8C452B"]
    markers = ["o", "s", "^", "D", "P"]

    color_map = get_label_color_map(labels, colors)
    marker_map = get_label_marker_map(labels, markers)

    # Plot the PCA results
    plt.figure(figsize=(10, 7))

    for i, label in enumerate(labels):
        plt.scatter(
            features_pca[i, 0],
            features_pca[i, 1],
            s=50,
            color=color_map[label.item()],
            marker=marker_map[label.item()],
            label=maplabel(label.item()),
        )

    # Add legend
    handles, labels = plt.gca().get_legend_handles_labels()
    labels_handles_dict = dict(zip(labels, handles))
    sorted_labels = sorted(
        labels_handles_dict.keys(), key=lambda x: label_order.index(x)
    )
    sorted_handles = [labels_handles_dict[label] for label in sorted_labels]
    plt.legend(sorted_handles, sorted_labels, title="Labels")
    
    title = f"PCA - Epoch {epoch+1} - Fold {fold}"
    plt.title(title)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    os.makedirs("output", exist_ok=True)
    
    filename = f"output/pca_epoch{epoch+1}_fold{fold}_{timestamp}.png"
    plt.savefig(filename, dpi=600, format="png")
    plt.close()

def plot_saliency_map(xyz, saliency, epoch, class_id, sample_name, fold):
    """
    Visualise and save the Grad-CAM saliency map of a point-cloud sample.

    Parameters
    ----------
    xyz : np.ndarray, shape (N, 3)
        The (x, y, z) coordinates of the input point cloud.
    saliency : np.ndarray, shape (N,)
        Grad-CAM importance values per point, normalised to ``[-1, 1]``.
    epoch : int
        1-based epoch index at which the map is generated.
    class_id : int, optional
        Predicted (or target) class ID associated with the sample.
    sample_name : str, optional
        Human-readable identifier of the ceramic sample.
    fold : int, optional
        Cross-validation fold currently being evaluated.
    """

    # Sequential colormap (viridis) over [-1, 1]
    cmap = plt.colormaps.get_cmap("viridis")
    norm = plt.Normalize(vmin=-1, vmax=1)
    colors = cmap(norm(saliency))

    fig = plt.figure()
    ax = fig.add_subplot(111, projection="3d")
    sc = ax.scatter(xyz[:, 0], xyz[:, 1], xyz[:, 2], c=colors, s=8)

    # ax.set_axis_off()  # hide axes/grid
    for axis in [ax.xaxis, ax.yaxis, ax.zaxis]:
        axis.pane.fill = False  # remove background pane
        axis.line.set_visible(False)

    title = f"Grad-CAM Epoch {epoch}"
    if sample_name:
        title = f"{title} - {sample_name}"

    # Append predicted class label if provided
    if class_id is not None:
        try:
            class_label = maplabel(class_id)
        except Exception:
            class_label = str(class_id)
        title = f"{title} | Predicted: {class_label}"

    title = f"{title} (Fold {fold})"
    plt.title(title)

    mappable = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    mappable.set_array([])
    cbar = fig.colorbar(mappable, ax=ax, fraction=0.03, pad=0.07)
    cbar.set_label("Importance (-1=negative, +1=positive)")

    os.makedirs("output", exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    if sample_name:
        filename = f"output/gradcam_{sample_name}_epoch{epoch}_fold{fold}_{timestamp}.png"
    else:
        filename = f"output/gradcam_epoch{epoch}_fold{fold}_{timestamp}.png"
    
    # Append class label to filename if provided (for forced class maps)
    if class_id is not None:
        filename_parts = filename.split('.png')[0]
        filename = f"{filename_parts}_class_{class_id}.png"
    
    plt.savefig(filename, dpi=600, format="png")
    plt.close()


def grad_cam_pointcloud(model, inputs, target_class=None, device=None):
    """
    Compute a Grad-CAM saliency map for a single point-cloud sample.

    Parameters
    ----------
    model : torch.nn.Module
        Trained CeramicNet + PointTransformer network.
    inputs : torch.Tensor, shape (1, N, C)
        A single sample to be evaluated.
    target_class : int, optional
        Class index for which to back-propagate the gradient.  
        If ``None`` (default), the model's predicted class is used.
    device : torch.device, optional
        Device on which the computation is carried out.  
        Defaults to the device of ``model``'s parameters.

    Returns
    -------
    np.ndarray, shape (N,)
        Normalised contribution of each point in the range ``[-1, 1]``.
    """
    model.eval()
    if device is None:
        device = next(model.parameters()).device
    x = inputs.to(device, non_blocking=True).requires_grad_(True)

    # Obtain transformer1 layer, compatible with DataParallel
    net          = model.module if hasattr(model, "module") else model
    target_layer = net.transformer1

    activations, gradients = {}, {}

    # Save activation via forward hook and attach backward hook to capture gradients
    def fwd_hook(mod, _inp, out):
        activations["v"] = out[0]       # out = (features, attn)
        def bwd_hook(grad):
            gradients["v"] = grad
        out[0].register_hook(bwd_hook)
    handle = target_layer.register_forward_hook(fwd_hook)

    # forward
    logits, _ = model(x)
    if target_class is None:
        target_class = int(logits.argmax(dim=1).item())

    # backward (one-hot)
    score = logits[:, target_class].squeeze()
    model.zero_grad()
    score.backward(retain_graph=True)
    handle.remove()

    A   = activations["v"][0]           # (N, F)
    dA  = gradients["v"][0]             # (N, F)

    # 1) channel-wise weights: average gradient over all points
    weights = dA.mean(dim=0)             # (F,)
    # 2) weighted sum over channels to get per-point saliency
    cam = torch.einsum('nf,f->n', A, weights)  # (N,)
    # 3) keep both positive and negative contributions and scale to [-1, 1]
    cam = cam.detach().cpu()
    # Scale by the largest absolute value so that max(|cam|) == 1
    cam = cam / (cam.abs().max() + 1e-8)
    cam = cam.numpy()
    return cam

def visualize(
    model,
    device,
    test_loader,
    features,
    labels,
    epoch,
    fold,
):
    """Run clustering‐/PCA‐plots and Grad-CAM visualisation for one epoch.

    Parameters
    ----------
    model, device : current network / device
    test_loader   : DataLoader holding the test split (needed to fetch samples) 
    features      : torch.Tensor   features of first test batch (for clustering)
    labels        : torch.Tensor   corresponding labels of that batch
    epoch         : int            zero-based epoch counter (as in training loop)
    fold          : int            current CV-fold (for filenames)
    """
    # =====================================================================
    # 1. Define target samples and find their indices in the test dataset
    # =====================================================================
    
    # Regular samples (using predicted class for saliency)
    typical_samples = [
        "DC_No409NN32K67",
        "DBR_No804K14K14",
        "DB_No970O10K40",
        "B_No510O10O9",
        "P_No944NN32K67",
    ]
    
    # Samples for B/DB class comparison (force class_id 2=DB, 3=B)
    bdb_samples = [
        "DB_No885O10K84",
        "B_No692NN32NN32",
        "DB_No64O10K7",
        "B_No854O10K40",
        "DB_No38O10IG78",
        "B_No85O10K7",
    ]

    # Find the indices of target / B-DB samples in the test dataset
    typical_indices = {}
    bdb_indices = {}
    for i, datapath in enumerate(test_loader.dataset.datapath):
        _, file_path = datapath
        filename = os.path.basename(file_path)
        sample_name = os.path.splitext(filename)[0]
        if sample_name in typical_samples:
            typical_indices[sample_name] = i
        if sample_name in bdb_samples:
            bdb_indices[sample_name] = i
    
    print(f"Found {len(typical_indices)} target samples in fold {fold} test set")
    for sample_name in typical_indices:
        print(f"  - {sample_name}")

    if bdb_indices:
        print(f"Found {len(bdb_indices)} B/DB samples in fold {fold} test set")
        for sample_name in bdb_indices:
            print(f"  - {sample_name}")

    # =====================================================================
    # 2. Run clustering and PCA on the first batch features
    # =====================================================================
    f_reshaped = reshape_features(features)
    cluster_and_plot_dendrogram(f_reshaped, labels, epoch, fold)
    apply_pca_and_plot(f_reshaped, labels, epoch, fold)

    # =====================================================================
    # 3. Generate Grad-CAM saliency maps for regular target samples
    # =====================================================================
    print(f"\nCreating saliency maps for {len(typical_indices)} samples at epoch {epoch+1}...")
    with torch.enable_grad():
        for sample_name, idx in typical_indices.items():
            sample_input_np, _ = test_loader.dataset[idx]
            sample_input = (
                torch.from_numpy(sample_input_np)
                .float()
                .unsqueeze(0)
                .to(device, non_blocking=True)
            )

            outputs, _ = model(sample_input)
            _, predicted = torch.max(outputs.data, 1)

            sal = grad_cam_pointcloud(
                model,
                sample_input,
                target_class=predicted[0].item(),
                device=device,
            )

            plot_saliency_map(
                sample_input.detach()[0, :, :3].cpu().numpy(),
                saliency=sal,
                epoch=epoch + 1,
                class_id=predicted[0].item(),
                sample_name=sample_name,
                fold=fold,
            )
            print(f"  Created saliency map for {sample_name}")

    # =====================================================================
    # 4. Generate Grad-CAM with forced classes (2=DB, 3=B) for comparison
    # =====================================================================
    if bdb_indices:
        print(
            f"\nCreating B/DB saliency maps for {len(bdb_indices)} samples at epoch {epoch+1}..."
        )
        with torch.enable_grad():
            for sample_name, idx in bdb_indices.items():
                sample_input_np, _ = test_loader.dataset[idx]
                sample_input = (
                    torch.from_numpy(sample_input_np)
                    .float()
                    .unsqueeze(0)
                    .to(device, non_blocking=True)
                )

                # Get model's prediction
                outputs, _ = model(sample_input)
                _, predicted = torch.max(outputs.data, 1)
                predicted_class = predicted[0].item()

                # Generate Grad-CAM for predicted class
                sal = grad_cam_pointcloud(
                    model,
                    sample_input,
                    target_class=predicted_class,
                    device=device,
                )

                plot_saliency_map(
                    sample_input.detach()[0, :, :3].cpu().numpy(),
                    saliency=sal,
                    epoch=epoch + 1,
                    class_id=predicted_class,
                    sample_name=f"{sample_name}_predicted",
                    fold=fold,
                )
                print(
                    f"  Created saliency map for {sample_name} (predicted class {predicted_class})"
                )

                # Generate Grad-CAM for alternative class (DB if predicted was B, B if predicted was DB)
                alternative_class = 3 if predicted_class == 2 else 2
                sal = grad_cam_pointcloud(
                    model,
                    sample_input,
                    target_class=alternative_class,
                    device=device,
                )

                plot_saliency_map(
                    sample_input.detach()[0, :, :3].cpu().numpy(),
                    saliency=sal,
                    epoch=epoch + 1,
                    class_id=alternative_class,
                    sample_name=f"{sample_name}_alternative",
                    fold=fold,
                )
                print(
                    f"  Created saliency map for {sample_name} (alternative class {alternative_class})"
                )


def train(model, device, cfg, train_loader, test_loader=None, epochs=200, val_step=5, fold=1):

    if epochs is None:
        epochs = cfg.epoch

    criterion = nn.CrossEntropyLoss()
    if cfg.optimizer == "Adam":
        optimizer = torch.optim.Adam(
            model.parameters(),
            lr=cfg.learning_rate,
            betas=(0.9, 0.999),
            eps=1e-08,
            weight_decay=cfg.weight_decay,
        )
    elif cfg.optimizer == "MadGrad":
        optimizer = madgrad.MADGRAD(
            model.parameters(),
            lr=cfg.learning_rate,
            momentum=0.9,
            weight_decay=cfg.weight_decay,
        )
    else:
        optimizer = torch.optim.SGD(
            model.parameters(),
            lr=cfg.learning_rate,
            momentum=0.9,
            weight_decay=cfg.weight_decay,
        )
    scheduler = torch.optim.lr_scheduler.MultiStepLR(
        optimizer, milestones=[epochs * 6 // 10, epochs * 8 // 10], gamma=0.1
    )

    val_accs = []
    train_accs = []
    best_val_acc = -1.0
    loss = 0
    
    for epoch in tqdm(range(epochs), position=0, leave=True):
        model.train()
        correct = total = 0
        for i, data in enumerate(train_loader, 0):
            inputs, labels = data
            inputs, labels = (
                inputs.to(device, non_blocking=True),
                labels.to(device, non_blocking=True).squeeze(),
            )
            optimizer.zero_grad()
            outputs, features = model(inputs)
            loss = criterion(outputs, torch.squeeze(labels).long())
            loss.backward()
            optimizer.step()
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
        train_acc = 100.0 * correct / total
        train_accs.append(train_acc)

        if (epoch + 1) % val_step == 0:
            model.eval()
            correct = total = 0
            all_labels = []
            all_preds = []
            if test_loader:
                with torch.no_grad():
                    for i, data in enumerate(test_loader):
                        inputs, labels = data
                        inputs, labels = (
                            inputs.to(device, non_blocking=True),
                            labels.to(device, non_blocking=True).squeeze(),
                        )
                        outputs, features = model(inputs)
                        _, predicted = torch.max(outputs.data, 1)
                        total += labels.size(0)
                        correct += (predicted == labels).sum().item()
                        all_labels.extend(labels.cpu().numpy())
                        all_preds.extend(predicted.cpu().numpy())

                        if (
                            epoch == (val_step - 1)
                            or epoch == (val_step * 3 - 1)
                            or epoch == (val_step * 5 - 1)
                            or epoch == (val_step * 10 - 1)
                            or epoch == (epochs / 2 - 1)
                            or epoch == epochs - 1
                        ) and i == 0:
                            visualize(
                                model,
                                device,
                                test_loader,
                                features,
                                labels,
                                epoch,
                                fold,
                            )

                val_acc = 100.0 * correct / total
                val_accs.append(val_acc)
                report = classification_report(
                    all_labels, all_preds, output_dict=True, zero_division=0
                )

                if epoch == epochs - 1:
                    report_df = pd.DataFrame(report).transpose()
                    report_df.index = [maplabel(idx) for idx in report_df.index]
                    print("\nMetrics:")
                    print(report_df)

                    v = np.vectorize(maplabel)
                    vlabels = v(all_labels)
                    vpreds = v(all_preds)
                    all_classes = np.unique(np.concatenate((vlabels, vpreds)))
                    conf_matrix = confusion_matrix(vlabels, vpreds, labels=all_classes)
                    conf_matrix_df = pd.DataFrame(conf_matrix, index=all_classes, columns=all_classes)
                    conf_matrix_df.reindex(index=label_order, columns=label_order)
                    print("\nConfusion Matrix:")
                    print(conf_matrix_df)

                print(
                    "\n Epoch: %d, Train accuracy: %.1f %%, Test accuracy: %.1f %%"
                    % (epoch + 1, train_acc, val_acc)
                )
            if val_accs[-1] > best_val_acc:
                torch.save(model.state_dict(), "checkpoint.pth")
        else:
            print("\n Epoch: %d, Train accuracy: %.1f %%" % (epoch + 1, train_acc))

        scheduler.step()

    return train_accs, val_accs, all_labels, all_preds, report


kfold_sample()

loaders = []
for i in range(fold_num):
    train_loader = torch.utils.data.DataLoader(
        CeramicNetDataLoader(
            root=local_base_dir,
            split="train",
            process_data=True,
            transforms=None,
            use_uniform_sample=False,
            fold=i+1,
        ),
        batch_size=cfg.batch_size,
        shuffle=True,
        pin_memory=True,
    )

    test_loader = torch.utils.data.DataLoader(
        CeramicNetDataLoader(
            root=local_base_dir,
            split="test",
            process_data=True,
            transforms=None,
            use_uniform_sample=False,
            fold=i+1,
        ),
        batch_size=128,
        shuffle=True,
        pin_memory=True,
    )
    loaders.append((train_loader, test_loader))


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
results = []
all_labels = []
all_preds = []
all_reports = []
val_step = 5

for i, (train_loader, test_loader) in enumerate(loaders):
    print(f"current fold is {i+1}")
    # initialize model
    model = torch.nn.DataParallel(PointTransformerClassifier(cfg), device_ids=[0])
    model.to(device, non_blocking=True)

    # train
    [train_accs, val_accs, labels, preds, report] = train(
        model, device, cfg, train_loader, test_loader, 200, val_step, fold=i+1
    )
    results.append((train_accs, val_accs))
    all_labels = np.concatenate((all_labels, labels))
    all_preds = np.concatenate((all_preds, preds))
    all_reports.append(report)


precision_dict = {}
recall_dict = {}
f1_dict = {}
macro_precision = []
macro_recall = []
macro_f1 = []

# Process each report in all_reports
for report in all_reports:
    for label, metrics in report.items():
        if label not in ["accuracy", "macro avg", "weighted avg"]:
            if label not in precision_dict:
                precision_dict[label] = []
                recall_dict[label] = []
                f1_dict[label] = []
            precision_dict[label].append(metrics["precision"])
            recall_dict[label].append(metrics["recall"])
            f1_dict[label].append(metrics["f1-score"])
    # Collect macro avg metrics
    macro_precision.append(report["macro avg"]["precision"])
    macro_recall.append(report["macro avg"]["recall"])
    macro_f1.append(report["macro avg"]["f1-score"])

# Compute averages
averages = {
    "precision": {label: np.mean(scores) for label, scores in precision_dict.items()},
    "recall": {label: np.mean(scores) for label, scores in recall_dict.items()},
    "f1-score": {label: np.mean(scores) for label, scores in f1_dict.items()},
}

# Add macro avg to averages
averages["precision"]["macro avg"] = np.mean(macro_precision)
averages["recall"]["macro avg"] = np.mean(macro_recall)
averages["f1-score"]["macro avg"] = np.mean(macro_f1)

averages_df = pd.DataFrame(averages)
averages_df.index = [maplabel(label) for label in averages_df.index]
print("\nAvg Metrics:")
print(averages_df)
print("\nAvg Metrics Latex:")
print(averages_df.to_latex(float_format="%.4f"))

v = np.vectorize(maplabel)
vlabels = v(all_labels)
vpreds = v(all_preds)
all_classes = np.unique(np.concatenate((vlabels, vpreds)))
conf_matrix = confusion_matrix(vlabels, vpreds, labels=all_classes)
conf_matrix_df = pd.DataFrame(conf_matrix, index=all_classes, columns=all_classes)
conf_matrix_df["Total"] = conf_matrix_df.sum(axis=1)
total_row = conf_matrix_df.sum(axis=0)
total_row.name = "Total"
conf_matrix_df = pd.concat([conf_matrix_df, pd.DataFrame(total_row).T])
label_order_total = label_order + ["Total"]
idx_df = conf_matrix_df.reindex(index=label_order_total, columns=label_order_total)

conf_matrix_latex = idx_df.to_latex()
print("\nConfusion Matrix:")
print(idx_df)
print("\nConfusion Matrix Latex:")
print(conf_matrix_latex)

for fold_idx, (train_accs, val_accs) in enumerate(results, start=1):
    plt.figure()
    epochs_range = list(range(1, len(train_accs) + 1))
    plt.plot(epochs_range, train_accs, label="Train accuracy")

    val_epochs_range = [val_step * (i + 1) for i in range(len(val_accs))]
    plt.plot(val_epochs_range, val_accs, label="Test accuracy")

    plt.xlabel("Epoch")
    plt.ylabel("Accuracy (%)")
    plt.legend()

    # Save figure
    os.makedirs("output", exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    plt.savefig(f"output/accuracy_epoch{len(train_accs)}_fold{fold_idx}_{timestamp}.png", dpi=300, format="png")
    plt.close()

