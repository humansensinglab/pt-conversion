# Created by Baole Fang at 11/8/23

import matplotlib.pyplot as plt
import numpy as np
import pymorton as pm
from plyfile import PlyData


def load_ply(path):
    plydata = PlyData.read(path)

    xyz = np.stack((np.asarray(plydata.elements[0]["x"]),
                    np.asarray(plydata.elements[0]["y"]),
                    np.asarray(plydata.elements[0]["z"])), axis=1)
    opacities = np.asarray(plydata.elements[0]["opacity"])[..., np.newaxis]

    features_dc = np.zeros((xyz.shape[0], 3))
    features_dc[:, 0] = np.asarray(plydata.elements[0]["f_dc_0"])
    features_dc[:, 1] = np.asarray(plydata.elements[0]["f_dc_1"])
    features_dc[:, 2] = np.asarray(plydata.elements[0]["f_dc_2"])

    scale_names = [p.name for p in plydata.elements[0].properties if p.name.startswith("scale_")]
    scale_names = sorted(scale_names, key=lambda x: int(x.split('_')[-1]))
    scales = np.zeros((xyz.shape[0], len(scale_names)))
    for idx, attr_name in enumerate(scale_names):
        scales[:, idx] = np.asarray(plydata.elements[0][attr_name])

    rot_names = [p.name for p in plydata.elements[0].properties if p.name.startswith("rot")]
    rot_names = sorted(rot_names, key=lambda x: int(x.split('_')[-1]))
    rots = np.zeros((xyz.shape[0], len(rot_names)))
    for idx, attr_name in enumerate(rot_names):
        rots[:, idx] = np.asarray(plydata.elements[0][attr_name])
    return np.hstack([xyz, features_dc, opacities, scales, rots])


def sort(ply_data):
    points = ply_data[:, :3]
    max_bound = points.max(axis=0)
    min_bound = points.min(axis=0)
    r = max_bound - min_bound
    scale = (1 << 21) - 1
    ipoints = (points - min_bound) / r * scale
    ipoints = ipoints.astype(int)
    mortoncode = []
    for ipoint in ipoints:
        mortoncode.append(pm.interleave(*ipoint.tolist()))

    order = np.array(mortoncode).argsort()
    return ply_data[order]


def ply22D(ply_data: np.ndarray, n_rows: int = 50, n_cols: int = 50, chunk_w: int = 14, chunk_h: int = 14,
           rescale: bool = False):
    chunksize = chunk_w * chunk_h
    ply_data = sort(ply_data)
    n_channels = ply_data.shape[-1]
    output = np.zeros((chunk_h * n_rows, chunk_w * n_cols, n_channels))
    for i in range(len(ply_data) // chunksize):
        x, y = divmod(i, n_cols)
        x *= chunk_h
        y *= chunk_w
        chunk = ply_data[i * chunksize:(i + 1) * chunksize].reshape(chunk_h, chunk_w, n_channels)
        if rescale:
            chunk = normalize(chunk)
        output[x:x + chunk_h, y:y + chunk_w, :] = chunk
    return output


def normalize(x: np.ndarray):
    axis = tuple(range(x.ndim - 1))
    x_max = x.max(axis=axis)
    x_min = x.min(axis=axis)
    return (x - x_min) / (x_max - x_min)


def visualize(ply_data: np.ndarray):
    fig, ax = plt.subplots(2, 3, figsize=(12, 9))

    ax[0, 0].imshow(normalize(ply_data[:, :, 0:3]))
    ax[0, 0].axis('off')
    ax[0, 0].set_title('position')

    ax[0, 1].imshow(normalize(ply_data[:, :, 3:6]))
    ax[0, 1].axis('off')
    ax[0, 1].set_title('color')

    ax[0, 2].imshow(normalize(ply_data[:, :, 6]), cmap='gray')
    ax[0, 2].axis('off')
    ax[0, 2].set_title('opacity')

    ax[1, 0].imshow(normalize(ply_data[:, :, 7:10]))
    ax[1, 0].axis('off')
    ax[1, 0].set_title('scale')

    ax[1, 1].imshow(normalize(ply_data[:, :, 10:14]))
    ax[1, 1].axis('off')
    ax[1, 1].set_title('rotation')

    ax[1, 2].axis('off')

    plt.show()


if __name__ == '__main__':
    ply = load_ply('original_with_densification.ply')
    results = ply22D(ply)
    np.save('original_with_densification.npy', results)
    normalize(results)
    visualize(results)