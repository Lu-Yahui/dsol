import os
import numpy as np

header = '''# .PCD v.7 - Point Cloud Data file format
VERSION .7
FIELDS x y z i rgb
SIZE 4 4 4 4 4
TYPE F F F F F
COUNT 1 1 1 1 1
WIDTH $N
HEIGHT 1
VIEWPOINT 0 0 0 1 0 0 0
POINTS $N
DATA ascii\n'''

def load_all_points(data_dir):
    xs, ys, zs, intensities, colors_r, colors_g, colors_b = [], [], [], [], [], [], []
    files = os.listdir(data_dir)
    for f in files:
        filename = os.path.join(data_dir, f)
        with open(filename, "r") as ff:
            for line in ff:
                values = line.strip().split(" ")
                x = float(values[0])
                y = float(values[1])
                z = float(values[2])
                i = float(values[3])
                xs.append(x)
                ys.append(y)
                zs.append(z)
                intensities.append(i)
                colors_r.append(int(values[4]))
                colors_g.append(int(values[5]))
                colors_b.append(int(values[6]))

    k = len(xs)
    with open("map.pcd", "w") as f:
        h = header.replace("$N", f"{k}")
        f.write(h)
        for x, y, z, i, r, g, b in zip(xs, ys, zs, intensities, colors_r, colors_g, colors_b):
            rgb = (r << 16) + (g << 8) + b
            f.write(f"{x} {y} {z} {i} {rgb}\n")
    with open("map.txt", "w") as f:
        for x, y, z, i, r, g, b in zip(xs, ys, zs, intensities, colors_r, colors_g, colors_b):
            f.write(f"{x} {y} {z} {i} {r} {g} {b}\n")

if __name__ == "__main__":
    load_all_points("./data")