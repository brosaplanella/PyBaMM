import pybamm
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

plt.rcParams.update({'font.size': 8})

# Figure 10
# EIS_cathode = np.arange(1, 21)
EIS_cathode = [1, 5, 10, 15, 17, 20]
EIS_cathode_file = "/results/LGM50/data/EIS/Cathode02_discharge_03_PEIS_C09_{}_"
m = int(np.ceil(np.sqrt(np.size(EIS_cathode))))
n = int(np.ceil(np.size(EIS_cathode) / m))

fig101, axes101 = plt.subplots(n, m, num=101, figsize=(6, 3.5))
for j in range(0, n):
    for i in range(0, m):
        k = j * m + i
        if k > np.size(EIS_cathode):
            break
        EIS_raw = pd.read_csv(
            pybamm.root_dir() + EIS_cathode_file.format(EIS_cathode[k]) + "raw.csv",
            sep="\t",
            skiprows=6
        ).to_numpy()
        EIS_fit = pd.read_csv(
            pybamm.root_dir() + EIS_cathode_file.format(EIS_cathode[k]) + "fit.csv",
            sep="\t",
            skiprows=6
        ).to_numpy()
        axes101[j, i].plot(EIS_fit[:, 2], -EIS_fit[:, 3], color="red", zorder=1)
        axes101[j, i].scatter(EIS_raw[:, 2], -EIS_raw[:, 3], c="black", s=1, zorder=2)
        axes101[j, i].set_xlabel("Re(Z) ($\Omega$)")
        axes101[j, i].set_ylabel("-Im(Z) ($\Omega$)")
        axes101[j, i].set_xlim(left=0)
        axes101[j, i].set_ylim(bottom=0)
        axes101[j, i].set_title("Cathode {}%".format(EIS_cathode[k] * 5))
plt.tight_layout()

plt.savefig(
    pybamm.root_dir() + "/results/LGM50/figures/fig10a.png",
    dpi=300
)


# EIS_anode = np.arange(1, 21)
EIS_anode = [1, 6, 11, 16, 18, 20]
EIS_anode_file = "/results/LGM50/data/EIS/Anode02_EIS_charge_02_PEIS_C16_{}_"
m = int(np.ceil(np.sqrt(np.size(EIS_anode))))
n = int(np.ceil(np.size(EIS_anode) / m))

fig102, axes102 = plt.subplots(n, m, num=102, figsize=(6, 3.5))  # figsize=(12.8, 9.6)
for j in range(0, n):
    for i in range(0, m):
        k = j * m + i
        if k > np.size(EIS_anode):
            break
        EIS_raw = pd.read_csv(
            pybamm.root_dir() + EIS_anode_file.format(EIS_anode[k]) + "raw.csv",
            sep="\t",
            skiprows=6
        ).to_numpy()
        EIS_fit = pd.read_csv(
            pybamm.root_dir() + EIS_anode_file.format(EIS_anode[k]) + "fit.csv",
            sep="\t",
            skiprows=6
        ).to_numpy()
        axes102[j, i].plot(EIS_fit[:, 2], -EIS_fit[:, 3], color="red", zorder=1)
        axes102[j, i].scatter(EIS_raw[:, 2], -EIS_raw[:, 3], c="black", s=1, zorder=2)
        axes102[j, i].set_xlabel("Re(Z) ($\Omega$)")
        axes102[j, i].set_ylabel("-Im(Z) ($\Omega$)")
        axes102[j, i].set_xlim(left=0)
        axes102[j, i].set_ylim(bottom=0)
        axes102[j, i].set_title("Anode {}%".format(105 - EIS_anode[k] * 5))
plt.tight_layout()

plt.savefig(
    pybamm.root_dir() + "/results/LGM50/figures/fig10b.png",
    dpi=300
)
