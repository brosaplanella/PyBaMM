#import os
import pybamm
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
#from matplotlib.ticker import ScalarFormatter, AutoMinorLocator
#import matplotlib as mpl


# Figure 3
particle_distribution_graphite = pd.read_csv(
    pybamm.root_dir() + "/results/LGM50/data/particle_distribution_graphite.csv"
)

particle_distribution_silicon = pd.read_csv(
    pybamm.root_dir() + "/results/LGM50/data/particle_distribution_silicon.csv"
)

particle_distribution_NMC = pd.read_csv(
    pybamm.root_dir() + "/results/LGM50/data/particle_distribution_NMC.csv"
)

data_graphite = []
for v in particle_distribution_graphite.to_numpy():
    data_graphite = np.append(data_graphite, np.full(int(v[1]), v[0]))

data_silicon = []
for v in particle_distribution_silicon.to_numpy():
    data_silicon = np.append(data_silicon, np.full(int(v[1]), v[0]))

data_NMC = []
for v in particle_distribution_NMC.to_numpy():
    data_NMC = np.append(data_NMC, np.full(int(v[1]), v[0]))

plt.figure(31)
plt.hist(data_NMC, bins=np.arange(0, 15))
plt.xlim(0, 14)
plt.xlabel("Particle radius ($\mu$m)")
plt.ylabel("Number of observations")
plt.title("Cathode: NMC")

plt.figure(32)
plt.hist(data_graphite, bins=np.arange(0, 13))
plt.xlim(0, 12)
plt.xlabel("Particle radius ($\mu$m)")
plt.ylabel("Number of observations")
plt.title("Anode: graphite")

plt.figure(33)
plt.hist(data_silicon, bins=np.arange(0, 4.5, 0.5))
plt.xlim(0, 4)
plt.xlabel("Particle radius ($\mu$m)")
plt.ylabel("Number of observations")
plt.title("Anode: silicon")


# Figure 5
cathode_0pt02C_OCP = pd.read_csv(
    pybamm.root_dir() + "/results/LGM50/data/cathode_0pt02C_OCP.csv"
)
anode_0pt02C_OCP = pd.read_csv(
    pybamm.root_dir() + "/results/LGM50/data/anode_0pt02C_OCP.csv"
)

A_coin_cathode = np.pi / 4 * 1.48 ** 2  # in cm^2
A_coin_anode = np.pi / 4 * 1.5 ** 2     # in cm^2

plt.figure(51)
plt.plot(
    cathode_0pt02C_OCP.to_numpy()[:, 0] / A_coin_cathode,
    cathode_0pt02C_OCP.to_numpy()[:, 4],
    color="blue"
)
plt.xlabel("Capacity (mA h $\mathrm{cm}^{-2}$)")
plt.ylabel("Potential (V)")
plt.title("Cathode")

plt.savefig(
    pybamm.root_dir() + "/results/LGM50/figures/fig5a.png",
    dpi=300
)

plt.figure(52)
plt.plot(
    -anode_0pt02C_OCP.to_numpy()[:, 0] / A_coin_anode,
    anode_0pt02C_OCP.to_numpy()[:, 3],
    color="red"
)
plt.xlabel("Capacity (mA h $\mathrm{cm}^{-2}$)")
plt.ylabel("Potential (V)")
plt.title("Anode")

plt.savefig(
    pybamm.root_dir() + "/results/LGM50/figures/fig5b.png",
    dpi=300
)


# Figure 6
# How do I generate these plots? What is x = 0 and what is x = 1?

# Figure 7


# Figure 8
cathode_GITT_lithiation = pd.read_csv(
    pybamm.root_dir() + "/results/LGM50/data/cathode_GITT_lithiation.csv"
)
cathode_GITT_delithiation = pd.read_csv(
    pybamm.root_dir() + "/results/LGM50/data/cathode_GITT_delithiation.csv"
)
anode_GITT_lithiation = pd.read_csv(
    pybamm.root_dir() + "/results/LGM50/data/anode_GITT_lithiation.csv"
)
anode_GITT_delithiation = pd.read_csv(
    pybamm.root_dir() + "/results/LGM50/data/anode_GITT_delithiation.csv"
)

plt.figure(82)
plt.plot(
    cathode_GITT_delithiation.to_numpy()[:, 0] / A_coin_cathode,
    cathode_GITT_delithiation.to_numpy()[:, 1],
    color="blue",
    linewidth=0.5,
    label="delithiation"
)
plt.plot(
    cathode_GITT_lithiation.to_numpy()[:, 0] / A_coin_cathode,
    cathode_GITT_lithiation.to_numpy()[:, 1],
    color="red",
    linewidth=0.5,
    label="lithiation"
)
plt.xlabel("Capacity (mA h $\mathrm{cm}^{-2}$)")
plt.ylabel("Potential (V)")
plt.title("Cathode")
plt.legend()

plt.savefig(
    pybamm.root_dir() + "/results/LGM50/figures/fig8a.png",
    dpi=300
)

plt.figure(84)
plt.plot(
    anode_GITT_delithiation.to_numpy()[:, 0] / A_coin_anode,
    anode_GITT_delithiation.to_numpy()[:, 1],
    color="blue",
    linewidth=0.5,
    label="delithiation"
)
plt.plot(
    anode_GITT_lithiation.to_numpy()[:, 0] / A_coin_anode,
    anode_GITT_lithiation.to_numpy()[:, 1],
    color="red",
    linewidth=0.5,
    label="lithiation"
)
plt.xlabel("Capacity (mA h $\mathrm{cm}^{-2}$)")
plt.ylabel("Potential (V)")
plt.title("Anode")
plt.legend()

plt.savefig(
    pybamm.root_dir() + "/results/LGM50/figures/fig8b.png",
    dpi=300
)


# Figure 10
EIS_cathode = np.arange(0, 20)
EIS_cathode_file = "/results/LGM50/data/EIS/Cathode02_discharge_03_PEIS_C09_{}_"
m = int(np.ceil(np.sqrt(np.size(EIS_cathode))))
n = int(np.ceil(np.size(EIS_cathode) / m))

fig101, axes101 = plt.subplots(n, m, num=101, figsize=(12.8, 9.6))
for j in range(0, n):
    for i in range(0, m):
        k = j * m + i + 1
        if k > np.size(EIS_cathode):
            break
        EIS_raw = pd.read_csv(
            pybamm.root_dir() + EIS_cathode_file.format(k) + "raw.csv",
            sep="\t",
            skiprows=6
        ).to_numpy()
        EIS_fit = pd.read_csv(
            pybamm.root_dir() + EIS_cathode_file.format(k) + "fit.csv",
            sep="\t",
            skiprows=6
        ).to_numpy()
        axes101[j, i].plot(EIS_fit[:, 2], -EIS_fit[:, 3], color="red")
        axes101[j, i].scatter(EIS_raw[:, 2], -EIS_raw[:, 3], c="black", s=1)
        axes101[j, i].set_xlabel("Re(Z) ($\Omega$)")
        axes101[j, i].set_ylabel("-Im(Z) ($\Omega$)")
        axes101[j, i].set_title("Cathode {}%".format(k * 5))
plt.tight_layout()


EIS_anode = np.arange(0, 20)
EIS_anode_file = "/results/LGM50/data/EIS/Anode02_EIS_charge_02_PEIS_C16_{}_"
m = int(np.ceil(np.sqrt(np.size(EIS_anode))))
n = int(np.ceil(np.size(EIS_anode) / m))

fig102, axes102 = plt.subplots(n, m, num=102, figsize=(12.8, 9.6))
for j in range(0, n):
    for i in range(0, m):
        k = j * m + i + 1
        if k > np.size(EIS_anode):
            break
        EIS_raw = pd.read_csv(
            pybamm.root_dir() + EIS_anode_file.format(k) + "raw.csv",
            sep="\t",
            skiprows=6
        ).to_numpy()
        EIS_fit = pd.read_csv(
            pybamm.root_dir() + EIS_anode_file.format(k) + "fit.csv",
            sep="\t",
            skiprows=6
        ).to_numpy()
        axes102[j, i].plot(EIS_fit[:, 2], -EIS_fit[:, 3], color="red")
        axes102[j, i].scatter(EIS_raw[:, 2], -EIS_raw[:, 3], c="black", s=1)
        axes102[j, i].set_xlabel("Re(Z) ($\Omega$)")
        axes102[j, i].set_ylabel("-Im(Z) ($\Omega$)")
        axes102[j, i].set_title("Anode {}%".format(105 - k * 5))
plt.tight_layout()

plt.show()

# Figure 11


# Figure 12


# Figure 13


# Figure 14
swagelok_GITT = pd.read_csv(
    pybamm.root_dir() + "/results/LGM50/data/swagelok_GITT.csv"
)

A_swagelok = np.pi / 4 * 1.2 ** 2     # in cm^2

plt.figure(14)
plt.plot(
    swagelok_GITT.to_numpy()[:, 0] / A_swagelok,
    swagelok_GITT.to_numpy()[:, 3],
    color="black",
    label="full"
)
plt.plot(
    swagelok_GITT.to_numpy()[:, 0] / A_swagelok,
    swagelok_GITT.to_numpy()[:, 4],
    color="blue",
    label="cathode"
)
plt.plot(
    swagelok_GITT.to_numpy()[:, 0] / A_swagelok,
    swagelok_GITT.to_numpy()[:, 5],
    color="red",
    label="anode"
)
plt.xlabel("Capacity (mA h $\mathrm{cm}^{-2}$)")
plt.ylabel("Potential (V)")
plt.title("Swagelok cell")
plt.legend()

plt.savefig(
    pybamm.root_dir() + "/results/LGM50/figures/fig14.png",
    dpi=300
)

# Figure 15


plt.show()
