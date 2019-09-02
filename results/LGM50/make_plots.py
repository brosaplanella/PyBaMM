#import os
import pybamm
import numpy as np
from scipy import interpolate, optimize
import pandas as pd
import matplotlib.pyplot as plt
#from matplotlib.ticker import ScalarFormatter, AutoMinorLocator
#import matplotlib as mpl

plt.rcParams.update({'font.size': 8})


# Figure 3

fig3, axes3 = plt.subplots(1, 3, num=3, figsize=(6, 2))
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

axes3[0].hist(data_NMC, bins=np.arange(0, 15))
axes3[0].set_xlim(0, 14)
axes3[0].set_xlabel("Particle radius ($\mu$m)")
axes3[0].set_ylabel("Frequency")
axes3[0].set_title("Cathode: NMC")

axes3[1].hist(data_graphite, bins=np.arange(0, 13))
axes3[1].set_xlim(0, 12)
axes3[1].set_xlabel("Particle radius ($\mu$m)")
axes3[1].set_ylabel("Frequency")
axes3[1].set_title("Anode: graphite")

axes3[2].hist(data_silicon, bins=np.arange(0, 4.5, 0.5))
axes3[2].set_xlim(0, 4)
axes3[2].set_xlabel("Particle radius ($\mu$m)")
axes3[2].set_ylabel("Frequency")
axes3[2].set_title("Anode: silicon")

plt.tight_layout()

plt.savefig(
    pybamm.root_dir() + "/results/LGM50/figures/fig3.png",
    dpi=300
)


# Figure 5
cathode_0pt02C_OCP = pd.read_csv(
    pybamm.root_dir() + "/results/LGM50/data/cathode_0pt02C_OCP.csv"
)
anode_0pt02C_OCP = pd.read_csv(
    pybamm.root_dir() + "/results/LGM50/data/anode_0pt02C_OCP.csv"
)

A_coin_cathode = np.pi / 4 * 1.48 ** 2  # in cm^2
A_coin_anode = np.pi / 4 * 1.5 ** 2     # in cm^2

fig5, axes5 = plt.subplots(1, 2, num=5, figsize=(6, 2.5))
axes5[0].plot(
    cathode_0pt02C_OCP.to_numpy()[:, 0] / A_coin_cathode,
    cathode_0pt02C_OCP.to_numpy()[:, 4],
    color="blue"
)
axes5[0].set_xlabel("Capacity (mA h $\mathrm{cm}^{-2}$)")
axes5[0].set_ylabel("Potential (V)")
axes5[0].set_title("Cathode")

axes5[1].plot(
    -anode_0pt02C_OCP.to_numpy()[:, 0] / A_coin_anode,
    anode_0pt02C_OCP.to_numpy()[:, 3],
    color="red"
)
axes5[1].set_xlabel("Capacity (mA h $\mathrm{cm}^{-2}$)")
axes5[1].set_ylabel("Potential (V)")
axes5[1].set_title("Anode")

plt.tight_layout()

plt.savefig(
    pybamm.root_dir() + "/results/LGM50/figures/fig5.png",
    dpi=300
)


# Figure 6
# How do I generate these plots? What is x = 0 and what is x = 1?

# Figure 7
cathode_dQdE_lithiation = pd.read_csv(
    pybamm.root_dir() + "/results/LGM50/data/cathode_dQdE_lithiation.csv"
)
cathode_dQdE_delithiation = pd.read_csv(
    pybamm.root_dir() + "/results/LGM50/data/cathode_dQdE_delithiation.csv"
)
anode_dQdE_lithiation = pd.read_csv(
    pybamm.root_dir() + "/results/LGM50/data/anode_dQdE_lithiation.csv"
)
anode_dQdE_delithiation = pd.read_csv(
    pybamm.root_dir() + "/results/LGM50/data/anode_dQdE_delithiation.csv"
)

fig7, axes7 = plt.subplots(1, 2, num=7, figsize=(6, 2.5))
axes7[0].plot(
    cathode_dQdE_delithiation.to_numpy()[:, 0],
    cathode_dQdE_delithiation.to_numpy()[:, 1],
    color="blue",
    label="delithiation"
)
axes7[0].plot(
    cathode_dQdE_lithiation.to_numpy()[:, 0],
    cathode_dQdE_lithiation.to_numpy()[:, 1],
    color="red",
    label="lithiation"
)
axes7[0].set_xlim(3.5, 4.2)
axes7[0].set_xlabel("Potential (V)")
axes7[0].set_ylabel("dQ/dE (mAh/V)")
axes7[0].set_title("Cathode")
# axes7[0].legend()

axes7[1].plot(
    anode_dQdE_delithiation.to_numpy()[:, 0],
    anode_dQdE_delithiation.to_numpy()[:, 1],
    color="blue",
    label="delithiation"
)
axes7[1].plot(
    anode_dQdE_lithiation.to_numpy()[:, 0],
    anode_dQdE_lithiation.to_numpy()[:, 1],
    color="red",
    label="lithiation"
)
axes7[1].set_xlim(0.0, 0.6)
axes7[1].set_xlabel("Potential (V)")
axes7[1].set_ylabel("dQ/dE (mAh/V)")
axes7[1].set_title("Anode")
axes7[1].legend()

plt.tight_layout()

plt.savefig(
    pybamm.root_dir() + "/results/LGM50/figures/fig7.png",
    dpi=300
)


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
cathode_diffusivity_lithiation = pd.read_csv(
    pybamm.root_dir() + "/results/LGM50/data/cathode_diffusivity_lithiation.csv"
)
cathode_diffusivity_delithiation = pd.read_csv(
    pybamm.root_dir() + "/results/LGM50/data/cathode_diffusivity_delithiation.csv"
)
anode_diffusivity_lithiation = pd.read_csv(
    pybamm.root_dir() + "/results/LGM50/data/anode_diffusivity_lithiation.csv"
)
anode_diffusivity_delithiation = pd.read_csv(
    pybamm.root_dir() + "/results/LGM50/data/anode_diffusivity_delithiation.csv"
)

fig8, axes8 = plt.subplots(2, 2, num=8, figsize=(6, 4.5))
axes8[0, 0].semilogy(
    cathode_diffusivity_delithiation.to_numpy()[:, 0],
    10 ** cathode_diffusivity_delithiation.to_numpy()[:, 1] * 1E-4,
    color="blue", linestyle="None", marker="o", markersize=2, label="delithiation"
)
axes8[0, 0].semilogy(
    cathode_diffusivity_lithiation.to_numpy()[:, 0],
    10 ** cathode_diffusivity_lithiation.to_numpy()[:, 1] * 1E-4,
    color="red", linestyle="None", marker="o", markersize=2, label="lithiation"
)
axes8[0, 0].set_xlim(0, 1)
axes8[0, 0].set_xlabel("State of Charge")
axes8[0, 0].set_ylabel("Diffusivity ($\mathrm{m}^2 \mathrm{s}^{-1}$)")
axes8[0, 0].set_title("Cathode")
# axes8[0, 0].legend(loc="upper left")

axes8[1, 0].plot(
    cathode_GITT_delithiation.to_numpy()[:, 0] / A_coin_cathode,
    cathode_GITT_delithiation.to_numpy()[:, 1],
    color="blue",
    linewidth=0.5,
    label="delithiation"
)
axes8[1, 0].plot(
    cathode_GITT_lithiation.to_numpy()[:, 0] / A_coin_cathode,
    cathode_GITT_lithiation.to_numpy()[:, 1],
    color="red",
    linewidth=0.5,
    label="lithiation"
)
axes8[1, 0].set_xlabel("Capacity (mA h $\mathrm{cm}^{-2}$)")
axes8[1, 0].set_ylabel("Potential (V)")
axes8[1, 0].set_title("Cathode")
# axes8[0, 1].legend(loc="upper left")

axes8[0, 1].semilogy(
    anode_diffusivity_delithiation.to_numpy()[:, 0],
    10 ** anode_diffusivity_delithiation.to_numpy()[:, 1] * 1E-4,
    color="blue", linestyle="None", marker="o", markersize=2, label="delithiation"
)
axes8[0, 1].semilogy(
    anode_diffusivity_lithiation.to_numpy()[:, 0],
    10 ** anode_diffusivity_lithiation.to_numpy()[:, 1] * 1E-4,
    color="red", linestyle="None", marker="o", markersize=2, label="lithiation"
)
axes8[0, 1].set_xlim(0, 1)
axes8[0, 1].set_xlabel("State of Charge")
axes8[0, 1].set_ylabel("Diffusivity ($\mathrm{m}^2 \mathrm{s}^{-1}$)")
axes8[0, 1].set_title("Anode")
# axes8[1, 0].legend(loc="upper left")

axes8[1, 1].plot(
    anode_GITT_delithiation.to_numpy()[:, 0] / A_coin_anode,
    anode_GITT_delithiation.to_numpy()[:, 1],
    color="blue",
    linewidth=0.5,
    label="delithiation"
)
axes8[1, 1].plot(
    anode_GITT_lithiation.to_numpy()[:, 0] / A_coin_anode,
    anode_GITT_lithiation.to_numpy()[:, 1],
    color="red",
    linewidth=0.5,
    label="lithiation"
)
axes8[1, 1].set_xlabel("Capacity (mA h $\mathrm{cm}^{-2}$)")
axes8[1, 1].set_ylabel("Potential (V)")
axes8[1, 1].set_title("Anode")
axes8[1, 1].legend(loc="upper right")

plt.tight_layout()

plt.savefig(
    pybamm.root_dir() + "/results/LGM50/figures/fig8.png",
    dpi=300
)


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

# Figure 11
cathode_exchange_current = pd.read_csv(
    pybamm.root_dir() + "/results/LGM50/data/cathode_exchange_current.csv"
)
anode_exchange_current = pd.read_csv(
    pybamm.root_dir() + "/results/LGM50/data/anode_exchange_current.csv"
)

fig11, axes11 = plt.subplots(1, 2, num=11, figsize=(6, 2.5))
axes11[0].scatter(
    cathode_exchange_current.to_numpy()[:, 0],
    cathode_exchange_current.to_numpy()[:, 1] * 10,
    c="black", s=5
)
axes11[0].set_xlabel("State of Charge")
axes11[0].set_ylabel("Exchange current (A $\mathrm{m}^{-2}$)")
axes11[0].set_title("Cathode")

axes11[1].scatter(
    anode_exchange_current.to_numpy()[:, 0],
    anode_exchange_current.to_numpy()[:, 1] * 10,
    c="black", s=5
)
axes11[1].set_xlabel("State of Charge")
axes11[1].set_ylabel("Exchange current (A $\mathrm{m}^{-2}$)")
axes11[1].set_title("Anode")

plt.tight_layout()

plt.savefig(
    pybamm.root_dir() + "/results/LGM50/figures/fig11.png",
    dpi=300
)

# Figure 12
cathode_EIS_30degC = pd.read_csv(
    pybamm.root_dir() + "/results/LGM50/data/cathode_EIS_30degC.csv"
)
cathode_EIS_40degC = pd.read_csv(
    pybamm.root_dir() + "/results/LGM50/data/cathode_EIS_40degC.csv"
)
cathode_EIS_50degC = pd.read_csv(
    pybamm.root_dir() + "/results/LGM50/data/cathode_EIS_50degC.csv"
)
cathode_EIS_60degC = pd.read_csv(
    pybamm.root_dir() + "/results/LGM50/data/cathode_EIS_60degC.csv"
)
anode_EIS_30degC = pd.read_csv(
    pybamm.root_dir() + "/results/LGM50/data/anode_EIS_30degC.csv"
)
anode_EIS_40degC = pd.read_csv(
    pybamm.root_dir() + "/results/LGM50/data/anode_EIS_40degC.csv"
)
anode_EIS_50degC = pd.read_csv(
    pybamm.root_dir() + "/results/LGM50/data/anode_EIS_50degC.csv"
)
anode_EIS_60degC = pd.read_csv(
    pybamm.root_dir() + "/results/LGM50/data/anode_EIS_60degC.csv"
)

fig12, axes12 = plt.subplots(1, 2, num=12, figsize=(6, 2.5))
axes12[0].scatter(
    cathode_EIS_30degC.to_numpy()[:, 0],
    cathode_EIS_30degC.to_numpy()[:, 1],
    c="black", s=5, label="30°C"
)
axes12[0].scatter(
    cathode_EIS_40degC.to_numpy()[:, 0],
    cathode_EIS_40degC.to_numpy()[:, 1],
    c="red", s=5, label="40°C"
)
axes12[0].scatter(
    cathode_EIS_50degC.to_numpy()[:, 0],
    cathode_EIS_50degC.to_numpy()[:, 1],
    c="green", s=5, label="50°C"
)
axes12[0].scatter(
    cathode_EIS_60degC.to_numpy()[:, 0],
    cathode_EIS_60degC.to_numpy()[:, 1],
    c="blue", s=5, label="60°C"
)
axes12[0].set_xlabel("Re(Z) ($\Omega$)")
axes12[0].set_ylabel("-Im(Z) ($\Omega$)")
axes12[0].set_title("Cathode")
# axes12[0].legend()

axes12[1].scatter(
    anode_EIS_30degC.to_numpy()[:, 0],
    anode_EIS_30degC.to_numpy()[:, 1],
    c="black", s=5, label="30°C"
)
axes12[1].scatter(
    anode_EIS_40degC.to_numpy()[:, 0],
    anode_EIS_40degC.to_numpy()[:, 1],
    c="red", s=5, label="40°C"
)
axes12[1].scatter(
    anode_EIS_50degC.to_numpy()[:, 0],
    anode_EIS_50degC.to_numpy()[:, 1],
    c="green", s=5, label="50°C"
)
axes12[1].scatter(
    anode_EIS_60degC.to_numpy()[:, 0],
    anode_EIS_60degC.to_numpy()[:, 1],
    c="blue", s=5, label="60°C"
)
axes12[1].set_xlabel("Re(Z) ($\Omega$)")
axes12[1].set_ylabel("-Im(Z) ($\Omega$)")
axes12[1].set_title("Anode")
axes12[1].legend()

plt.tight_layout()

plt.savefig(
    pybamm.root_dir() + "/results/LGM50/figures/fig12.png",
    dpi=300
)

# Figure 13
exchange_current_activation_energy = pd.read_csv(
    pybamm.root_dir() + "/results/LGM50/data/exchange_current_activation_energy.csv"
)
fit_cathode = np.polyfit(
    1000. / exchange_current_activation_energy.to_numpy()[:, 0],
    np.log(exchange_current_activation_energy.to_numpy()[:, 1] * 10),
    deg=1
)
fit_anode = np.polyfit(
    1000. / exchange_current_activation_energy.to_numpy()[:, 0],
    np.log(exchange_current_activation_energy.to_numpy()[:, 2] * 10),
    deg=1
)

plt.figure(13)
plt.plot(
    1000. / exchange_current_activation_energy.to_numpy()[:, 0],
    np.log(exchange_current_activation_energy.to_numpy()[:, 1] * 10),
    color="black", marker='o', markersize=5, linestyle="None"
)
plt.plot(
    1000. / exchange_current_activation_energy.to_numpy()[:, 0],
    np.log(exchange_current_activation_energy.to_numpy()[:, 2] * 10),
    color="black", marker='o', markersize=5, linestyle="None"
)
plt.plot(
    1000. / exchange_current_activation_energy.to_numpy()[:, 0],
    np.polyval(
        fit_cathode,
        1000. / exchange_current_activation_energy.to_numpy()[:, 0]
    ),
    color="blue",
    label="cathode"
)
plt.plot(
    1000. / exchange_current_activation_energy.to_numpy()[:, 0],
    np.polyval(
        fit_anode,
        1000. / exchange_current_activation_energy.to_numpy()[:, 0]
    ),
    color="red",
    label="anode"
)
plt.xlabel("1000/T ($\mathrm{m}^{-1})$")
plt.ylabel("$\log(j_0)$")
plt.legend()

plt.savefig(
    pybamm.root_dir() + "/results/LGM50/figures/fig13.png",
    dpi=300
)

print("Cathode: ", fit_cathode[0] * 8.314 * 1000)
print("Anode: ", fit_anode[0] * 8.314 * 1000)

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
# full_cell_0pt02C_OCV_ch = pd.read_csv(
#     pybamm.root_dir() + "/results/LGM50/data/full_cell_0pt02C_OCV_ch.csv"
# )
# full_cell_0pt02C_OCV_dch = pd.read_csv(
#     pybamm.root_dir() + "/results/LGM50/data/full_cell_0pt02C_OCV_dch.csv"
# )

# idx_cathode = np.argmax(cathode_0pt02C_OCP.to_numpy()[:, 0])
# idx_anode = np.argmax(-anode_0pt02C_OCP.to_numpy()[:, 0])

# interpolated_cathode_delithiation = interpolate.PchipInterpolator(
#     cathode_0pt02C_OCP.to_numpy()[10:idx_cathode, 0] / A_coin_cathode,
#     cathode_0pt02C_OCP.to_numpy()[10:idx_cathode, 4],
#     extrapolate=True
# )

# interpolated_cathode_lithiation = interpolate.PchipInterpolator(
#     cathode_0pt02C_OCP.to_numpy()[-1:idx_cathode:-1, 0] / A_coin_cathode,
#     cathode_0pt02C_OCP.to_numpy()[-1:idx_cathode:-1, 4],
#     extrapolate=True
# )

# interpolated_anode_lithiation = interpolate.PchipInterpolator(
#     -anode_0pt02C_OCP.to_numpy()[0:idx_anode, 0] / A_coin_anode,
#     anode_0pt02C_OCP.to_numpy()[0:idx_anode, 3],
#     extrapolate=True
# )

# interpolated_anode_delithiation = interpolate.PchipInterpolator(
#     -anode_0pt02C_OCP.to_numpy()[-1:idx_anode:-1, 0] / A_coin_anode,
#     anode_0pt02C_OCP.to_numpy()[-1:idx_anode:-1, 3],
#     extrapolate=True
# )


# plt.figure(15)
# plt.plot(
#     cathode_0pt02C_OCP.to_numpy()[:, 0] / A_coin_cathode,
#     #- cathode_0pt02C_OCP.to_numpy()[-1, 0] / A_coin_cathode,
#     cathode_0pt02C_OCP.to_numpy()[:, 4],
#     color="blue", label="cathode"
# )
# plt.plot(
#     -anode_0pt02C_OCP.to_numpy()[:, 0] / A_coin_anode,
#     anode_0pt02C_OCP.to_numpy()[:, 3],
#     color="red", label="anode"
# )
# plt.plot(
#     full_cell_0pt02C_OCV_dch.to_numpy()[:, 0] / A_coin_cathode,
#     full_cell_0pt02C_OCV_dch.to_numpy()[:, 4],
#     color="black", label="full cell"
# )
# plt.plot(
#     full_cell_0pt02C_OCV_ch.to_numpy()[:, 0] / A_coin_cathode,
#     full_cell_0pt02C_OCV_ch.to_numpy()[:, 4],
#     color="black"
# )
# plt.xlabel("Capacity (mA h $\mathrm{cm}^{-2}$)")
# plt.ylabel("Potential (V)")


# def error_function_dch(params):
#     model = interpolated_cathode_lithiation(
#         params[0] + full_cell_0pt02C_OCV_dch.to_numpy()[:, 0] / A_coin_cathode
#     ) - interpolated_anode_delithiation(
#         params[1] + full_cell_0pt02C_OCV_dch.to_numpy()[:, 0] / A_coin_cathode
#     )
#     return np.linalg.norm(model - full_cell_0pt02C_OCV_dch.to_numpy()[:, 4])


# def error_function_ch(params):
#     model = interpolated_cathode_delithiation(
#         params[0] + full_cell_0pt02C_OCV_ch.to_numpy()[:, 0] / A_coin_cathode
#     ) - interpolated_anode_lithiation(
#         params[1] + full_cell_0pt02C_OCV_ch.to_numpy()[:, 0] / A_coin_cathode
#     )
#     return np.linalg.norm(model - full_cell_0pt02C_OCV_ch.to_numpy()[:, 4])

# idx_ch = np.argmax(full_cell_0pt02C_OCV_ch.to_numpy()[:, 0] > 0.5 * A_coin_cathode)
# idx_dch = np.argmax(full_cell_0pt02C_OCV_dch.to_numpy()[:, 0] < 0.5 * A_coin_cathode)

# idx_dch = -1
# idx_ch = 0


# def error_function_full(params):
#     model_dch = interpolated_cathode_lithiation(
#         params[0] + full_cell_0pt02C_OCV_dch.to_numpy()[0:idx_dch, 0] / A_coin_cathode
#     ) - interpolated_anode_delithiation(
#         params[1] + full_cell_0pt02C_OCV_dch.to_numpy()[0:idx_dch, 0] / A_coin_cathode
#     )
#     model_ch = interpolated_cathode_delithiation(
#         params[0] + full_cell_0pt02C_OCV_ch.to_numpy()[idx_ch:-1, 0] / A_coin_cathode
#     ) - interpolated_anode_lithiation(
#         params[1] + full_cell_0pt02C_OCV_ch.to_numpy()[idx_ch:-1, 0] / A_coin_cathode
#     )
#     error = np.linalg.norm(
#         model_dch - full_cell_0pt02C_OCV_dch.to_numpy()[0:idx_dch, 4]
#     ) + np.linalg.norm(
#         model_ch - full_cell_0pt02C_OCV_ch.to_numpy()[idx_ch:-1, 4]
#     )
#     return error


# optimal_dch = optimize.minimize(
#     error_function_dch, (0.15, 0.1), method="Nelder-Mead",
#     options={"xatol": 1E-6, "fatol": 1E-6}
# )

# shift_dch = optimal_dch.x
# print("Shift discharge: ", optimal_dch.x)
# print("Error discharge: ", optimal_dch.fun)

# optimal_ch = optimize.minimize(
#     error_function_ch, (0.15, 0.1), method="Nelder-Mead",
#     options={"xatol": 1E-6, "fatol": 1E-6}
# )

# shift_ch = optimal_ch.x
# print("Shift discharge: ", optimal_ch.x)
# print("Error discharge: ", optimal_ch.fun)

# optimal_full = optimize.minimize(
#     error_function_full, (0.1, 0.1), method="Nelder-Mead"
# )

# shift_full = optimal_full.x
# print("Shift full: ", optimal_full.x)
# print("Error full: ", optimal_full.fun)
# print("Error full (dch): ", error_function_dch(shift_full))
# print("Error full (ch): ", error_function_ch(shift_full))


# plt.figure(151)
# plt.scatter(
#     full_cell_0pt02C_OCV_dch.to_numpy()[:, 0] / A_coin_cathode,
#     full_cell_0pt02C_OCV_dch.to_numpy()[:, 4],
#     color="gray", s=3
# )
# plt.plot(
#     full_cell_0pt02C_OCV_dch.to_numpy()[:, 0] / A_coin_cathode,
#     interpolated_cathode_lithiation(
#         shift_dch[0] + full_cell_0pt02C_OCV_dch.to_numpy()[:, 0] / A_coin_cathode
#     )
#     - interpolated_anode_delithiation(
#         shift_dch[1] + full_cell_0pt02C_OCV_dch.to_numpy()[:, 0] / A_coin_cathode
#     ),
#     color="black"
# )
# plt.plot(
#     shift_dch[0] + cathode_0pt02C_OCP.to_numpy()[-1:idx_cathode:-1, 0] / A_coin_cathode,
#     cathode_0pt02C_OCP.to_numpy()[-1:idx_cathode:-1, 4],
#     color="blue"
# )
# plt.plot(
#     -shift_dch[1] - anode_0pt02C_OCP.to_numpy()[-1:idx_anode:-1, 0] / A_coin_anode,
#     anode_0pt02C_OCP.to_numpy()[-1:idx_anode:-1, 3],
#     color="red"
# )
# plt.xlabel("Capacity (mA h $\mathrm{cm}^{-2}$)")
# plt.ylabel("Potential (V)")
# plt.title("Discharge")
# plt.ylim(0, 5)


# plt.figure(152)
# plt.scatter(
#     full_cell_0pt02C_OCV_ch.to_numpy()[:, 0] / A_coin_cathode,
#     full_cell_0pt02C_OCV_ch.to_numpy()[:, 4],
#     color="gray", s=3
# )
# plt.plot(
#     full_cell_0pt02C_OCV_ch.to_numpy()[:, 0] / A_coin_cathode,
#     interpolated_cathode_delithiation(
#         shift_ch[0] + full_cell_0pt02C_OCV_ch.to_numpy()[:, 0] / A_coin_cathode
#     )
#     - interpolated_anode_lithiation(
#         shift_ch[1] + full_cell_0pt02C_OCV_ch.to_numpy()[:, 0] / A_coin_cathode
#     ),
#     color="black"
# )
# plt.plot(
#     shift_ch[0] + cathode_0pt02C_OCP.to_numpy()[0:idx_cathode, 0] / A_coin_cathode,
#     cathode_0pt02C_OCP.to_numpy()[0:idx_cathode, 4],
#     color="blue"
# )
# plt.plot(
#     -shift_ch[1] - anode_0pt02C_OCP.to_numpy()[0:idx_anode, 0] / A_coin_anode,
#     anode_0pt02C_OCP.to_numpy()[0:idx_anode, 3],
#     color="red"
# )
# plt.xlabel("Capacity (mA h $\mathrm{cm}^{-2}$)")
# plt.ylabel("Potential (V)")
# plt.title("Charge")
# plt.ylim(0, 5)


# plt.figure(153)
# plt.scatter(
#     full_cell_0pt02C_OCV_ch.to_numpy()[:, 0] / A_coin_cathode,
#     full_cell_0pt02C_OCV_ch.to_numpy()[:, 4],
#     color="gray", s=3, label="experimental full cell"
# )
# plt.scatter(
#     full_cell_0pt02C_OCV_dch.to_numpy()[:, 0] / A_coin_cathode,
#     full_cell_0pt02C_OCV_dch.to_numpy()[:, 4],
#     color="gray", s=3
# )
# plt.plot(
#     full_cell_0pt02C_OCV_ch.to_numpy()[idx_ch:-1, 0] / A_coin_cathode,
#     interpolated_cathode_delithiation(
#         shift_full[0] + full_cell_0pt02C_OCV_ch.to_numpy()[idx_ch:-1, 0] / A_coin_cathode
#     )
#     - interpolated_anode_lithiation(
#         shift_full[1] + 
#         full_cell_0pt02C_OCV_ch.to_numpy()[idx_ch:-1, 0] / A_coin_cathode
#     ),
#     color="black", label="theoretical full cell"
# )
# plt.plot(
#     full_cell_0pt02C_OCV_dch.to_numpy()[0:idx_dch, 0] / A_coin_cathode,
#     interpolated_cathode_lithiation(
#         shift_full[0] + 
#         full_cell_0pt02C_OCV_dch.to_numpy()[0:idx_dch, 0] / A_coin_cathode
#     )
#     - interpolated_anode_delithiation(
#         shift_full[1] + full_cell_0pt02C_OCV_dch.to_numpy()[0:idx_dch, 0] / A_coin_cathode
#     ),
#     color="black"
# )
# plt.plot(
#     shift_full[0] + cathode_0pt02C_OCP.to_numpy()[:, 0] / A_coin_cathode,
#     cathode_0pt02C_OCP.to_numpy()[:, 4],
#     color="blue", label="cathode"
# )
# plt.plot(
#     -shift_full[1] - anode_0pt02C_OCP.to_numpy()[:, 0] / A_coin_anode,
#     anode_0pt02C_OCP.to_numpy()[:, 3],
#     color="red", label="anode"
# )
# plt.xlabel("Capacity (mA h $\mathrm{cm}^{-2}$)")
# plt.ylabel("Potential (V)")
# plt.legend()
# plt.ylim(0, 5)

# plt.savefig(
#     pybamm.root_dir() + "/results/LGM50/figures/fig15a.png",
#     dpi=300
# )


# plt.figure(154)
# plt.scatter(
#     full_cell_0pt02C_OCV_ch.to_numpy()[:, 0] / A_coin_cathode,
#     full_cell_0pt02C_OCV_ch.to_numpy()[:, 4],
#     color="gray", s=3, label="experimental full cell"
# )
# plt.scatter(
#     full_cell_0pt02C_OCV_dch.to_numpy()[:, 0] / A_coin_cathode,
#     full_cell_0pt02C_OCV_dch.to_numpy()[:, 4],
#     color="gray", s=3
# )
# plt.plot(
#     full_cell_0pt02C_OCV_ch.to_numpy()[idx_ch:-1, 0] / A_coin_cathode,
#     interpolated_cathode_delithiation(
#         shift_full[0] + full_cell_0pt02C_OCV_ch.to_numpy()[idx_ch:-1, 0] / A_coin_cathode
#     )
#     - interpolated_anode_lithiation(
#         shift_full[1] + full_cell_0pt02C_OCV_ch.to_numpy()[idx_ch:-1, 0] / A_coin_cathode
#     ),
#     color="black", label="theoretical full cell"
# )
# plt.plot(
#     full_cell_0pt02C_OCV_dch.to_numpy()[0:idx_dch, 0] / A_coin_cathode,
#     interpolated_cathode_lithiation(
#         shift_full[0] + 
#         full_cell_0pt02C_OCV_dch.to_numpy()[0:idx_dch, 0] / A_coin_cathode
#     )
#     - interpolated_anode_delithiation(
#         shift_full[1] + 
#         full_cell_0pt02C_OCV_dch.to_numpy()[0:idx_dch, 0] / A_coin_cathode
#     ),
#     color="black"
# )
# plt.xlabel("Capacity (mA h $\mathrm{cm}^{-2}$)")
# plt.ylabel("Potential (V)")
# plt.legend()

# plt.savefig(
#     pybamm.root_dir() + "/results/LGM50/figures/fig15b.png",
#     dpi=300
# )

plt.show()
