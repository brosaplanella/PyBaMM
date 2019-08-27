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

plt.savefig(
    pybamm.root_dir() + "/results/LGM50/figures/fig3a.png",
    dpi=300
)

plt.figure(32)
plt.hist(data_graphite, bins=np.arange(0, 13))
plt.xlim(0, 12)
plt.xlabel("Particle radius ($\mu$m)")
plt.ylabel("Number of observations")
plt.title("Anode: graphite")

plt.savefig(
    pybamm.root_dir() + "/results/LGM50/figures/fig3b.png",
    dpi=300
)

plt.figure(33)
plt.hist(data_silicon, bins=np.arange(0, 4.5, 0.5))
plt.xlim(0, 4)
plt.xlabel("Particle radius ($\mu$m)")
plt.ylabel("Number of observations")
plt.title("Anode: silicon")

plt.savefig(
    pybamm.root_dir() + "/results/LGM50/figures/fig3c.png",
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

plt.figure(61)
plt.plot(
    cathode_dQdE_delithiation.to_numpy()[:, 0],
    cathode_dQdE_delithiation.to_numpy()[:, 1],
    color="blue",
    label="delithiation"
)
plt.plot(
    cathode_dQdE_lithiation.to_numpy()[:, 0],
    cathode_dQdE_lithiation.to_numpy()[:, 1],
    color="red",
    label="lithiation"
)
plt.xlim(3.5, 4.2)
plt.xlabel("Potential (V)")
plt.ylabel("dQ/dE (mAh/V)")
plt.title("Cathode")
plt.legend()

plt.savefig(
    pybamm.root_dir() + "/results/LGM50/figures/fig6a.png",
    dpi=300
)

plt.figure(62)
plt.plot(
    anode_dQdE_delithiation.to_numpy()[:, 0],
    anode_dQdE_delithiation.to_numpy()[:, 1],
    color="blue",
    label="delithiation"
)
plt.plot(
    anode_dQdE_lithiation.to_numpy()[:, 0],
    anode_dQdE_lithiation.to_numpy()[:, 1],
    color="red",
    label="lithiation"
)
plt.xlim(0.0, 0.6)
plt.xlabel("Potential (V)")
plt.ylabel("dQ/dE (mAh/V)")
plt.title("Anode")
plt.legend()

plt.savefig(
    pybamm.root_dir() + "/results/LGM50/figures/fig6b.png",
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

plt.figure(81)
plt.semilogy(
    cathode_diffusivity_delithiation.to_numpy()[:, 0],
    10 ** cathode_diffusivity_delithiation.to_numpy()[:, 1] * 1E-4,
    color="blue", linestyle="None", marker="o", markersize=2, label="delithiation"
)
plt.semilogy(
    cathode_diffusivity_lithiation.to_numpy()[:, 0],
    10 ** cathode_diffusivity_lithiation.to_numpy()[:, 1] * 1E-4,
    color="red", linestyle="None", marker="o", markersize=2, label="lithiation"
)
plt.xlim(0,1)
plt.xlabel("State of Charge")
plt.ylabel("Diffusivity ($\mathrm{m}^2 \mathrm{s}^{-1}$)")
plt.title("Cathode")
plt.legend(loc="upper left")

plt.savefig(
    pybamm.root_dir() + "/results/LGM50/figures/fig8a.png",
    dpi=300
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
plt.legend(loc="upper left")

plt.savefig(
    pybamm.root_dir() + "/results/LGM50/figures/fig8b.png",
    dpi=300
)

plt.figure(83)
plt.semilogy(
    anode_diffusivity_delithiation.to_numpy()[:, 0],
    10 ** anode_diffusivity_delithiation.to_numpy()[:, 1] * 1E-4,
    color="blue", linestyle="None", marker="o", markersize=2, label="delithiation"
)
plt.semilogy(
    anode_diffusivity_lithiation.to_numpy()[:, 0],
    10 ** anode_diffusivity_lithiation.to_numpy()[:, 1] * 1E-4,
    color="red", linestyle="None", marker="o", markersize=2, label="lithiation"
)
plt.xlim(0,1)
plt.xlabel("State of Charge")
plt.ylabel("Diffusivity ($\mathrm{m}^2 \mathrm{s}^{-1}$)")
plt.title("Cathode")
plt.legend(loc="upper left")

plt.savefig(
    pybamm.root_dir() + "/results/LGM50/figures/fig8c.png",
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
plt.legend(loc="upper right")

plt.savefig(
    pybamm.root_dir() + "/results/LGM50/figures/fig8d.png",
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

plt.savefig(
    pybamm.root_dir() + "/results/LGM50/figures/fig10a.png",
    dpi=300
)


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

plt.figure(111)
plt.scatter(
    cathode_exchange_current.to_numpy()[:, 0],
    cathode_exchange_current.to_numpy()[:, 1] * 10,
    c="black", s=5
)
plt.xlabel("State of Charge")
plt.ylabel("Exchange current (A $\mathrm{m}^{-2}$)")
plt.title("Cathode")

plt.savefig(
    pybamm.root_dir() + "/results/LGM50/figures/fig11a.png",
    dpi=300
)

plt.figure(112)
plt.scatter(
    anode_exchange_current.to_numpy()[:, 0],
    anode_exchange_current.to_numpy()[:, 1] * 10,
    c="black", s=5
)
plt.xlabel("State of Charge")
plt.ylabel("Exchange current (A $\mathrm{m}^{-2}$)")
plt.title("Anode")

plt.savefig(
    pybamm.root_dir() + "/results/LGM50/figures/fig11b.png",
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

plt.figure(121)
plt.scatter(
    cathode_EIS_30degC.to_numpy()[:, 0],
    cathode_EIS_30degC.to_numpy()[:, 1],
    c="black", s=5, label="30°C"
)
plt.scatter(
    cathode_EIS_40degC.to_numpy()[:, 0],
    cathode_EIS_40degC.to_numpy()[:, 1],
    c="red", s=5, label="40°C"
)
plt.scatter(
    cathode_EIS_50degC.to_numpy()[:, 0],
    cathode_EIS_50degC.to_numpy()[:, 1],
    c="green", s=5, label="50°C"
)
plt.scatter(
    cathode_EIS_60degC.to_numpy()[:, 0],
    cathode_EIS_60degC.to_numpy()[:, 1],
    c="blue", s=5, label="60°C"
)
plt.xlabel("Re(Z) ($\Omega$)")
plt.ylabel("-Im(Z) ($\Omega$)")
plt.title("Cathode")
plt.legend()

plt.savefig(
    pybamm.root_dir() + "/results/LGM50/figures/fig12a.png",
    dpi=300
)

plt.figure(122)
plt.scatter(
    anode_EIS_30degC.to_numpy()[:, 0],
    anode_EIS_30degC.to_numpy()[:, 1],
    c="black", s=5, label="30°C"
)
plt.scatter(
    anode_EIS_40degC.to_numpy()[:, 0],
    anode_EIS_40degC.to_numpy()[:, 1],
    c="red", s=5, label="40°C"
)
plt.scatter(
    anode_EIS_50degC.to_numpy()[:, 0],
    anode_EIS_50degC.to_numpy()[:, 1],
    c="green", s=5, label="50°C"
)
plt.scatter(
    anode_EIS_60degC.to_numpy()[:, 0],
    anode_EIS_60degC.to_numpy()[:, 1],
    c="blue", s=5, label="60°C"
)
plt.xlabel("Re(Z) ($\Omega$)")
plt.ylabel("-Im(Z) ($\Omega$)")
plt.title("Anode")
plt.legend()

plt.savefig(
    pybamm.root_dir() + "/results/LGM50/figures/fig12b.png",
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


plt.show()
