import pybamm
import numpy as np
from scipy import interpolate, optimize
import pandas as pd
import matplotlib.pyplot as plt


electrodes = ["cathode", "anode"]
OCPs = ["OCP3", "OCP2", "pseudo"]
process = ["ch", "dch"]

data = {}
interpolators = {}
error_funs = {}

data_3e_OCP = pd.read_csv(
    pybamm.root_dir() + "/results/LGM50/data/ElCell_OCP.csv"
)
data_3e_pseudo = pd.read_csv(
    pybamm.root_dir() + "/results/LGM50/data/ElCell_pseudo.csv"
)
data_cathode = pd.read_csv(
    pybamm.root_dir() + "/results/LGM50/data/cathode_OCP_half.csv"
)
data_anode = pd.read_csv(
    pybamm.root_dir() + "/results/LGM50/data/anode_OCP_half.csv"
)

for i in electrodes:
    for j in OCPs:
        for k in process:
            name = i + "_" + j + "_" + k
            data[name] = pd.read_csv(
                pybamm.root_dir() + "/results/LGM50/data/" + name + ".csv"
            )
            interpolators[name] = interpolate.PchipInterpolator(
                data[name].to_numpy()[:, 0],
                data[name].to_numpy()[:, 1],
                extrapolate=True
            )

x_cathode_ch = np.linspace(
    np.max(
        [np.min(data["cathode_OCP2_ch"].to_numpy()[:, 0]),
        np.min(data["cathode_OCP3_ch"].to_numpy()[:, 0])]
    ), np.min(
        [np.max(data["cathode_OCP2_ch"].to_numpy()[:, 0]),
        np.max(data["cathode_OCP3_ch"].to_numpy()[:, 0])]
    ), num=1000
)
x_cathode_dch = np.linspace(
    np.max(
        [np.min(data["cathode_OCP2_dch"].to_numpy()[:, 0]),
        np.min(data["cathode_OCP3_dch"].to_numpy()[:, 0])]
    ), np.min(
        [np.max(data["cathode_OCP2_dch"].to_numpy()[:, 0]),
        np.max(data["cathode_OCP3_dch"].to_numpy()[:, 0])]
    ), num=1000
)

def error_cathode_2vs3(p):
    error = np.linalg.norm(
        interpolators["cathode_OCP2_ch"](p + x_cathode_ch) - 
        interpolators["cathode_OCP3_ch"](x_cathode_ch)
    ) + np.linalg.norm(
        interpolators["cathode_OCP2_dch"](p + x_cathode_dch) - 
        interpolators["cathode_OCP3_dch"](x_cathode_dch)
    )
    return error

x_anode_ch = np.linspace(
    np.max(
        [np.min(data["anode_OCP2_ch"].to_numpy()[:, 0]),
        np.min(data["anode_OCP3_ch"].to_numpy()[:, 0])]
    ), np.min(
        [np.max(data["anode_OCP2_ch"].to_numpy()[:, 0]),
        np.max(data["anode_OCP3_ch"].to_numpy()[:, 0])]
    ), num=1000
)
x_anode_dch = np.linspace(
    np.max(
        [np.min(data["anode_OCP2_dch"].to_numpy()[:, 0]),
        np.min(data["anode_OCP3_dch"].to_numpy()[:, 0])]
    ), np.min(
        [np.max(data["anode_OCP2_dch"].to_numpy()[:, 0]),
        np.max(data["anode_OCP3_dch"].to_numpy()[:, 0])]
    ), num=1000
)


def error_anode_2vs3(p):
    error = np.linalg.norm(
        interpolators["anode_OCP2_ch"](p + x_anode_ch) - 
        interpolators["anode_OCP3_ch"](x_anode_ch)
    ) + np.linalg.norm(
        interpolators["anode_OCP2_dch"](p + x_anode_dch) - 
        interpolators["anode_OCP3_dch"](x_anode_dch)
    )
    return error


optimal_dch = optimize.minimize(
    error_function_dch, (0.15, 0.1), method="Nelder-Mead",
    options={"xatol": 1E-6, "fatol": 1E-6}
)

shift_dch = optimal_dch.x
print("Shift discharge: ", optimal_dch.x)
print("Error discharge: ", optimal_dch.fun)

optimal_ch = optimize.minimize(
    error_function_ch, (0.15, 0.1), method="Nelder-Mead",
    options={"xatol": 1E-6, "fatol": 1E-6}
)

shift_ch = optimal_ch.x
print("Shift discharge: ", optimal_ch.x)
print("Error discharge: ", optimal_ch.fun)

optimal_full = optimize.minimize(
    error_function_full, (0.1, 0.1), method="Nelder-Mead"
)

shift_full = optimal_full.x
print("Shift full: ", optimal_full.x)
print("Error full: ", optimal_full.fun)
print("Error full (dch): ", error_function_dch(shift_full))
print("Error full (ch): ", error_function_ch(shift_full))


plt.figure(151)
plt.scatter(
    full_cell_0pt02C_OCV_dch.to_numpy()[:, 0] / A_coin_cathode,
    full_cell_0pt02C_OCV_dch.to_numpy()[:, 4],
    color="gray", s=3
)
plt.plot(
    full_cell_0pt02C_OCV_dch.to_numpy()[:, 0] / A_coin_cathode,
    interpolated_cathode_lithiation(
        shift_dch[0] + full_cell_0pt02C_OCV_dch.to_numpy()[:, 0] / A_coin_cathode
    )
    - interpolated_anode_delithiation(
        shift_dch[1] + full_cell_0pt02C_OCV_dch.to_numpy()[:, 0] / A_coin_cathode
    ),
    color="black"
)
plt.plot(
    shift_dch[0] + cathode_0pt02C_OCP.to_numpy()[-1:idx_cathode:-1, 0] / A_coin_cathode,
    cathode_0pt02C_OCP.to_numpy()[-1:idx_cathode:-1, 4],
    color="blue"
)
plt.plot(
    -shift_dch[1] - anode_0pt02C_OCP.to_numpy()[-1:idx_anode:-1, 0] / A_coin_anode,
    anode_0pt02C_OCP.to_numpy()[-1:idx_anode:-1, 3],
    color="red"
)
plt.xlabel("Capacity (mA h $\mathrm{cm}^{-2}$)")
plt.ylabel("Potential (V)")
plt.title("Discharge")
plt.ylim(0, 5)


plt.figure(152)
plt.scatter(
    full_cell_0pt02C_OCV_ch.to_numpy()[:, 0] / A_coin_cathode,
    full_cell_0pt02C_OCV_ch.to_numpy()[:, 4],
    color="gray", s=3
)
plt.plot(
    full_cell_0pt02C_OCV_ch.to_numpy()[:, 0] / A_coin_cathode,
    interpolated_cathode_delithiation(
        shift_ch[0] + full_cell_0pt02C_OCV_ch.to_numpy()[:, 0] / A_coin_cathode
    )
    - interpolated_anode_lithiation(
        shift_ch[1] + full_cell_0pt02C_OCV_ch.to_numpy()[:, 0] / A_coin_cathode
    ),
    color="black"
)
plt.plot(
    shift_ch[0] + cathode_0pt02C_OCP.to_numpy()[0:idx_cathode, 0] / A_coin_cathode,
    cathode_0pt02C_OCP.to_numpy()[0:idx_cathode, 4],
    color="blue"
)
plt.plot(
    -shift_ch[1] - anode_0pt02C_OCP.to_numpy()[0:idx_anode, 0] / A_coin_anode,
    anode_0pt02C_OCP.to_numpy()[0:idx_anode, 3],
    color="red"
)
plt.xlabel("Capacity (mA h $\mathrm{cm}^{-2}$)")
plt.ylabel("Potential (V)")
plt.title("Charge")
plt.ylim(0, 5)


plt.figure(153)
plt.scatter(
    full_cell_0pt02C_OCV_ch.to_numpy()[:, 0] / A_coin_cathode,
    full_cell_0pt02C_OCV_ch.to_numpy()[:, 4],
    color="gray", s=3, label="experimental full cell"
)
plt.scatter(
    full_cell_0pt02C_OCV_dch.to_numpy()[:, 0] / A_coin_cathode,
    full_cell_0pt02C_OCV_dch.to_numpy()[:, 4],
    color="gray", s=3
)
plt.plot(
    full_cell_0pt02C_OCV_ch.to_numpy()[idx_ch:-1, 0] / A_coin_cathode,
    interpolated_cathode_delithiation(
        shift_full[0] + full_cell_0pt02C_OCV_ch.to_numpy()[idx_ch:-1, 0] / A_coin_cathode
    )
    - interpolated_anode_lithiation(
        shift_full[1] +
        full_cell_0pt02C_OCV_ch.to_numpy()[idx_ch:-1, 0] / A_coin_cathode
    ),
    color="black", label="theoretical full cell"
)
plt.plot(
    full_cell_0pt02C_OCV_dch.to_numpy()[0:idx_dch, 0] / A_coin_cathode,
    interpolated_cathode_lithiation(
        shift_full[0] +
        full_cell_0pt02C_OCV_dch.to_numpy()[0:idx_dch, 0] / A_coin_cathode
    )
    - interpolated_anode_delithiation(
        shift_full[1] + full_cell_0pt02C_OCV_dch.to_numpy()[0:idx_dch, 0] / A_coin_cathode
    ),
    color="black"
)
plt.plot(
    shift_full[0] + cathode_0pt02C_OCP.to_numpy()[:, 0] / A_coin_cathode,
    cathode_0pt02C_OCP.to_numpy()[:, 4],
    color="blue", label="cathode"
)
plt.plot(
    -shift_full[1] - anode_0pt02C_OCP.to_numpy()[:, 0] / A_coin_anode,
    anode_0pt02C_OCP.to_numpy()[:, 3],
    color="red", label="anode"
)
plt.xlabel("Capacity (mA h $\mathrm{cm}^{-2}$)")
plt.ylabel("Potential (V)")
plt.legend()
plt.ylim(0, 5)

plt.savefig(
    pybamm.root_dir() + "/results/LGM50/figures/fig15a.png",
    dpi=300
)


plt.figure(154)
plt.scatter(
    full_cell_0pt02C_OCV_ch.to_numpy()[:, 0] / A_coin_cathode,
    full_cell_0pt02C_OCV_ch.to_numpy()[:, 4],
    color="gray", s=3, label="experimental full cell"
)
plt.scatter(
    full_cell_0pt02C_OCV_dch.to_numpy()[:, 0] / A_coin_cathode,
    full_cell_0pt02C_OCV_dch.to_numpy()[:, 4],
    color="gray", s=3
)
plt.plot(
    full_cell_0pt02C_OCV_ch.to_numpy()[idx_ch:-1, 0] / A_coin_cathode,
    interpolated_cathode_delithiation(
        shift_full[0] + full_cell_0pt02C_OCV_ch.to_numpy()[idx_ch:-1, 0] / A_coin_cathode
    )
    - interpolated_anode_lithiation(
        shift_full[1] + full_cell_0pt02C_OCV_ch.to_numpy()[idx_ch:-1, 0] / A_coin_cathode
    ),
    color="black", label="theoretical full cell"
)
plt.plot(
    full_cell_0pt02C_OCV_dch.to_numpy()[0:idx_dch, 0] / A_coin_cathode,
    interpolated_cathode_lithiation(
        shift_full[0] +
        full_cell_0pt02C_OCV_dch.to_numpy()[0:idx_dch, 0] / A_coin_cathode
    )
    - interpolated_anode_delithiation(
        shift_full[1] +
        full_cell_0pt02C_OCV_dch.to_numpy()[0:idx_dch, 0] / A_coin_cathode
    ),
    color="black"
)
plt.xlabel("Capacity (mA h $\mathrm{cm}^{-2}$)")
plt.ylabel("Potential (V)")
plt.legend()

plt.savefig(
    pybamm.root_dir() + "/results/LGM50/figures/fig15b.png",
    dpi=300
)

plt.show()