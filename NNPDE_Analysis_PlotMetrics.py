import os
import json

import pandas as pd

import numpy as np

import torch

import matplotlib.pyplot as plt
import seaborn as sns


save_plot = False

### Setting
PDE_name = "HeatEquation"  # "HeatEquation" or "AllenCahnEquation"

g_target_name = "difficult"  # "linear_t", "linear_x", "linear_y", "quadratic_t", "quadratic_x", "tanh_1", "tanh_2", "tanh_5", "difficult"

optimizer_name = "Adam"  # "Adam", "RMSprop", "SGD"

run = "001"

PLOT_SETTING = 'compare'  # "compare" or "pickOne" or "experimental"
COMPARE_beta_or_N = 'N'  # "N" or "beta"

num_epochs_plot = 60000


#######################################################
runnumber = (
    'results' + "/" +
    PDE_name + "/" +
    g_target_name + "/" +
    optimizer_name + "/" +
    run
)


### Load results
results = torch.load(os.path.join(runnumber, 'results.pt'))

# load NN specifications (N and beta) from config.json
config_path = os.path.join(runnumber, 'config.json')
with open(config_path, 'r') as f:
    config = json.load(f)
beta = config['run_params']['beta']
N = config['run_params']['N']

# specifying beta and N depending on PLOT_SETTING
selected_betas = beta
if PLOT_SETTING == 'compare':
    selected_N = N
elif PLOT_SETTING == 'pickOne':
    selected_N = [5000]
elif PLOT_SETTING == 'experimental':
    selected_N = [5000]

beta = np.array(beta)
N = np.array(N)

beta_indices = [i for i, b in enumerate(beta) if b in selected_betas]
N_indices = [i for i, n in enumerate(N) if n in selected_N]

loss_history_TABLE = results['loss_history_TABLE'].clone().detach()
rel_loss_history_TABLE = results['rel_loss_history_TABLE'].clone().detach()
max_error_history_TABLE = results['max_error_history_TABLE'].clone().detach()
rel_max_error_history_TABLE = results['rel_max_error_history_TABLE'].clone().detach()
grad_norm_history_TABLE = results['grad_norm_history_TABLE'].clone().detach()
adjoint_norm_history_TABLE = results['adjoint_norm_history_TABLE'].clone().detach()
best_loss_history_TABLE = results['best_loss_history_TABLE'].clone().detach()
lr_history_TABLE = results['lr_history_TABLE'].clone().detach()


_, num_beta, num_N, num_epochs, num_runs = loss_history_TABLE.shape

if num_epochs_plot is None:
    num_epochs_plot = num_epochs

sqrt_loss_history_TABLE = np.sqrt(np.array(loss_history_TABLE))
sqrt_rel_loss_history_TABLE = np.sqrt(np.array(rel_loss_history_TABLE))
max_error_history_TABLE = np.array(max_error_history_TABLE)
rel_max_error_history_TABLE =np.sqrt(np.array(rel_max_error_history_TABLE))
grad_norm_history_TABLE = np.array(grad_norm_history_TABLE)
adjoint_norm_history_TABLE = np.array(adjoint_norm_history_TABLE)
best_loss_history_TABLE = np.array(best_loss_history_TABLE)
lr_history_TABLE = np.array(lr_history_TABLE)

# Compute the running minimum (best loss so far) along the epoch axis
# Shape: (1, num_beta, num_N, num_epochs, num_runs)
sqrt_best_loss_history_TABLE = np.minimum.accumulate(sqrt_loss_history_TABLE, axis=3)
sqrt_best_rel_loss_history_TABLE = np.minimum.accumulate(sqrt_rel_loss_history_TABLE, axis=3)


### Prepare data for seaborn

# Reshape the data for seaborn
epochs = np.arange(num_epochs)

# Compute summary statistics
sqrt_rel_mean_loss_history = np.mean(sqrt_rel_loss_history_TABLE, axis=4)
sqrt_rel_median_loss_history = np.median(sqrt_rel_loss_history_TABLE, axis=4)
sqrt_rel_min_loss_history = np.min(sqrt_rel_loss_history_TABLE, axis=4)
sqrt_rel_max_loss_history = np.max(sqrt_rel_loss_history_TABLE, axis=4)
sqrt_rel_std_loss_history = np.std(sqrt_rel_loss_history_TABLE, axis=4)

sqrt_best_rel_mean_loss_history = np.mean(sqrt_best_rel_loss_history_TABLE, axis=4)
sqrt_best_rel_median_loss_history = np.median(sqrt_best_rel_loss_history_TABLE, axis=4)
sqrt_best_rel_min_loss_history = np.min(sqrt_best_rel_loss_history_TABLE, axis=4)
sqrt_best_rel_max_loss_history = np.max(sqrt_best_rel_loss_history_TABLE, axis=4)


data = {
    "Epoch": np.tile(epochs, num_runs),
    "Relative RMSE": rel_loss_history_TABLE[0, 0, 0, :, :].flatten(),
    "Run": np.repeat(np.arange(num_runs), len(epochs))
}
df = pd.DataFrame(data)

# Compute the 25th and 75th percentiles for the 50% confidence interval
rel_25th_percentile = np.percentile(rel_loss_history_TABLE, 25, axis=4)
rel_75th_percentile = np.percentile(rel_loss_history_TABLE, 75, axis=4)

# Create a DataFrame for the averaged data
data_avg = {
    "Epoch": epochs,
    "Relative RMSE (Mean)": sqrt_rel_mean_loss_history[0, 0, 0, :],
    "Relative RMSE (25th Percentile)": rel_25th_percentile[0, 0, 0, :],
    "Relative RMSE (75th Percentile)": rel_75th_percentile[0, 0, 0, :]
}
df_avg = pd.DataFrame(data_avg)


### Seaborn Plotting
# Set the figure size
plt.figure(figsize=(10, 6))
# Set the style of seaborn
sns.set_theme(style="whitegrid")
sns.set_theme(style="ticks")
# Set the font scale
sns.set_context("notebook", font_scale=1)
# Set the color palette
if COMPARE_beta_or_N == 'N':
    palette_colors = sns.color_palette("viridis", n_colors=len(N))
elif COMPARE_beta_or_N == 'beta':
    palette_colors = sns.color_palette("tab10", n_colors=len(beta))

palette_linestyles = ['-', '--', '-.', ':']

skip = 1  # plotting only every sth epoch
for jj, j in enumerate(beta):

    if COMPARE_beta_or_N == 'N':
        linestyle = palette_linestyles[jj]
    elif COMPARE_beta_or_N == 'beta':
        color = palette_colors[jj]

    for kk, k in enumerate(N):

        if COMPARE_beta_or_N == 'N':
            color = palette_colors[kk]
            label = fr"$N={N[kk]}$"
        elif COMPARE_beta_or_N == 'beta':
            linestyle = palette_linestyles[kk]
            label = fr"$\beta={beta[jj]}$"

        if k not in selected_N:
            continue

        # plot best mean
        if PLOT_SETTING == 'compare':
            sns.lineplot(
                x=epochs[:num_epochs_plot:skip],
                y=sqrt_best_rel_mean_loss_history[0, jj, kk, :num_epochs_plot:skip],
                color=color,
                alpha=1,
                #label=fr"$N={N[kk]}, \beta={beta[jj]}$",
                label=label,
                linestyle=linestyle
            )
        # plot deviation
        if PLOT_SETTING == 'compare':
            plt.fill_between(
                epochs[:num_epochs_plot:skip],
                sqrt_best_rel_min_loss_history[0, jj, kk, :num_epochs_plot:skip],
                sqrt_best_rel_max_loss_history[0, jj, kk, :num_epochs_plot:skip],
                color=color,
                alpha=0.2
            )

        if PLOT_SETTING == 'pickOne':
            sns.lineplot(
                    x=epochs[:num_epochs_plot:skip],
                    y=sqrt_rel_mean_loss_history[0, jj, kk, :num_epochs_plot:skip],
                    color=color,
                    alpha=1,
                    label=label,
                    linestyle=linestyle
                )
            # plt.scatter(
            #         epochs[:num_epochs_plot:skip],
            #         sqrt_rel_mean_loss_history[0, jj, kk, :num_epochs_plot:skip],
            #         color=color,
            #         alpha=0.05,
            #         s=4,  # size of the dots
            #     )
            # plot standard deviation
            plt.fill_between(
                epochs[:num_epochs_plot:skip],
                sqrt_rel_min_loss_history[0, jj, kk, :num_epochs_plot:skip],
                sqrt_rel_max_loss_history[0, jj, kk, :num_epochs_plot:skip],
                color=color,
                alpha=0.2
            )

        # plot individual runs
        for i in range(num_runs):
            if PLOT_SETTING == 'experimental':
                sns.lineplot(
                    x=epochs[:num_epochs_plot:skip],
                    y=sqrt_rel_loss_history_TABLE[0, jj, kk, :num_epochs_plot:skip, i],
                    #y=sqrt_best_rel_mean_loss_history[0, jj, kk, :num_epochs_plot:skip],
                    color=color,
                    alpha=1,
                    label=label,
                )
                # plt.scatter(
                #     epochs[:num_epochs_plot:skip],
                #     sqrt_rel_loss_history_TABLE[0, jj, kk, :num_epochs_plot:skip, i],
                #     color=color,
                #     alpha=0.01,
                #     s=4,  # size of the dots
                # )
                sns.lineplot(
                    x=epochs[:num_epochs_plot:skip],
                    y=grad_norm_history_TABLE[0, jj, kk, :num_epochs_plot:skip, i],
                    color=color,
                    linestyle=linestyle,
                    alpha=0.5
                )
                sns.lineplot(
                    x=epochs[:num_epochs_plot:skip],
                    y=lr_history_TABLE[0, jj, kk, :num_epochs_plot:skip, i].squeeze(),
                    color=color,
                    linestyle=':',
                    alpha=0.5
                )

plt.xlabel("epoch")
plt.ylabel("relative RMSE")
plt.yscale("log")
plt.gca().set_ylim(top=1e-1)
if PDE_name == "HeatEquation":
    plt.gca().set_ylim(bottom=5e-4)
elif PDE_name == "AllenCahnEquation":
    plt.gca().set_ylim(bottom=1e-3)
plt.grid(True)
plt.tight_layout()
plt.legend(ncol=4, loc='upper right', frameon=True)

# Save (as pdf and png) or show the plot
if save_plot:
    output_dir = os.path.join(runnumber)
    os.makedirs(output_dir, exist_ok=True)
    if PLOT_SETTING == 'compare':
        plt.savefig(os.path.join(output_dir, "relative_RMSE_plot.pdf"), format="pdf")
        plt.savefig(os.path.join(output_dir, "relative_RMSE_plot.png"), format="png", dpi=300)
    elif PLOT_SETTING == 'pickOne':
        plt.savefig(os.path.join(output_dir, f"relative_RMSE_plot_fixedN{selected_N[0]}.pdf"), format="pdf")
        plt.savefig(os.path.join(output_dir, f"relative_RMSE_plot_fixedN{selected_N[0]}.png"), format="png", dpi=300)
else:
    plt.show()


