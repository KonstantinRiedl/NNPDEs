### auxiliary plotting functions
import numpy as np
import os
import matplotlib.pyplot as plt

def plot_PDEsolution(u, N, save_plot, title):
    
    # Create the directory to save plots if it doesn't exist
    plot_dir = os.path.join(title)
    os.makedirs(plot_dir, exist_ok=True)

    nt, nx, ny = u.shape

    for n in np.arange(0, nt, min(nt, int(nt/200)+1)):
        plt.imshow(u.detach().numpy()[n, :, :], cmap='viridis', origin='lower')
        plt.colorbar()
        if N is None:
            plt.title(r"Target data $h$")
        else:
            plt.title(rf"Solution $u^N_{{\theta}}$ for $N={N}$ neurons")
        
        plt.clim(vmin=u.min().item(), vmax=u.max().item())
        # Save the plot with the respective n
        if save_plot:
           plt.savefig(os.path.join(plot_dir, f"plot_{str(n).zfill(4)}.png"))
        plt.close()

    # Save a video of the plots using ffmpeg
    if save_plot:
        if N is None:
            video_path = os.path.join(plot_dir, f"{'videoTargetSolution'}.mp4")
        else:
            video_path = os.path.join(plot_dir, f"{'videoSolution'}.mp4")
        os.system(f"ffmpeg -y -loglevel quiet -framerate 10 -i {os.path.join(plot_dir, 'plot_%04d.png')} -c:v libx264 -pix_fmt yuv420p {video_path}")

        # Remove all images used for the video
        for file_name in os.listdir(plot_dir):
            if file_name.startswith("plot_") and file_name.endswith(".png"):
                os.remove(os.path.join(plot_dir, file_name))

def plot_PDEadjoint(uhat, N, save_plot, title):
    
    # Create the directory to save plots if it doesn't exist
    plot_dir = os.path.join(title)
    os.makedirs(plot_dir, exist_ok=True)

    nt, nx, ny = uhat.shape

    for n in np.arange(0, nt, min(nt, int(nt/200)+1)):
        plt.imshow(uhat.detach().numpy()[n, :, :], cmap='viridis', origin='lower')
        plt.colorbar()
        if N is None:
            plt.title(r"Target adjoint $0$")
        else:
            plt.title(rf"Adjoint $\widehat{{u}}^N_{{\theta}}$ for $N={N}$ neurons")
        plt.clim(vmin=uhat.min().item(), vmax=uhat.max().item())
        # Save the plot with the respective n
        if save_plot:
           plt.savefig(os.path.join(plot_dir, f"plot_{str(n).zfill(4)}.png"))
        plt.close()

    # Save a video of the plots using ffmpeg
    if save_plot:
        if N is None:
            video_path = os.path.join(plot_dir, f"{'videoTargetAdjoint'}.mp4")
        else:
            video_path = os.path.join(plot_dir, f"{'videoAdjoint'}.mp4")
        os.system(f"ffmpeg -y -loglevel quiet -framerate 10 -i {os.path.join(plot_dir, 'plot_%04d.png')} -c:v libx264 -pix_fmt yuv420p {video_path}")

        # Remove all images used for the video
        for file_name in os.listdir(plot_dir):
            if file_name.startswith("plot_") and file_name.endswith(".png"):
                os.remove(os.path.join(plot_dir, file_name))

def plot_PDEsourceterm(g, N, save_plot, title):
    
    # Create the directory to save plots if it doesn't exist
    plot_dir = os.path.join(title)
    os.makedirs(plot_dir, exist_ok=True)

    nt, nx, ny = g.shape

    for n in np.arange(0, nt, min(nt, int(nt/200)+1)):
        plt.imshow(g.detach().numpy()[n, :, :], cmap='viridis', origin='lower')
        plt.colorbar()
        plt.clim(vmin=g.min().item(), vmax=g.max().item())
        if N is None:
            plt.title(r"Target source term $g_{\text{target}}$")
        else:
            plt.title(rf"Source term $g^N_{{\theta}}$ for $N={N}$ neurons")
        # Save the plot with the respective n
        if save_plot:
            plt.savefig(os.path.join(plot_dir, f"plot_{str(n).zfill(4)}.png"))
        plt.close()

    # Save a video of the plots using ffmpeg
    if save_plot:
        if N is None:
            video_path = os.path.join(plot_dir, f"{'videoTarget_Sourceterm'}.mp4")
        else:
            video_path = os.path.join(plot_dir, f"{'videoSourceterm'}.mp4")
        os.system(f"ffmpeg -y -loglevel quiet -framerate 10 -i {os.path.join(plot_dir, 'plot_%04d.png')} -c:v libx264 -pix_fmt yuv420p {video_path}")

        # Remove all images used for the video
        for file_name in os.listdir(plot_dir):
            if file_name.startswith("plot_") and file_name.endswith(".png"):
                os.remove(os.path.join(plot_dir, file_name))

def plot_PDEsolutions(u, h, N, save_plot, title):

    # Create the directory to save plots if it doesn't exist
    plot_dir = os.path.join(title)
    os.makedirs(plot_dir, exist_ok=True)

    nt, nx, ny = u.shape

    for n in np.arange(0, nt, min(nt, int(nt/200)+1)):
        fig, axs = plt.subplots(1, 2, figsize=(12, 6))
        axs = np.array(axs).reshape((1, 2))
        axs[0, 0].imshow(u.detach().numpy()[n, :, :], cmap='viridis', origin='lower')
        axs[0, 0].set_title(rf"Solution $u^N_{{\theta}}$ for $N={N}$ neurons")
        axs[0, 1].imshow(h.detach().numpy()[n, :, :], cmap='viridis', origin='lower')
        axs[0, 1].set_title(r"Target data $h$")
        plt.tight_layout()
        vmin = min(u.min().item(), h.min().item())
        vmax = max(u.max().item(), h.max().item())
        im1 = axs[0, 0].imshow(u.detach().numpy()[n, :, :], cmap='viridis', origin='lower', vmin=vmin, vmax=vmax)
        im2 = axs[0, 1].imshow(h.detach().numpy()[n, :, :], cmap='viridis', origin='lower', vmin=vmin, vmax=vmax)
        fig.colorbar(im1, ax=axs[0, :], orientation='vertical', fraction=0.02, pad=0.04)
        # Save the plot with the respective n
        if save_plot:
           plt.savefig(os.path.join(plot_dir, f"plot_{str(n).zfill(4)}.png"))
        plt.close()

    # Save a video of the plots using ffmpeg
    if save_plot:
        video_path = os.path.join(plot_dir, f"{'video_solutions'}.mp4")
        os.system(f"ffmpeg -y -loglevel quiet -framerate 10 -i {os.path.join(plot_dir, 'plot_%04d.png')} -c:v libx264 -pix_fmt yuv420p {video_path}")

        # Remove all images used for the video
        for file_name in os.listdir(plot_dir):
            if file_name.startswith("plot_") and file_name.endswith(".png"):
                os.remove(os.path.join(plot_dir, file_name))

def plot_PDEsourceterms(g, g_target, N, save_plot, title):

    # Create the directory to save plots if it doesn't exist
    plot_dir = os.path.join(title)
    os.makedirs(plot_dir, exist_ok=True)

    nt, nx, ny = g.shape

    for n in np.arange(0, nt, min(nt, int(nt/200)+1)):
        fig, axs = plt.subplots(1, 2, figsize=(12, 6))
        axs = np.array(axs).reshape((1, 2))
        axs[0, 0].imshow(g.detach().numpy()[n, :, :], cmap='viridis', origin='lower')
        axs[0, 0].set_title(rf"Learned source term $g^N_{{\theta}}$ with $N={N}$ neurons")
        axs[0, 1].imshow(g_target.detach().numpy()[n, :, :], cmap='viridis', origin='lower')
        axs[0, 1].set_title(r"Target source term $g_{\text{target}}$")
        plt.tight_layout()
        vmin = min(g.min().item(), g_target.min().item())
        vmax = max(g.max().item(), g_target.max().item())
        im1 = axs[0, 0].imshow(g.detach().numpy()[n, :, :], cmap='viridis', origin='lower', vmin=vmin, vmax=vmax)
        im2 = axs[0, 1].imshow(g_target.detach().numpy()[n, :, :], cmap='viridis', origin='lower', vmin=vmin, vmax=vmax)
        fig.colorbar(im1, ax=axs[0, :], orientation='vertical', fraction=0.02, pad=0.04)
        # Save the plot with the respective n
        if save_plot:
            plt.savefig(os.path.join(plot_dir, f"plot_{str(n).zfill(4)}.png"))
        plt.close()

    # Save a video of the plots using ffmpeg
    if save_plot:
        video_path = os.path.join(plot_dir, f"{'video_sourceterms'}.mp4")
        os.system(f"ffmpeg -y -loglevel quiet -framerate 10 -i {os.path.join(plot_dir, 'plot_%04d.png')} -c:v libx264 -pix_fmt yuv420p {video_path}")

        # Remove all images used for the video
        for file_name in os.listdir(plot_dir):
            if file_name.startswith("plot_") and file_name.endswith(".png"):
                os.remove(os.path.join(plot_dir, file_name))

def plot_PDEall(u, h, g, g_target, N, save_plot, title):
    
    # Create the directory to save plots if it doesn't exist
    plot_dir = os.path.join(title)
    os.makedirs(plot_dir, exist_ok=True)

    nt, nx, ny = u.shape

    for n in np.arange(0, nt, min(nt, int(nt/200)+1)):
        fig, axs = plt.subplots(2, 2, figsize=(12, 12))
        axs = np.array(axs).reshape((2, 2))
        
        # Top row: Source terms
        axs[0, 0].imshow(g.detach().numpy()[n, :, :], cmap='viridis', origin='lower')
        axs[0, 0].set_title(rf"Learned source term $g^N_{{\theta}}$ with $N={N}$ neurons")
        axs[0, 1].imshow(g_target.detach().numpy()[n, :, :], cmap='viridis', origin='lower')
        axs[0, 1].set_title(r"Target source term $g_{\text{target}}$")
        
        # Bottom row: Solutions
        axs[1, 0].imshow(u.detach().numpy()[n, :, :], cmap='viridis', origin='lower')
        axs[1, 0].set_title(rf"Solution $u^N_{{\theta}}$ for $N={N}$ neurons")
        axs[1, 1].imshow(h.detach().numpy()[n, :, :], cmap='viridis', origin='lower')
        axs[1, 1].set_title(r"Target data $h$")
        
        # Set color limits for each row
        vmin_row1 = min(g.min().item(), g_target.min().item())
        vmax_row1 = max(g.max().item(), g_target.max().item())
        vmin_row2 = min(u.min().item(), h.min().item())
        vmax_row2 = max(u.max().item(), h.max().item())

        # Apply color limits and add colorbars for each row
        im1 = axs[0, 0].imshow(g.detach().numpy()[n, :, :], cmap='viridis', origin='lower', vmin=vmin_row1, vmax=vmax_row1)
        im2 = axs[0, 1].imshow(g_target.detach().numpy()[n, :, :], cmap='viridis', origin='lower', vmin=vmin_row1, vmax=vmax_row1)
        fig.colorbar(im1, ax=axs[0, :], orientation='vertical', fraction=0.013, pad=0.04)
        for ax in axs.flat:
            ax.set_xticks([])
            ax.set_yticks([])

        im3 = axs[1, 0].imshow(u.detach().numpy()[n, :, :], cmap='viridis', origin='lower', vmin=vmin_row2, vmax=vmax_row2)
        im4 = axs[1, 1].imshow(h.detach().numpy()[n, :, :], cmap='viridis', origin='lower', vmin=vmin_row2, vmax=vmax_row2)
        fig.colorbar(im3, ax=axs[1, :], orientation='vertical', fraction=0.013, pad=0.04)
        for ax in axs.flat:
            ax.set_xticks([])
            ax.set_yticks([])

        #plt.tight_layout()
        
        # Save the plot with the respective n
        if save_plot:
            plt.savefig(os.path.join(plot_dir, f"plot_{str(n).zfill(4)}.png"))
        
        plt.close()

    # Save a video of the plots using ffmpeg
    if save_plot:
        video_path = os.path.join(plot_dir, f"{'video_all'}.mp4")
        os.system(f"ffmpeg -y -loglevel quiet -framerate 10 -i {os.path.join(plot_dir, 'plot_%04d.png')} -c:v libx264 -pix_fmt yuv420p {video_path}")

    # Remove all images used for the video
    if save_plot:
        for file_name in os.listdir(plot_dir):
            if file_name.startswith("plot_") and file_name.endswith(".png"):
                os.remove(os.path.join(plot_dir, file_name))
