import os
import numpy as np
import matplotlib.pyplot as plt
import pinnstf2

def read_data_fn(root_path):
    """Read and preprocess data from the specified root path.

    :param root_path: The root directory containing the data.
    :return: Processed data in the form of a PointCloudData object.
    """
    data = pinnstf2.utils.load_data(root_path, "cylinder_nektar_wake.mat")
    x = data["X_star"][:, 0:1]  # N x 1
    y = data["X_star"][:, 1:2]  # N x 1
    t = data["t"]             # T x 1
    U_star = data["U_star"]   # N x 2 x T
    exact_u = U_star[:, 0, :]  # N x T
    exact_v = U_star[:, 1, :]  # N x T
    exact_p = data["p_star"]   # N x T
    return pinnstf2.data.PointCloudData(
        spatial=[x, y], time=[t], solution={"u": exact_u, "v": exact_v, "p": exact_p}
    )

def save_summary_uvp(u_data, v_data, p_data, summary_path):
    """
    Save summary statistics for u, v, and p to a text file.
    """
    os.makedirs(os.path.dirname(summary_path), exist_ok=True)
    
    summary_text = (
        "Summary Statistics for u:\n"
        f"  Data shape: {u_data.shape}\n"
        f"  Min value: {np.min(u_data)}\n"
        f"  Max value: {np.max(u_data)}\n"
        f"  Mean value: {np.mean(u_data)}\n"
        f"  Standard Deviation: {np.std(u_data)}\n\n"
        "Summary Statistics for v:\n"
        f"  Data shape: {v_data.shape}\n"
        f"  Min value: {np.min(v_data)}\n"
        f"  Max value: {np.max(v_data)}\n"
        f"  Mean value: {np.mean(v_data)}\n"
        f"  Standard Deviation: {np.std(v_data)}\n\n"
        "Summary Statistics for p:\n"
        f"  Data shape: {p_data.shape}\n"
        f"  Min value: {np.min(p_data)}\n"
        f"  Max value: {np.max(p_data)}\n"
        f"  Mean value: {np.mean(p_data)}\n"
        f"  Standard Deviation: {np.std(p_data)}\n"
    )
    with open(summary_path, 'w') as f:
        f.write(summary_text)
    print(f"Summary statistics saved to: {summary_path}")

def plot_histogram_uvp(u_data, v_data, p_data, save_dir):
    """
    Generate a figure with histograms for u, v, and p in separate subplots.
    """
    os.makedirs(save_dir, exist_ok=True)
    
    fig, axes = plt.subplots(ncols=3, figsize=(18, 6))
    
    # Histogram for u values
    axes[0].hist(u_data.flatten(), bins=50, edgecolor="black", alpha=0.7)
    axes[0].set_title("Histogram of u values")
    axes[0].set_xlabel("u value")
    axes[0].set_ylabel("Frequency")
    
    # Histogram for v values
    axes[1].hist(v_data.flatten(), bins=50, edgecolor="black", alpha=0.7, color="orange")
    axes[1].set_title("Histogram of v values")
    axes[1].set_xlabel("v value")
    axes[1].set_ylabel("Frequency")
    
    # Histogram for p values
    axes[2].hist(p_data.flatten(), bins=50, edgecolor="black", alpha=0.7, color="green")
    axes[2].set_title("Histogram of p values")
    axes[2].set_xlabel("p value")
    axes[2].set_ylabel("Frequency")
    
    hist_path = os.path.join(save_dir, "histogram_uvp.png")
    plt.tight_layout()
    plt.savefig(hist_path)
    print(f"Histogram saved to: {hist_path}")
    plt.close()

def plot_heatmap_uvp(u_data, v_data, p_data, save_dir):
    """
    Generate a figure with heatmaps for u, v, and p in separate subplots.
    If a dataset is not 2D, a message is displayed instead.
    """
    os.makedirs(save_dir, exist_ok=True)
    
    fig, axes = plt.subplots(ncols=3, figsize=(18, 6))
    
    # Heatmap for u values
    if u_data.ndim == 2:
        im_u = axes[0].imshow(u_data, aspect="auto", cmap="viridis")
        fig.colorbar(im_u, ax=axes[0])
        axes[0].set_title("Heatmap of u values")
        axes[0].set_xlabel("Spatial index")
        axes[0].set_ylabel("Temporal index")
    else:
        axes[0].text(0.5, 0.5, "Data is not 2D", ha="center", va="center")
        axes[0].set_title("Heatmap not available for u")
    
    # Heatmap for v values
    if v_data.ndim == 2:
        im_v = axes[1].imshow(v_data, aspect="auto", cmap="viridis")
        fig.colorbar(im_v, ax=axes[1])
        axes[1].set_title("Heatmap of v values")
        axes[1].set_xlabel("Spatial index")
        axes[1].set_ylabel("Temporal index")
    else:
        axes[1].text(0.5, 0.5, "Data is not 2D", ha="center", va="center")
        axes[1].set_title("Heatmap not available for v")
    
    # Heatmap for p values
    if p_data.ndim == 2:
        im_p = axes[2].imshow(p_data, aspect="auto", cmap="viridis")
        fig.colorbar(im_p, ax=axes[2])
        axes[2].set_title("Heatmap of p values")
        axes[2].set_xlabel("Spatial index")
        axes[2].set_ylabel("Temporal index")
    else:
        axes[2].text(0.5, 0.5, "Data is not 2D", ha="center", va="center")
        axes[2].set_title("Heatmap not available for p")
    
    heatmap_path = os.path.join(save_dir, "heatmap_uvp.png")
    plt.tight_layout()
    plt.savefig(heatmap_path)
    print(f"Heatmap saved to: {heatmap_path}")
    plt.close()

def plot_lineplot_uvp(u_data, v_data, p_data, save_dir):
    """
    Generate a figure with line plots for a sample profile of u, v, and p.
    If the data is 2D, the first row is used as the sample profile.
    """
    os.makedirs(save_dir, exist_ok=True)
    
    fig, axes = plt.subplots(ncols=3, figsize=(18, 6))
    
    # Sample profile for u values
    if u_data.ndim == 2:
        u_line = u_data[0, :]
    else:
        u_line = u_data
    axes[0].plot(u_line, label="u")
    axes[0].set_title("Line Plot of u values (first row)")
    axes[0].set_xlabel("Index")
    axes[0].set_ylabel("u value")
    axes[0].legend()
    
    # Sample profile for v values
    if v_data.ndim == 2:
        v_line = v_data[0, :]
    else:
        v_line = v_data
    axes[1].plot(v_line, label="v", color="orange")
    axes[1].set_title("Line Plot of v values (first row)")
    axes[1].set_xlabel("Index")
    axes[1].set_ylabel("v value")
    axes[1].legend()
    
    # Sample profile for p values
    if p_data.ndim == 2:
        p_line = p_data[0, :]
    else:
        p_line = p_data
    axes[2].plot(p_line, label="p", color="green")
    axes[2].set_title("Line Plot of p values (first row)")
    axes[2].set_xlabel("Index")
    axes[2].set_ylabel("p value")
    axes[2].legend()
    
    lineplot_path = os.path.join(save_dir, "lineplot_uvp.png")
    plt.tight_layout()
    plt.savefig(lineplot_path)
    print(f"Line plot saved to: {lineplot_path}")
    plt.close()

def main():
    # Dataset directory as defined in your config
    dataset_root = "/home/users/aamit/Project/pinns-tf2/data/"
    # Directory where EDA outputs will be saved
    save_dir = "/home/users/aamit/Project/pinns-tf2/examples/navier_stokes/EDA"
    # File path to save summary statistics for u, v, and p
    summary_file_path = os.path.join(save_dir, "data_navier_stokes.txt")
    
    if not os.path.exists(dataset_root):
        print(f"Error: Dataset directory does not exist: {dataset_root}")
        return
    
    print(f"Loading dataset from: {dataset_root}")
    data = read_data_fn(dataset_root)
    # Extract u, v, and p from the solution dictionary
    u_data = data.solution["u"]
    v_data = data.solution["v"]
    p_data = data.solution["p"]
    
    print("u Data shape:", u_data.shape)
    print("u Min value:", np.min(u_data))
    print("u Max value:", np.max(u_data))
    print("u Mean value:", np.mean(u_data))
    print("u Standard Deviation:", np.std(u_data))
    
    # Save summary statistics for u, v, and p
    save_summary_uvp(u_data, v_data, p_data, summary_file_path)
    
    # Generate and save the individual visualizations for u, v, and p
    plot_histogram_uvp(u_data, v_data, p_data, save_dir)
    plot_heatmap_uvp(u_data, v_data, p_data, save_dir)
    plot_lineplot_uvp(u_data, v_data, p_data, save_dir)

if __name__ == "__main__":
    main()