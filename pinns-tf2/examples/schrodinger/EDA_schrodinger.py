import os
import numpy as np
import matplotlib.pyplot as plt
import pinnstf2

def read_data_fn(root_path):
    """Read and preprocess data from the specified root path.

    :param root_path: The root directory containing the data.
    :return: A dictionary containing the processed data for u, v, and h.
    """
    data = pinnstf2.utils.load_data(root_path, "NLS.mat")
    exact = data["uu"]
    exact_u = np.real(exact)
    exact_v = np.imag(exact)
    exact_h = np.sqrt(exact_u**2 + exact_v**2)
    return {"u": exact_u, "v": exact_v, "h": exact_h}

def save_summary_uvw(u_data, v_data, h_data, summary_path):
    """
    Save summary statistics of the u, v, and h data to a text file.
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
        "Summary Statistics for h:\n"
        f"  Data shape: {h_data.shape}\n"
        f"  Min value: {np.min(h_data)}\n"
        f"  Max value: {np.max(h_data)}\n"
        f"  Mean value: {np.mean(h_data)}\n"
        f"  Standard Deviation: {np.std(h_data)}\n"
    )
    with open(summary_path, 'w') as f:
        f.write(summary_text)
    print(f"Summary statistics saved to: {summary_path}")

def plot_histogram_uvw(u_data, v_data, h_data, save_dir):
    """
    Generate a figure with histograms for u, v, and h in separate subplots.
    """
    os.makedirs(save_dir, exist_ok=True)
    
    fig, axes = plt.subplots(ncols=3, figsize=(18, 6))
    
    # Histogram for u values
    axes[0].hist(u_data.flatten(), bins=50, edgecolor="black")
    axes[0].set_title("Histogram of u values")
    axes[0].set_xlabel("u value")
    axes[0].set_ylabel("Frequency")
    
    # Histogram for v values
    axes[1].hist(v_data.flatten(), bins=50, edgecolor="black", color="orange")
    axes[1].set_title("Histogram of v values")
    axes[1].set_xlabel("v value")
    axes[1].set_ylabel("Frequency")
    
    # Histogram for h values
    axes[2].hist(h_data.flatten(), bins=50, edgecolor="black", color="green")
    axes[2].set_title("Histogram of h values")
    axes[2].set_xlabel("h value")
    axes[2].set_ylabel("Frequency")
    
    hist_path = os.path.join(save_dir, "histogram_uvw.png")
    plt.tight_layout()
    plt.savefig(hist_path)
    print(f"Histogram saved to: {hist_path}")
    plt.close()

def plot_heatmap_uvw(u_data, v_data, h_data, save_dir):
    """
    Generate a figure with heatmaps for u, v, and h in separate subplots.
    If data is not 2D, a message is shown instead.
    """
    os.makedirs(save_dir, exist_ok=True)
    
    fig, axes = plt.subplots(ncols=3, figsize=(18, 6))
    
    # Heatmap for u values
    if u_data.ndim == 2:
        im = axes[0].imshow(u_data, aspect="auto", cmap="viridis")
        fig.colorbar(im, ax=axes[0])
        axes[0].set_title("Heatmap of u values")
        axes[0].set_xlabel("Spatial index")
        axes[0].set_ylabel("Temporal index")
    else:
        axes[0].text(0.5, 0.5, "Data is not 2D", ha="center", va="center")
        axes[0].set_title("Heatmap not available for u")
    
    # Heatmap for v values
    if v_data.ndim == 2:
        im = axes[1].imshow(v_data, aspect="auto", cmap="viridis")
        fig.colorbar(im, ax=axes[1])
        axes[1].set_title("Heatmap of v values")
        axes[1].set_xlabel("Spatial index")
        axes[1].set_ylabel("Temporal index")
    else:
        axes[1].text(0.5, 0.5, "Data is not 2D", ha="center", va="center")
        axes[1].set_title("Heatmap not available for v")
    
    # Heatmap for h values
    if h_data.ndim == 2:
        im = axes[2].imshow(h_data, aspect="auto", cmap="viridis")
        fig.colorbar(im, ax=axes[2])
        axes[2].set_title("Heatmap of h values")
        axes[2].set_xlabel("Spatial index")
        axes[2].set_ylabel("Temporal index")
    else:
        axes[2].text(0.5, 0.5, "Data is not 2D", ha="center", va="center")
        axes[2].set_title("Heatmap not available for h")
    
    heatmap_path = os.path.join(save_dir, "heatmap_uvw.png")
    plt.tight_layout()
    plt.savefig(heatmap_path)
    print(f"Heatmap saved to: {heatmap_path}")
    plt.close()

def plot_lineplot_uvw(u_data, v_data, h_data, save_dir):
    """
    Generate a figure with line plots for a sample profile from u, v, and h.
    If the data is 2D, the first row is used as the sample profile.
    """
    os.makedirs(save_dir, exist_ok=True)
    
    fig, axes = plt.subplots(ncols=3, figsize=(18, 6))
    
    # Line plot for u values
    if u_data.ndim == 2:
        u_line = u_data[0, :]
    else:
        u_line = u_data
    axes[0].plot(u_line)
    axes[0].set_title("Line Plot of u values (first row)")
    axes[0].set_xlabel("Index")
    axes[0].set_ylabel("u value")
    
    # Line plot for v values
    if v_data.ndim == 2:
        v_line = v_data[0, :]
    else:
        v_line = v_data
    axes[1].plot(v_line, color="orange")
    axes[1].set_title("Line Plot of v values (first row)")
    axes[1].set_xlabel("Index")
    axes[1].set_ylabel("v value")
    
    # Line plot for h values
    if h_data.ndim == 2:
        h_line = h_data[0, :]
    else:
        h_line = h_data
    axes[2].plot(h_line, color="green")
    axes[2].set_title("Line Plot of h values (first row)")
    axes[2].set_xlabel("Index")
    axes[2].set_ylabel("h value")
    
    lineplot_path = os.path.join(save_dir, "lineplot_uvw.png")
    plt.tight_layout()
    plt.savefig(lineplot_path)
    print(f"Line plot saved to: {lineplot_path}")
    plt.close()

def main():
    # Dataset directory (update this path as needed)
    dataset_root = "/home/users/aamit/Project/pinns-tf2/data/"  
    # Directory where EDA outputs will be saved
    save_dir = "/home/users/aamit/Project/pinns-tf2/examples/schrodinger/EDA"
    # Path to save the summary statistics for u, v, and h
    summary_file_path = os.path.join(save_dir, "data_schrodinger.txt")
    
    # Check if the dataset directory exists
    if not os.path.exists(dataset_root):
        print(f"Error: Dataset directory does not exist: {dataset_root}")
        return
    
    print(f"Loading dataset from: {dataset_root}")
    data = read_data_fn(dataset_root)
    u_data = data["u"]
    v_data = data["v"]
    h_data = data["h"]
    
    # Print and save basic summary statistics for u, v, and h
    print("u Data shape:", u_data.shape)
    print("u Min value:", np.min(u_data))
    print("u Max value:", np.max(u_data))
    print("u Mean value:", np.mean(u_data))
    print("u Standard Deviation:", np.std(u_data))
    
    save_summary_uvw(u_data, v_data, h_data, summary_file_path)
    
    # Generate and save individual visualizations for u, v, and h.
    plot_histogram_uvw(u_data, v_data, h_data, save_dir)
    plot_heatmap_uvw(u_data, v_data, h_data, save_dir)
    plot_lineplot_uvw(u_data, v_data, h_data, save_dir)

if __name__ == "__main__":
    main()