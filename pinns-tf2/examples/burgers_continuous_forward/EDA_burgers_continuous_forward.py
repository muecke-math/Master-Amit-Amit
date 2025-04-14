import os
import numpy as np
import matplotlib.pyplot as plt
import pinnstf2

def read_data_fn(root_path):
    """Read and preprocess data from the specified root path.

    :param root_path: The root directory containing the data.
    :return: Processed data will be used in Mesh class.
    """

    data = pinnstf2.utils.load_data(root_path, "burgers_shock.mat")
    exact_u = np.real(data["usol"])
    return {"u": exact_u}

def save_summary(u_data, summary_path):
    """
    Save summary statistics of the data to a text file.
    """
    # Ensure the directory for the summary file exists
    os.makedirs(os.path.dirname(summary_path), exist_ok=True)
    
    summary_text = (
        f"Data shape: {u_data.shape}\n"
        f"Min value: {np.min(u_data)}\n"
        f"Max value: {np.max(u_data)}\n"
        f"Mean value: {np.mean(u_data)}\n"
        f"Standard Deviation: {np.std(u_data)}\n"
    )
    with open(summary_path, 'w') as f:
        f.write(summary_text)
    print(f"Summary statistics saved to: {summary_path}")

def plot_data(u_data, save_dir):
    """
    Visualize the data in three ways:
      1. Histogram of all values.
      2. Heatmap (if data is 2D).
      3. Line plot of a sample profile.
    Saves the plots to the specified save directory.
    """
    # Ensure the save directory exists
    os.makedirs(save_dir, exist_ok=True)
    
    # 1. Histogram of all u values
    plt.figure(figsize=(8, 6))
    plt.hist(u_data.flatten(), bins=50, edgecolor="black")
    plt.title("Distribution of u values")
    plt.xlabel("u value")
    plt.ylabel("Frequency")
    hist_path = os.path.join(save_dir, "histogram.png")
    plt.savefig(hist_path)
    print(f"Histogram saved to: {hist_path}")
    plt.close()
    
    # 2. Heatmap of u values (if u_data is 2D)
    plt.figure(figsize=(8, 6))
    if u_data.ndim == 2:
        im = plt.imshow(u_data, aspect="auto", cmap="viridis")
        plt.colorbar(im)
        plt.title("Heatmap of u values")
        plt.xlabel("Spatial index")
        plt.ylabel("Temporal index (or row)")
    else:
        plt.text(0.5, 0.5, "Data is not 2D", ha="center", va="center")
        plt.title("Heatmap not available")
    heatmap_path = os.path.join(save_dir, "heatmap.png")
    plt.savefig(heatmap_path)
    print(f"Heatmap saved to: {heatmap_path}")
    plt.close()
    
    # 3. Line plot for a sample profile (first row if 2D)
    plt.figure(figsize=(8, 6))
    if u_data.ndim == 2:
        plt.plot(u_data[0, :])
        plt.title("Profile of u values (first row)")
        plt.xlabel("Index")
        plt.ylabel("u value")
    else:
        plt.plot(u_data)
        plt.title("u values")
        plt.xlabel("Index")
        plt.ylabel("u value")
    line_plot_path = os.path.join(save_dir, "line_plot.png")
    plt.savefig(line_plot_path)
    print(f"Line plot saved to: {line_plot_path}")
    plt.close()

def main():
    # Dataset directory (as defined in your config):
    dataset_root = "/home/users/aamit/Project/pinns-tf2/data/"  
    # Directory where EDA outputs will be saved:
    save_dir = "/home/users/aamit/Project/pinns-tf2/examples/burgers_continuous_forward/EDA"
    # Path to save the basic summary statistics
    summary_file_path = os.path.join(save_dir, "data_burgers_continuous_forward.txt")
    
    # Check if the dataset directory exists
    if not os.path.exists(dataset_root):
        print(f"Error: Dataset directory does not exist: {dataset_root}")
        return
    
    print(f"Loading dataset from: {dataset_root}")
    data = read_data_fn(dataset_root)
    u_data = data["u"]
    
    # Print and save basic summary statistics
    print("Data shape:", u_data.shape)
    print("Min value:", np.min(u_data))
    print("Max value:", np.max(u_data))
    print("Mean value:", np.mean(u_data))
    print("Standard Deviation:", np.std(u_data))
    save_summary(u_data, summary_file_path)
    
    # Generate and save visualizations
    plot_data(u_data, save_dir)
    

if __name__ == "__main__":
    main()