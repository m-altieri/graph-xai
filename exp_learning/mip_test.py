import numpy as np
import matplotlib.pyplot as plt


def save_mip_from_angle(binary_mask, output_dir):
    """
    Save Maximum Intensity Projection of a 3D binary mask from an angle.
    """
    # Compute Maximum Intensity Projection along each axis
    mip_xy = np.max(binary_mask, axis=2)
    mip_xz = np.max(binary_mask, axis=1)
    mip_yz = np.max(binary_mask, axis=0)

    # Create subplots for each projection
    fig, axs = plt.subplots(1, 3, figsize=(15, 5))

    # Plot Maximum Intensity Projections from different angles
    axs[0].imshow(np.flipud(np.rot90(mip_xy)), cmap="binary", aspect="auto")
    axs[0].set_title("XY Maximum Intensity Projection")

    axs[1].imshow(np.flipud(np.rot90(mip_xz)), cmap="binary", aspect="auto")
    axs[1].set_title("XZ Maximum Intensity Projection")

    axs[2].imshow(np.flipud(np.rot90(mip_yz.T)), cmap="binary", aspect="auto")
    axs[2].set_title("YZ Maximum Intensity Projection")

    # Save the plots to files
    plt.savefig(f"{output_dir}/mip_xy.png", bbox_inches="tight")
    plt.close()

    plt.savefig(f"{output_dir}/mip_xz.png", bbox_inches="tight")
    plt.close()

    plt.savefig(f"{output_dir}/mip_yz.png", bbox_inches="tight")
    plt.close()


# Example 3D binary mask
binary_mask = np.random.choice(
    [0, 1], size=(19, 7, 11), p=[0.9, 0.1]
)  # Equal probability for 0s and 1s

# Save Maximum Intensity Projection from an angle
output_dir = "mip"
save_mip_from_angle(binary_mask, output_dir)
