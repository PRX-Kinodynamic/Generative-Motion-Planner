from os import path
import imageio

uncertainty_img_name = "uncertainty.png"
samples_img_name = "new_samples.png"
fps = 2

def _get_image_paths(savepath: str, num_iterations: int, img_name: str):
    image_paths = []
    for i in range(num_iterations):
        iteration_dir = path.join(savepath, f"iteration_{i}")
        img_path = path.join(iteration_dir, img_name)
        if path.exists(img_path):
            image_paths.append(img_path)
    return image_paths

def generate_uncertainty_evolution_videos(
    num_iterations: int,
    savepath: str,
):
    uncertainty_images = []

    uncertainty_img_paths = _get_image_paths(savepath, num_iterations, uncertainty_img_name)
    for img_path in uncertainty_img_paths:
        uncertainty_images.append(imageio.imread(img_path))

    if len(uncertainty_images) > 1:
        uncertainty_video_path = path.join(savepath, "uncertainty_evolution.mp4")
        imageio.mimsave(uncertainty_video_path, uncertainty_images, fps=fps)
        print(f"[ adaptive_training/trainer ] Saved uncertainty evolution video to {uncertainty_video_path}")
    else:
        print("[ adaptive_training/trainer ] Not enough uncertainty images to create video.")

def generate_sample_evolution_videos(
    num_iterations: int,
    savepath: str,
):
    samples_images = []

    samples_img_paths = _get_image_paths(savepath, num_iterations, samples_img_name)
    for img_path in samples_img_paths:
        samples_images.append(imageio.imread(img_path))

    if len(samples_images) > 1:
        samples_video_path = path.join(savepath, "samples_evolution.mp4")
        imageio.mimsave(samples_video_path, samples_images, fps=fps)
        print(f"[ adaptive_training/trainer ] Saved samples evolution video to {samples_video_path}")
    else:
        print("[ adaptive_training/trainer ] Not enough new samples images to create video.")