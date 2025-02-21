import argparse
import importlib
import os
from os import path
import numpy as np
import time

from tqdm import tqdm

import mg_diffuse.utils.model
from mg_diffuse import utils

def generate_roa_labels_from_success_file(dataset):
    data = []
    
    roa_labels_fpath = path.join("data_trajectories", dataset, "roa_labels.txt")
    success_fpath = path.join("data_trajectories", dataset, "success.txt")
    
    if not os.path.exists(success_fpath):
        print(f"[ scripts/estimate_roa ] Success file {success_fpath} not found")
        raise FileNotFoundError()
    
    print(f"[ scripts/estimate_roa ] Success file found. Converting success file to RoA labels")
    
    with open(success_fpath, "r") as f:
        for line in tqdm(f):
            if len(line.strip()) == 0:
                continue
            line_data = line.strip().split(',')
            sequence_fname = line_data[0]
            label = int(line_data[1])
            
            sequence_fpath = path.join("data_trajectories", dataset, "trajectories", sequence_fname)
            
            if not os.path.exists(sequence_fpath):
                print(f"[ scripts/estimate_roa ] Sequence file {sequence_fpath} not found")
                raise FileNotFoundError()
            
            # Read the first line of the sequence file and get the start point
            with open(sequence_fpath, "r") as seq_f:
                start_point = seq_f.readline().strip().split(',')
                data.append([np.float32(start_point[0]), np.float32(start_point[1]), label])
    
    with open(roa_labels_fpath, "w") as f:
        for i, line_data in enumerate(data):
            f.write(f"{i} {line_data[0]} {line_data[1]} {line_data[2]}\n")

def load_ground_truth(dataset):
    print('[ scripts/estimate_roa ] Loading ground truth')
    roa_labels_fpath = path.join("data_trajectories", dataset, "roa_labels.txt")

    data = []
    
    if os.path.exists(roa_labels_fpath):
        with open(roa_labels_fpath, "r") as f:
            for line in f:
                line_data = line.strip().split(' ')[1:]
                data.append([
                    np.float32(line_data[0]),
                    np.float32(line_data[1]),
                    int(line_data[2])
                ])
    else:
        raise FileNotFoundError(f"File {roa_labels_fpath} not found")
    
    return np.array(data, dtype=np.float32)

def save_attractor_labels(start_points, labels, final_states, file_path):
    with open(file_path, "w") as f:
        f.write("#")
        for i in range(start_points.shape[1]):
            f.write(f"start_{i} ")

        f.write("label ")

        for i in range(final_states.shape[1]):
            f.write(f"final_{i} ")

        f.write("\n")

        for i in range(start_points.shape[0]):
            for j in range(start_points.shape[1]):
                f.write(f"{start_points[i, j]} ")

            f.write(f"{labels[i]} ")

            for j in range(final_states.shape[1]):
                f.write(f"{final_states[i, j]} ")

            f.write("\n")

def load_final_states_data(dir_path, runs, invalid_label):
    all_predicted_labels = []
    all_final_states = []
    print('[ scripts/estimate_roa ] Loading attractor labels')

    invalid_count = 0

    for i in tqdm(range(runs)):
        file_path = path.join(dir_path, f"attractor_labels_{i}.txt")
        try:
            with open(file_path, "r") as f:
                labels = []
                final_states = []
                for line in f:
                    if line[0] == "#":
                        continue

                    line_data = line.strip().split(' ')
                    # start_points.append([np.float32(line_data[0]), np.float32(line_data[1])])
                    if int(float(line_data[2])) == invalid_label:
                        invalid_count += 1
                    labels.append(int(float(line_data[2])))
                    final_states.append([np.float32(line_data[3]), np.float32(line_data[4])])
                all_predicted_labels.append(labels)
                all_final_states.append(final_states)
        except FileNotFoundError:
            print(f"[ scripts/estimate_roa ] File {file_path} not found")
            break
    print(f"Invalid labels count: {invalid_count}")
    return np.array(all_predicted_labels, dtype=np.int32), np.array(all_final_states, dtype=np.float32)

def generate_label_probabilities(all_predicted_labels, labels_set, invalid_label):
    n_points = all_predicted_labels.shape[1]
    n_runs = all_predicted_labels.shape[0]

    label_probabilities = np.zeros((n_points, len(labels_set) + 1))

    unique_labels, counts = np.unique(all_predicted_labels, return_counts=True, axis=0)
    label_probabilities[:, -1] = np.sum(all_predicted_labels == invalid_label, axis=0) / n_runs
    for label in range(len(labels_set)):
        label_probabilities[:, label] = np.sum(all_predicted_labels == label, axis=0) / n_runs

    return label_probabilities

def perform_runs(params, model, model_args, start_points, attractors, final_state_path, n_runs):
    attractor_threshold = params["attractor_threshold"]
    batch_size = params["batch_size"]
    invalid_label = params["invalid_label"]

    all_predicted_labels = []
    trajectories = []

    for i in range(n_runs):
        print(f"[ scripts/estimate_roa ] Run {i+1}/{n_runs}\n\n")

        generated_trajs = utils.generate_trajectories(
            model, model_args, start_points, only_execute_next_step=False, verbose=True, batch_size=batch_size,
        )

        final_states = generated_trajs[:, -1]

        run_predicted_labels = utils.get_trajectory_attractor_labels(final_states, attractors, attractor_threshold, invalid_label)

        save_attractor_labels(start_points, run_predicted_labels, final_states, path.join(final_state_path, f"attractor_labels_{i}.txt"))

        all_predicted_labels.append(run_predicted_labels)
        trajectories.append(generated_trajs)
        
    all_predicted_labels = np.array(all_predicted_labels)
    trajectories = np.array(trajectories)

    # Modify trajectories from (n_runs, n_points, 2) to (n_points, n_runs, 2)
    trajectories = trajectories.transpose(1, 0, 2)
    return trajectories, all_predicted_labels

def load_dataset_params(dataset):
    config = f'config.{dataset}'
    module = importlib.import_module(config)
    params = getattr(module, "base")["roa_estimation"]

    return params

def get_predicted_labels(label_probabilities, attractor_probability_upper_threshold, labels_array, invalid_label):
    predicted_labels = np.zeros(len(label_probabilities), dtype=np.int32)
    # Fill with invalid label
    predicted_labels.fill(invalid_label)

    for i in range(len(labels_array[:-1])):
        predicted_labels[label_probabilities[:, i] > attractor_probability_upper_threshold] = labels_array[i]

    return predicted_labels

def get_labels_array(expected_labels, invalid_label):
    labels_set = sorted(np.unique(expected_labels).tolist())
    labels_array = np.array([*labels_set, invalid_label])

    return labels_set, labels_array

def perform_roa_estimation(dataset, exp_name, timestamp, generate_img, path_prefix="diffusion", n_runs=None):
    print('[ scripts/estimate_roa ] Performing RoA estimation')
    
    params = load_dataset_params(dataset)

    invalid_label = params["invalid_label"]
    attractor_probability_upper_threshold = params["attractor_probability_upper_threshold"]
    n_runs = n_runs if n_runs is not None else params["n_runs"]

    gt_data = load_ground_truth(dataset)
    start_points = gt_data[:, :2]
    expected_labels = gt_data[:, 2]
    
    labels_set, labels_array = get_labels_array(expected_labels, invalid_label)

    exp_path = path.join("experiments", dataset, path_prefix, exp_name)
    all_predicted_labels, all_final_states = load_final_states_data(path.join(exp_path, "final_states", timestamp), n_runs, invalid_label)

    label_probabilities = generate_label_probabilities(all_predicted_labels[:1], labels_set, invalid_label)

    if generate_img and start_points.shape[1] <= 2:
        import matplotlib.pyplot as plt

        plt.scatter(start_points[:, 0], start_points[:, 1], c=label_probabilities[:, 1], s=0.1, cmap='viridis')
        plt.colorbar()
        plt.savefig(path.join(exp_path, "success_probability_img.png"))

        plt.clf()
        plt.scatter(start_points[:, 0], start_points[:, 1], c=label_probabilities[:, 0], s=0.1, cmap='viridis')
        plt.colorbar()
        plt.savefig(path.join(exp_path, "failure_probability_img.png"))

        plt.clf()
        plt.scatter(start_points[:, 0], start_points[:, 1], c=label_probabilities[:, 2], s=0.1, cmap='viridis')
        plt.colorbar()
        plt.savefig(path.join(exp_path, "invalid_probability_img.png"))

        predicted_labels_img = np.zeros(len(start_points), dtype=np.int32)
        predicted_labels_img[label_probabilities[:, 0] > attractor_probability_upper_threshold] = -1
        predicted_labels_img[label_probabilities[:, 1] > attractor_probability_upper_threshold] = 1

        plt.clf()
        plt.scatter(start_points[:, 0], start_points[:, 1], c=predicted_labels_img, s=0.1, cmap='RdBu')
        plt.colorbar()
        plt.savefig(path.join(exp_path,
                              f"predicted_labels_{str(attractor_probability_upper_threshold).replace('.', 'p')}_img.png"))

    predicted_labels = get_predicted_labels(label_probabilities, attractor_probability_upper_threshold, labels_array, invalid_label)

    # Compute the false positive and false negative rates, true positive and true negative rates, and accuracy and precision and recall and F1 score without confusion matrix

    fp = np.sum((predicted_labels == labels_array[0]) & (expected_labels == labels_array[1]))
    fn = np.sum((predicted_labels == labels_array[1]) & (expected_labels == labels_array[0]))
    tp = np.sum((predicted_labels == labels_array[1]) & (expected_labels == labels_array[1]))
    tn = np.sum((predicted_labels == labels_array[0]) & (expected_labels == labels_array[0]))

    fpr = fp / (fp + tn)
    tpr = tp / (tp + fn)
    fnr = fn / (fn + tp)
    tnr = tn / (tn + fp)

    accuracy = (tp + tn) / len(expected_labels)
    precision = tp / (tp + fp)
    recall = tp / (tp + fn)
    f1_score = 2 * precision * recall / (precision + recall)


    print('Attractor Threshold:', attractor_probability_upper_threshold)

    print("True Positive Rate:", tpr)
    print("False Positive Rate:", fpr)
    print("True Negative Rate:", tnr)
    print("False Negative Rate:", fnr)

    print("False Positives:", fp)
    print("False Negatives:", fn)
    print("True Positives:", tp)
    print("True Negatives:", tn)

    print("Accuracy:", accuracy)
    print("Precision:", precision)
    print("Recall:", recall)
    print("F1 score:", f1_score)

def generate_and_analyze_runs(dataset, exp_name, model_state_name, generate_img, path_prefix="diffusion", n_runs=None):
    params = load_dataset_params(dataset)

    attractors = params["attractors"]
    n_runs = n_runs if n_runs is not None else params["n_runs"]

    gt_data = load_ground_truth(dataset)
    start_points = gt_data[:, :2]

    print("Number of start points:", len(start_points))

    timestamp = time.strftime("%Y-%m-%d_%H-%M-%S")

    exp_path = path.join("experiments", dataset, path_prefix, exp_name)
    generated_trajectories_path = path.join(exp_path, "generated_trajectories", timestamp)
    final_state_path = path.join(generated_trajectories_path, "final_states")

    os.makedirs(final_state_path, exist_ok=True)

    model, model_args = mg_diffuse.utils.model.load_model(exp_path, model_state_name)

    trajectories, _ = perform_runs(params, model, model_args, start_points, attractors, final_state_path, n_runs)
    
    print(f"[ scripts/estimate_roa ] Attractor labels saved in {final_state_path}")

    np.save(path.join(generated_trajectories_path, "generated_trajectories.npy"), trajectories)
    
    perform_roa_estimation(dataset, exp_name, timestamp, generate_img, n_runs=n_runs, path_prefix=path_prefix)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Visualize model trajectories")

    parser.add_argument(
        "--path_prefix",
        type=str,
        default="diffusion",
        help="Path prefix for experiments",
    )
    parser.add_argument(
        "--dataset", type=str, required=True, help="Dataset name"
    )

    parser.add_argument("--exp_name", type=str, required=True, help="Experiment name")

    parser.add_argument(
        "--model_state_name", type=str, required=True, help="Model state file name"
    )

    parser.add_argument(
        "--no_img",
        action="store_true",
        help="Do not generate an image with the attractor labels. Only possible for 2D datasets",
    )

    parser.add_argument(
        "--type",
        type=str, required=True,
        help="Generate or analyze data",
    )

    parser.add_argument(
        "--timestamp",
        type=str,
        help="Timestamp for the experiment to load the attractor labels",
    )

    parser.add_argument(
        "--n_runs",
        type=int,
        help="Number of runs to generate or load",
    )

    args = parser.parse_args()

    if args.type == "generate":
        generate_and_analyze_runs(
            args.dataset, args.exp_name, args.model_state_name, not args.no_img, path_prefix=args.path_prefix, n_runs=args.n_runs
        )
    else:
        perform_roa_estimation(
            args.dataset, args.exp_name, args.timestamp, not args.no_img, path_prefix=args.path_prefix, n_runs=args.n_runs
        )