import numpy as np 
import os
import shutil
import argparse
import math
import json
import random

def unity2dnerf_coordinates(extrinsic_matrix):
    new_extrinsic_matrix = np.copy(extrinsic_matrix)
    new_extrinsic_matrix[0, :3] = extrinsic_matrix[2,:3]
    new_extrinsic_matrix[1, :3] = extrinsic_matrix[1,:3]
    new_extrinsic_matrix[2, :3] = extrinsic_matrix[0,:3]
    
    new_extrinsic_matrix[0, 3] = extrinsic_matrix[2, 3]
    new_extrinsic_matrix[1, 3] = extrinsic_matrix[1, 3]
    new_extrinsic_matrix[2, 3] = extrinsic_matrix[0, 3]
    return new_extrinsic_matrix

def load_data(filename):
    with open(filename, 'r') as file:
        lines = file.readlines()

    matrix1_lines = [line.strip() for line in lines[0:4]]
    matrix2_lines = [line.strip() for line in lines[5:9]]
    array_lines = [line.strip() for line in lines[12:]]

    matrix1 = unity2dnerf_coordinates(np.array([list(map(lambda x: float(x.replace(',', '.')), line.split())) for line in matrix1_lines]))
    matrix2 = np.array([list(map(lambda x: float(x.replace(',', '.')), line.split())) for line in matrix2_lines])
    
    intrinsic = np.array([matrix2[1][2] *2 , matrix2[0][2] * 2, matrix2[0][0]]) #(H,W,F)

    array_data = np.array([float(line.replace(',', '.')) for line in array_lines])

    return matrix1, intrinsic, array_data

def get_data(input_path):
    extrinsic_matrices = []
    intrinsics_matrices = []
    array_data = []

    for file in os.listdir(input_path):
        filename = os.fsdecode(file)
        if filename.endswith(".txt"):
            print(f'Processing file: {filename}')
            matrix1, intrinsic, array = load_data(os.path.join(input_path, filename))
            extrinsic_matrices.append(matrix1)
            intrinsics_matrices.append(intrinsic)
            array_data.append(array)

    return extrinsic_matrices, intrinsics_matrices, array_data


def rewrite_data(input_path, output_path):
    extrinsic_matrices, intrinsics_matrices, array_data = get_data(input_path)

    test_dir = os.path.join(output_path, "test")
    val_dir = os.path.join(output_path, "val")
    train_dir = os.path.join(output_path, "train")

    if not os.path.exists(test_dir):
        os.makedirs(test_dir)
    if not os.path.exists(val_dir):
        os.makedirs(val_dir)
    if not os.path.exists(train_dir):
        os.makedirs(train_dir)

    camera_angles_train = []
    camera_angles_test = []
    camera_angles_val = []
    train_frames = []
    test_frames = []
    val_frames = []

    N_images = len(extrinsic_matrices)

    test_indices = random.sample(range(N_images), 20)
    val_indices = random.sample(set(range(N_images)) - set(test_indices), 10)
    
    train_iterator = 1
    test_iterator = 1
    val_iterator = 1

    print(f'extrinsic_matrices[5] :\n {extrinsic_matrices[5]}')
    print(f'intrinsic_matrix : \n {intrinsics_matrices[0]}')

    for i in range(N_images):
        image_old_name = os.path.join(input_path, f"camera{i}.png")

        w, fl_x = intrinsics_matrices[i][0], intrinsics_matrices[i][2]
        h, fl_y = intrinsics_matrices[i][1], intrinsics_matrices[i][2]
        cx, cy = w / 2, h / 2

        camera_angle_x = math.atan(w / (fl_x * 2)) * 2
        # rotation = math.atan(h / (fl_y * 2)) * 2
        # Values for rotation and camera_angle are hardcoded as in the dnerf dataset

        transform_matrix = extrinsic_matrices[i]

        if i in test_indices:
            new_path = os.path.join(test_dir, f"image_{test_iterator:04d}.png") 
            new_name = f"./test/image_{test_iterator:04d}"
            shutil.copy(image_old_name, new_path)
            camera_angles_test.append(camera_angle_x)
            rotation = 0.3141592653589793
            test_iterator += 1
        elif i in val_indices:
            new_path = os.path.join(val_dir, f"image_{val_iterator:04d}.png")
            new_name = f"./val/image_{val_iterator:04d}"
            shutil.copy(image_old_name, new_path)
            camera_angles_val.append(camera_angle_x)
            rotation = 0.5711986642890533
            val_iterator += 1
        else:
            new_path = os.path.join(train_dir, f"image_{train_iterator:04d}.png")
            new_name = f"./train/image_{train_iterator:04d}"
            shutil.copy(image_old_name, new_path)
            camera_angles_train.append(camera_angle_x)
            rotation = 0.12566370614359174
            train_iterator += 1

        frame_data = {
            "file_path": new_name,
            "rotation": rotation,
            "transform_matrix": transform_matrix,
            "cx": cx,
            "cy": cy,
            "fl_x": fl_x,
            "fl_y": fl_y,
            "h": h,
            "w": w
        }

        if i in test_indices:
            test_frames.append(frame_data)
        elif i in val_indices:
            val_frames.append(frame_data)
        else:
            train_frames.append(frame_data)

    return train_frames, test_frames, val_frames, camera_angles_train, camera_angles_test, camera_angles_val

def generate_json(input_path, output_path):
    train_frames, test_frames, val_frames, camera_angles_train, camera_angles_test, camera_angles_val = rewrite_data(input_path, output_path)

    train_frames = [{**frame_data, "transform_matrix": frame_data["transform_matrix"].tolist()} for frame_data in train_frames]
    test_frames = [{**frame_data, "transform_matrix": frame_data["transform_matrix"].tolist()} for frame_data in test_frames]
    val_frames = [{**frame_data, "transform_matrix": frame_data["transform_matrix"].tolist()} for frame_data in val_frames]


    # train_data = {
    #     "camera_angle_x": float(camera_angles_train[0]), 
    #     "frames": train_frames
    # }
    # test_data = {
    #     "camera_angle_x": float(camera_angles_test[0]), 
    #     "frames": test_frames
    # }
    # val_data = {
    #     "camera_angle_x": float(camera_angles_val[0]), 
    #     "frames": val_frames
    # }

    train_data = {
        "camera_angle_x": 0.6911112070083618, 
        "frames": train_frames
    }
    test_data = {
        "camera_angle_x": 0.6911112070083618, 
        "frames": test_frames
    }
    val_data = {
        "camera_angle_x": 0.6911112070083618, 
        "frames": val_frames
    }

    # with open(os.path.join(output_path, "transforms_train.json"), 'w') as f:
    with open(os.path.join(output_path, "transforms_train.json"), 'w') as f:
        json.dump(train_data, f, indent=4)
    with open(os.path.join(output_path, "transforms_test.json"), 'w') as f:
        json.dump(test_data, f, indent=4)
    with open(os.path.join(output_path, "transforms_val.json"), 'w') as f:
        json.dump(val_data, f, indent=4)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Generate json files from txt files')
    parser.add_argument('--path', type=str, help='Path to the directory containing the txt and png files')
    parser.add_argument('--output', type=str, help='Path to the directory where the json files will be saved')
    args = parser.parse_args()

    generate_json(args.path, args.output)
