# Applying Policy Graphs to NuScenes

Description

## Installation

1. **Clone This Repository**

    ```bash
    git clone [Repository URL]
    cd [Repository Name]
    ```
    
2. **Install Dependencies**

    ```bash
    pip install -r PATH_TO_REQUIREMENTS
    ```

3. **Prepare the Dataset**

    Download and uncompress the nuScenes dataset. For the mini version:

    ```bash
    cd [Repository Name]/example/nuscenes/dataset
    mkdir -p /data/sets/nuscenes
    wget https://www.nuscenes.org/data/v1.0-mini.tgz
    tar -xf v1.0-mini.tgz -C ./data/sets/nuscenes
    ```

    Adjust the paths according to your dataset version and storage preferences.

## NuScenes Dataset Processing

1. **Configure the Script**

    This Python script processes data from the nuScenes dataset to compute local displacements, velocity, acceleration, and heading change rate for objects in the dataset. It outputs the processed data to a CSV file.

    Edit the script or use command-line arguments to specify:
    - the dataset directory,
    - the output directory for the processed CSV file,
    - the dataset version (`v1.0-mini`, `v1.0-trainval`, etc.),
    - whether to process key frames only
    - which sensor modality to filter for (`all`, `lidar`, `camera`, `radar`).

2. **Run the Script**

    To process the mini dataset:

    ```bash
    python3 generate_dataset_from_ego.py --dataroot 'data/sets/nuscenes' --version 'v1.0-mini' --dataoutput '[Output Directory]' --key_frames 'True' --sensor lidar
    ```

    To process the full dataset:

    ```bash
    python3 generate_dataset_from_ego.py --dataroot '[Your Dataset Path]' --version 'v1.0-trainval' --dataoutput '[Output Directory]' --key_frames 'True' --sensor lidar
    ```

    Replace `[Output Directory]` with your desired output path for the processed CSV file.


## Demo of Generating a Policy Graph
 ```bash
    python3 demo.ipynb
 ```

## Contributing

Contributions are welcome! If you have suggestions for improvements or bug fixes, please open an issue or a pull request.

<!-- 
## License

[Specify the license under which this project is available]
-->