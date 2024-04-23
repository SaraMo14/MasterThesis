# Applying Policy Graphs to NuScenes

## Description
This project applies Policy Graphs to nuScenes data to explain the behavior of agents in the dataset.

## Folder Structure
- `example/`: This folder contains the source code of the project.
    - `dataset/`: Contains the script `generate_dataset_from_ego.py` for generating trajectory datasets
            from NuScenes raw data.
        - `data/`: Contains the generated Policy Graphs and the NuScenes dataset.
- `pgeon/`: This folder contains a tailored version of the **_pgeon_** Python package, which provides explanations for opaque agents using Policy Graphs.
- `generate_pg.py`: Script for generating a Policy Graph from an agent.
- `test_pg.py`: Script for testing the performance of the Policy Graph.
- `data_analysis.ipynb`: Jupyter notebook for data analysis of the nuScenes dataset.

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


## Generating a Policy Graph
Run `extract_pg.py` if you want to generate a Policy Graph of a trained agent. The flag  `--normalize` specifies whether the probabilities in the PG are stored normalized or not. The flag `--input` specifies the input data location.  The flag `--output` indicates the output formate of the PG.
 ```bash
    python3 extract_pg.py [--normalize] [--verbose] --input DATAFOLDER --output {csv, pickle, gram}
 ```
The resulting Policy Graph will be stored in `example/data/policy_graphs`.

## Evaluating a policy

## Contributing

Contributions are welcome! If you have suggestions for improvements or bug fixes, please open an issue or a pull request.

<!-- 
## License

[Specify the license under which this project is available]
-->
