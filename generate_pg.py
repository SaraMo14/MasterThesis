import argparse
import pgeon.policy_graph as PG
from pathlib import Path
from example.environment import SelfDrivingEnvironment
from example.discretizer.discretizer import AVDiscretizer
#from example.discretizer.discretizer_d1 import AVDiscretizerD1
from example.discretizer.discretizer_d2 import AVDiscretizerD2
from example.discretizer.discretizer_d3 import AVDiscretizerD3
import pandas as pd

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', help='Input data folder of states and actions.', default=".")
    parser.add_argument('--file', help='Input data file name of states and actions.')
    parser.add_argument('--city', help='Specify city to consider when building the PG.', choices=['all', 'b','s1','s2', 's3'], default="all")
    parser.add_argument('--discretizer', type=int, help='Specify the discretizer of the input data.', choices=[0, 1, 2,3], default=0)
    parser.add_argument('--output', help='Which format to output the Policy Graph',
                        default='csv', choices=['pickle', 'csv', 'gram'])
    parser.add_argument('--verbose', help='Whether to make the Policy Graph code output log statements or not',
                        action='store_true')
     
    args = parser.parse_args()
    data_folder, data_file, city_id, verbose, output_format, discretizer_id = args.input, args.file, args.city, args.verbose, args.output, args.discretizer

     

    dtype_dict = {
        'modality': 'category',  # for limited set of modalities, 'category' is efficient
        'scene_token': 'str',  
        'steering_angle': 'float64', 
        'location': 'str',
        'timestamp': 'str',  # To enable datetime operations
        'rotation': 'object',  # Quaternion (lists)
        'x': 'float64',
        'y': 'float64',
        'z': 'float64',
        'yaw': 'float64',  
        'velocity': 'float64',
        'acceleration': 'float64',
        'yaw_rate': 'float64'
    }

    if city_id == 'b': 
        city = 'boston-seaport'
    elif city_id == 's1':
        city = 'singapore-hollandvillage'
    elif city_id == 's2':
        city = 'singapore-onenorth'
    elif city_id == 's3':
        city = 'singapore-queenstown'
    #else:
    #    city=None
 
    #TODO: update
    env = SelfDrivingEnvironment(city)
    discretizer = {0: AVDiscretizer, 1: None, 2: AVDiscretizerD2, 3: AVDiscretizerD3}[discretizer_id](env)
   

    df = pd.read_csv(Path(data_folder) / data_file, dtype=dtype_dict, parse_dates=['timestamp'])
    
    if city_id != 'all':
        df = df[df['location'] == city]
    
    
    pg = PG.PolicyGraph(env, discretizer)
    pg = pg.fit(df, update=False, verbose=verbose)
        
    split = 'mini' if 'mini' in data_file else 'trainval'

    if output_format == 'csv':
        nodes_path = f'example/dataset/data/policy_graphs/pg_{split}_C{city_id}_D{discretizer_id}_nodes.{output_format}'
        edges_path = f'example/dataset/data/policy_graphs/pg_{split}_C{city_id}_D{discretizer_id}_edges.{output_format}'
        traj_path = f'example/dataset/data/policy_graphs/pg_{split}_C{city_id}_D{discretizer_id}_traj.{output_format}'
        pg.save(output_format, [nodes_path, edges_path, traj_path])
    else:
        pg.save(output_format, f'example/dataset/data/policy_graphs/pg_{split}_C{city_id}_D{discretizer_id}.{output_format}')


    if verbose:
        print(f'Successfully generated Policy Graph with {len(pg.nodes)} nodes and {len(pg.edges)} edges.')

    #python3 generate_pg.py --input /home/saramontese/Desktop/MasterThesis/example/dataset/data/sets/nuscenes --verbose --file train_v1.0-trainval_lidar_0.csv --output csv --discretizer 1 --city 'b'
