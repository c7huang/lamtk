import os
os.environ['CUDA_VISIBLE_DEVICES'] = ''
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import argparse
import pickle
import multiprocessing as mp
import numpy as np
from pathlib import Path
from functools import partial
from tqdm.autonotebook import tqdm
from lamtk.aggregation import WaymoAggregator, SceneAggregator

def worker(dataset_path, split, indices):
    rank = mp.current_process()._identity[0]
    agg = WaymoAggregator(
        dataset_path=dataset_path,
        split=split,
        # remove_ground=True,
        data_pipeline = ['load_frame_info', 'load_points',
                         'extract_obj_points'],
        data_fields = ['frame_id', 'ego_pose', 'timestamp',
                       'bg_pts_xyz', 'bg_pts_feats'],
        obj_aggregator = None,
        scene_aggregator = SceneAggregator(ds_voxel_size=0.0325,
                                           ds_average=True,
                                           remove_outliers=False,
                                           estimate_normals=False),
        split_scene=False,
        rank=rank
    )
    results = dict(scene_infos=dict(), obj_infos=dict(), frame_infos=dict())
    for idx in tqdm(indices, desc=f'[{rank}] Aggregating Waymo dataset', position=2*rank):
        scene_id = agg.scene_ids[idx]
        results_i = agg[scene_id]
        scene_info = results_i['scene_info']
        frame_infos = results_i['frame_infos']

        scene_dir = out_dir / 'scenes' / scene_id
        os.makedirs(scene_dir, exist_ok=True)
        scene_info['path'] = Path('scenes') / scene_id
        scene_pts_data = scene_info.pop('pts_data')
        for name, data in scene_pts_data.items():
            data.astype(np.float32).tofile(scene_dir / f'{name}.bin')
        results['scene_infos'][scene_id] = scene_info
        results['frame_infos'].update(frame_infos)
    return results

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Create aggregated waymo dataset')
    parser.add_argument(
        '--dataset-path',
        type=str,
        default='./data/waymo',
        required=False,
        help='specify the root path of dataset')
    parser.add_argument(
        '--split',
        type=str,
        default=None,
        required=False,
        help='specify the split of dataset')
    parser.add_argument(
        '--out-dir',
        type=str,
        default='./data/lamtk/sa_no_gnd/waymo',
        required=False,
        help='the output directory')
    parser.add_argument(
        '--workers',
        type=int,
        default=8,
        required=False,
        help='the number of workers')
    args = parser.parse_args()

    results = dict(scene_infos=dict(), obj_infos=dict(), frame_infos=dict())
    out_dir = Path(args.out_dir)
    os.makedirs(out_dir, exist_ok=True)

    ctx = mp.get_context('fork')
    with ctx.Pool(args.workers) as p:
        imap = p.imap(partial(worker, args.dataset_path, 'training'), np.array_split(np.arange(798), args.workers))
        for results_i in imap:
            results['scene_infos'].update(results_i['scene_infos'])
            results['frame_infos'].update(results_i['frame_infos'])
        imap = p.imap(partial(worker, args.dataset_path, 'validation'), np.array_split(np.arange(202), args.workers))
        for results_i in imap:
            results['scene_infos'].update(results_i['scene_infos'])
            results['frame_infos'].update(results_i['frame_infos'])

    metadata_path = out_dir / f'metadata.pkl'
    with open(metadata_path, 'wb') as f:
        pickle.dump(results, f)
