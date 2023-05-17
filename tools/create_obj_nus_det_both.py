import os
import argparse
import pickle
import numpy as np
from pathlib import Path
from tqdm.autonotebook import tqdm
from lamtk.aggregation import NuScenesAggregator, ObjectAggregator, NuScenesAggregatorFromDetector, NuScenesAggregatorFromDetectorImages



# python tools/create_obj_nus_det.py --dataroot /home/datasets/nuscenes --version v1.0-medium --out-dir /home/datasets/lamtk/sparse-medium-det --begin 0 --end 250
# python tools/create_obj_nus_det.py --dataroot /home/datasets/nuscenes --version v1.0-mini --out-dir /home/datasets/lamtk/sparse-mini-det --begin 0 --end 10
# python tools/create_obj_nus_det.py --dataroot /scratch/shared-data/nuscenes --version v1.0-trainval --out-dir /home/datasets/lamtk/sparse-trainval-det --begin 0 --end 850
# python tools/create_obj_nus_det.py --dataroot /scratch/shared-data/nuscenes --version v1.0-trainval --out-dir /home/datasets/lamtk/sparse-trainval-images-det --begin 0 --end 850
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Create object nuScenes dataset')
    parser.add_argument(
        '--dataroot',
        type=str,
        default='./data/nuscenes',
        required=False,
        help='specify the root path of dataset')
    parser.add_argument(
        '--version',
        type=str,
        default='v1.0-trainval',
        required=False,
        help='specify the split of dataset')
    parser.add_argument(
        '--out-dir',
        type=str,
        default='./data/lamtk/obj/nuscenes',
        required=False,
        help='the output directory')
    parser.add_argument(
        '--begin',
        type=int,
        default=0,
        required=False,
        help='')
    parser.add_argument(
        '--end',
        type=int,
        default=850,
        required=False,
        help='')
    parser.add_argument(
        '--rank',
        type=int,
        default=0,
        required=False,
        help='the rank of the process')
    args = parser.parse_args()

    agg = NuScenesAggregatorFromDetectorImages(
        merged_global_dets_path='/home/datasets/lamtk/merged_global.pt',
        merged_local_dets_path='/home/datasets/lamtk/merged_local.pt',
        confidence_threshold=0.01,
        tp_threshold=0.01,
        bbox_scale=1.0,
        version=args.version,
        dataroot=args.dataroot,
        data_pipeline = ['load_frame_info', 
                         'load_points', 
                         'load_images', 
                         'load_obj_boxes3d', 
                         'load_obj_labels', 
                         'extract_obj_images',
                         'extract_obj_points',
                         ],
        data_fields = ['frame_id', 
                        'ego_pose', 
                        'timestamp',
                        'obj_pts_xyz', 
                        'obj_pts_feats',
                        'obj_labels', 
                        'obj_boxes3d',
                        'obj_images',
                        'obj_visibility'],
        crop_size=(224,224),
        obj_aggregator = ObjectAggregator(
            align=True, combine=False, mirror=False,
            propagate_labels=False, down_sample=False,
            remove_outliers=False, estimate_normals=False),
        scene_aggregator = None,
        split_scene=False,
        rank=args.rank,
        skip_non_keyframe=True,
    )

    results = dict(scene_infos=dict(), obj_infos=dict(), frame_infos=dict())
    out_dir = Path(args.out_dir)
    os.makedirs(out_dir, exist_ok=True)

    # print(len(agg.scene_ids))
    # print(obj)

    for idx in tqdm(range(args.begin, args.end), desc=f'[{args.rank}] Aggregating nuScenes dataset', position=2*args.rank):
        if idx >= len(agg.scene_ids):
            print('in break')
            break

        scene_id = agg.scene_ids[idx]
        results_i = agg[scene_id]
        obj_infos = results_i['obj_infos']
        scene_info = results_i['scene_info']
        frame_infos = results_i['frame_infos']

        for obj_id, obj_info in obj_infos.items():
            obj_pts_data = obj_info.pop('pts_data')
            obj_images = obj_info.pop('img_crops')
            obj_visibility = obj_info.pop('visibility')



            obj_dir = out_dir / 'objects' / obj_id
            os.makedirs(obj_dir, exist_ok=True)
            obj_info['path'] = Path('objects') / obj_id
            obj_info['sample_token'] = agg.obj_id_to_sample[obj_id]
            obj_info['num_pts'] = {}
            obj_info['visibility'] = {}



            #save pts data
            for name, data in obj_pts_data.items():
                for idx, data_i in enumerate(data):
                    if data_i is None:
                        continue

                    obj_info['num_pts'][idx] = data_i.shape[0]
                    data_dir = obj_dir / str(idx)
                    os.makedirs(data_dir, exist_ok=True)
                    data_i.astype(np.float32).tofile(data_dir / f'{name}.bin')


            try:
                assert len(obj_images) == len(obj_visibility)
            except AssertionError:
                print('AssertionError')
                print(obj_images)
                print(obj_visibility)
                raise AssertionError

            #save images
            for i,obj_images in enumerate(obj_images):
                if obj_images is None:
                    continue

                obj_info['visibility'][i] = obj_visibility[i]
                obj_info['crop_size'] = obj_images.shape[-2:]
                data_dir = obj_dir / str(i)
                os.makedirs(data_dir, exist_ok=True)
                obj_images.astype(np.float32).tofile(data_dir / f'img_crop.bin')

            
                    
            results['obj_infos'][obj_id] = obj_info

        # exit(0)
        results['scene_infos'][scene_id] = scene_info
        results['frame_infos'].update(frame_infos)

    metadata_dir = out_dir / 'metadata'
    os.makedirs(metadata_dir, exist_ok=True)
    metadata_suffix = f'trainval_{args.begin:0>3}-{args.end:0>3}'
    metadata_path = metadata_dir / f'metadata_{metadata_suffix}.pkl'
    with open(metadata_path, 'wb') as f:
        pickle.dump(results, f)
