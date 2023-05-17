import os
import sys
import copy
import pickle
import numpy as np
import PIL
from typing import List, Any
from tqdm.autonotebook import tqdm
from scipy.interpolate import interpn, BPoly
from .dataset_aggregator import DatasetAggregator
from ..utils.transforms import affine_transform, apply_transform
from .utils import get_crops_per_image
from nuscenes.utils.geometry_utils import BoxVisibility


DEFAULT_DATA_PIPELINE = [
    'load_frame_info',
    'load_images',
    'load_points',
    'load_pts_rgb',
    'load_pts_labels',
    'load_obj_boxes3d',
    'load_obj_labels',      # Object labels can depend on interpolated boxes
    'extract_obj_points',
    'load_obj_boxes',       # Object image boxes require isolated object points
    'extract_obj_images'
]


def startswith(s):
    if not isinstance(s, list):
        s = [s]

    def _startswith(k):
        for si in s:
            if k.startswith(si):
                return True
        return False
    return _startswith


def is_not_none(x):
    return x is not None



class NuScenesAggregator(DatasetAggregator):
    SENSORS = [
        'LIDAR_TOP',
        'CAM_FRONT',
        'CAM_FRONT_RIGHT',
        'CAM_BACK_RIGHT',
        'CAM_BACK',
        'CAM_BACK_LEFT',
        'CAM_FRONT_LEFT'
    ]
    CAMERAS = SENSORS[1:]

    def __init__(self, version='v1.0-trainval', dataroot='data/nuscenes', crop_size=(224,224,),
                 verbose=True, data_pipeline=DEFAULT_DATA_PIPELINE, **kwargs):
        from nuscenes import NuScenes
        super().__init__(data_pipeline=data_pipeline, **kwargs)
        self.nusc = NuScenes(version, dataroot, verbose)
        self.scene_ids = [scene['token'] for scene in self.nusc.scene]
        self.crop_size = crop_size

    def gather_frames(self, scene_id: str) -> None:
        frames = []
        scene = self.nusc.get('scene', scene_id)
        s = self.nusc.get('sample', scene['first_sample_token'])
        s_prev = s_next = s
        sd = {sensor: self.nusc.get(
            'sample_data', s['data'][sensor]) for sensor in self.SENSORS}
        lidar_token = s['data']['LIDAR_TOP']
        idx = 0
        while lidar_token != '':
            frame = dict(idx=idx)
            idx += 1

            ####################################################################
            # Load lidar data and appropriate camera data based on timestamp
            ####################################################################
            sd['LIDAR_TOP'] = self.nusc.get('sample_data', lidar_token)
            lidar_ts = sd['LIDAR_TOP']['timestamp'] / 1e6
            for cam in self.CAMERAS:
                for _ in range(2):      # Check the next 2 frames
                    if sd[cam]['next'] == '':
                        continue
                    cam_ts = sd[cam]['timestamp'] / 1e6
                    cam_sd_next = self.nusc.get('sample_data', sd[cam]['next'])
                    cam_ts_next = cam_sd_next['timestamp'] / 1e6
                    time_diff = abs(lidar_ts - cam_ts)
                    time_diff_next = abs(lidar_ts - cam_ts_next)
                    if time_diff_next > time_diff:
                        continue
                    sd[cam] = cam_sd_next
            frame['sd'] = copy.copy(sd)

            ####################################################################
            # Load previous and next key-frames if necessary
            ####################################################################
            if sd['LIDAR_TOP']['is_key_frame']:
                s_prev = s_next
                if s_prev['next'] != '':
                    s_next = self.nusc.get('sample', s_prev['next'])
                else:
                    s_next = None
                frame['s'] = s_prev
            else:
                frame['s_prev'] = s_prev
                frame['s_next'] = s_next

            if len(frames) > 0:
                frame['prev'] = frames[-1]
                frame['prev']['next'] = frame
            frames.append(frame)
            lidar_token = sd['LIDAR_TOP']['next']

        return frames

    def load_frame_info(self, frame: dict) -> None:
        # Since nuscenes point cloud is in LiDAR coordinate,
        # ego pose in this case is the transformation from LiDAR to global
        sd = frame['sd']['LIDAR_TOP']
        sensor2ego = self.nusc.get(
            'calibrated_sensor', sd['calibrated_sensor_token'])
        ego2global = self.nusc.get('ego_pose', sd['ego_pose_token'])
        frame['frame_id'] = sd['token']
        frame['ego_pose'] = affine_transform(
            rotation=np.roll(ego2global['rotation'], -1),
            rotation_format='quat',
            translation=ego2global['translation']
        ) @ affine_transform(
            rotation=np.roll(sensor2ego['rotation'], -1),
            rotation_format='quat',
            translation=sensor2ego['translation']
        )
        frame['timestamp'] = sd['timestamp'] / 1e6

    def load_images(self, frame: dict) -> None:
        frame['images'] = {}
        frame['cam_extrinsics'] = {}
        frame['cam_intrinsics'] = {}
        for cam in self.CAMERAS:
            sd = frame['sd'][cam]
            sensor2ego = self.nusc.get(
                'calibrated_sensor', sd['calibrated_sensor_token'])
            ego2global = self.nusc.get('ego_pose', sd['ego_pose_token'])
            frame['images'][cam] = np.array(PIL.Image.open(
                f'{self.nusc.dataroot}/{sd["filename"]}'))
            # Camera extrinsic is set to cam2global transformation since
            # camera is not well aligned with lidar ego pose
            frame['cam_extrinsics'][cam] = affine_transform(
                rotation=np.roll(ego2global['rotation'], -1),
                rotation_format='quat',
                translation=ego2global['translation']
            ) @ affine_transform(
                rotation=np.roll(sensor2ego['rotation'], -1),
                rotation_format='quat',
                translation=sensor2ego['translation']
            )
            frame['cam_intrinsics'][cam] = sensor2ego['camera_intrinsic']


    def extract_obj_images(self, frame: Any) -> None:
        if not frame['sd']['LIDAR_TOP']['is_key_frame']:
            pass
            

        frame['obj_images'] = dict()
        images = []
        ci_list = []
        l2c_list = []
        for cam in self.CAMERAS:
            images.append(frame['images'][cam] )
            l2c_list.append((np.linalg.inv(frame['cam_extrinsics'][cam])\
                             @ frame['lidar_extrinsic']).astype(np.float32))
            ci_temp = np.eye(4).astype(np.float32)
            ci_temp[:3, :3] = frame['cam_intrinsics'][cam]
            ci_list.append(ci_temp)

        obj_boxes = frame.get('obj_boxes3d', dict())

        if len(obj_boxes) > 0:
            boxes_stacked =  np.stack(list(frame['obj_boxes3d'].values()))[:, :7].astype(np.float32)
            crops, idx = get_crops_per_image(img_list=images,
                                             ci_list=ci_list,
                                             l2c_list=l2c_list,
                                             boxes_3d=torch.from_numpy(boxes_stacked),
                                             device='cpu',
                                             imsize=(1600,900),
                                             crop_size=self.crop_size,
                                             visibility=BoxVisibility.ANY)

            box_keys = list(frame['obj_boxes3d'].keys())

            for i,box_idx in enumerate(idx):
                try:
                    id = box_keys[box_idx]
                    frame['obj_images'][id] = crops[i,...].numpy()
                except IndexError:
                    print()
                    print("Inbdex error for i:",i)
                    print(idx)
                    print('idx',idx.shape)
                    print('crops',crops.shape)
                    print('boxes_stacked',boxes_stacked.shape,)
                    exit(0)


            # for images without crops
            box_idx = torch.ones(boxes_stacked.shape[0])
            box_idx[idx] = 0
            
            for box_i in torch.where(box_idx == 1)[0]:
                id = box_keys[box_i]
                frame['obj_images'][id] = None


        else:
            pass


    def load_points(self, frame: dict) -> None:
        if 'pts_xyz' in frame:
            return
        sd = frame['sd']['LIDAR_TOP']
        points = np.fromfile(
            f'{self.nusc.dataroot}/{sd["filename"]}', dtype=np.float32
        ).reshape(-1, 5)
        # points = points[np.linalg.norm(points[:,:3], axis=1) > 2]
        frame['pts_xyz'] = points[:, :3]
        frame['pts_range'] = np.linalg.norm(
            frame['pts_xyz'], axis=-1, keepdims=True)
        frame['pts_dir'] = frame['pts_xyz'] / frame['pts_range']
        # Add frame idx and timestamp as additional feature channel
        frame['pts_feats'] = np.concatenate([
            points[:, 3:5], np.tile(frame['idx'], (points.shape[0], 1)),
        ], axis=-1).astype(np.float32)
        frame['lidar_extrinsic'] = frame['ego_pose']

    def load_pts_rgb(self, frame: dict) -> None:
        if frame.get('images', None) is None:
            frame['pts_rgb'] = np.zeros_like(frame['pts_xyz'])
            return

        pts_xyz = frame['pts_xyz']
        pts_rgb = np.zeros((pts_xyz.shape[0], 3), dtype=np.float32)
        valid_mask = np.zeros(pts_xyz.shape[0], dtype=bool)
        num_valid = np.zeros(pts_xyz.shape[0], dtype=np.uint8)

        if isinstance(frame['images'], dict):
            images = frame['images'].values()
            cam_intrinsics = frame['cam_intrinsics'].values()
            cam_extrinsics = frame['cam_extrinsics'].values()
        elif not isinstance(frame['images'], list):
            images = [frame['images']]
            cam_intrinsics = [frame['cam_intrinsics']]
            cam_extrinsics = [frame['cam_extrinsics']]

        for img, K, E in zip(images, cam_intrinsics, cam_extrinsics):
            img = np.swapaxes(img, 0, 1)

            # 1. Transform points to camera frame
            # lidar frame -> global frame -> camera frame -> image frame
            lidar2img = affine_transform(
                K) @ np.linalg.inv(E) @ frame['lidar_extrinsic']
            points_img = apply_transform(lidar2img, pts_xyz)
            points_img[:, :2] = points_img[:, :2] / points_img[:, 2:3]

            # 2. Filter out-of-bound points
            valid_mask_i = points_img[:, 2] > 0
            valid_mask_i &= points_img[:, 0] >= 0
            valid_mask_i &= points_img[:, 0] <= img.shape[0] - 1
            valid_mask_i &= points_img[:, 1] >= 0
            valid_mask_i &= points_img[:, 1] <= img.shape[1] - 1

            # 3. Get rgb features from image
            points_rgb_i = np.zeros((pts_xyz.shape[0], 3))
            points_rgb_i[valid_mask_i] = interpn(
                (np.arange(img.shape[0]), np.arange(img.shape[1])), img,
                (points_img[valid_mask_i][:, 0],
                 points_img[valid_mask_i][:, 1]),
                method='nearest')

            # 4. Record rgb features and number of observations
            pts_rgb += points_rgb_i
            valid_mask |= valid_mask_i
            num_valid[valid_mask_i] += 1

        pts_rgb[valid_mask] = pts_rgb[valid_mask] / \
            num_valid[valid_mask][:, None]
        frame['pts_rgb'] = pts_rgb

    def load_pts_labels(self, frame: dict) -> None:
        if 'pts_labels' in frame:
            return
        if frame['sd']['LIDAR_TOP']['is_key_frame']:
            lidarseg = self.nusc.get(
                'lidarseg', frame['sd']['LIDAR_TOP']['token'])
            frame['pts_labels'] = np.fromfile(
                f'{self.nusc.dataroot}/{lidarseg["filename"]}',
                dtype=np.uint8).astype(np.int32)[:, None]
        else:
            frame['pts_labels'] = np.full(
                (frame['pts_xyz'].shape[0], 1), -1, dtype=np.int32)

    def load_obj_labels(self, frame: dict) -> None:
        frame['obj_labels'] = dict()
        for instance_token in frame.get('obj_boxes3d', dict()).keys():
            instance = self.nusc.get('instance', instance_token)
            category_token = instance['category_token']
            category = self.nusc.get('category', category_token)
            frame['obj_labels'][instance_token] = category['name']

    def load_obj_boxes(self, frame: dict) -> None:
        super().load_obj_boxes(frame)

    def load_obj_boxes3d(self, frame: dict) -> None:
        if frame['sd']['LIDAR_TOP']['is_key_frame']:
            frame['obj_boxes3d'] = self.load_obj_boxes3d_from_sample(
                frame['s'])
        else:
            frame['obj_boxes3d'] = self.interp_obj_boxes3d(frame)

    def convert_box3d_format(self, boxes: List) -> np.ndarray:
        locs = np.array([b.center for b in boxes]).reshape(-1, 3)
        dims = np.array([b.wlh for b in boxes]).reshape(-1, 3)
        rots = np.array([b.orientation.yaw_pitch_roll[0]
                         for b in boxes]).reshape(-1, 1)
        return np.concatenate([locs, dims, -rots - np.pi / 2], axis=1)

    def load_obj_boxes3d_from_sample(
        self, sample: dict,
        load_global_coords: bool = False,
        load_velocity: bool = False
    ) -> None:
        box3d_list = self.nusc.get_sample_data(sample['data']['LIDAR_TOP'])[1]
        box3d_list = self.convert_box3d_format(box3d_list)
        boxes3d = dict()
        if load_velocity:
            velos = dict()

        for ann_token, box in zip(sample['anns'], box3d_list):
            ann = self.nusc.get('sample_annotation', ann_token)
            instance_token = ann['instance_token']
            if load_global_coords:
                box[:3] = ann['translation']
            boxes3d[instance_token] = box
            if load_velocity:
                velos[instance_token] = self.nusc.box_velocity(ann_token)

        if load_velocity:
            return boxes3d, velos
        else:
            return boxes3d

    def interp_obj_boxes3d(self, frame: dict) -> None:
        ts = frame['sd']['LIDAR_TOP']['timestamp'] / 1e6

        s_prev = frame['s_prev']
        sd_prev = self.nusc.get('sample_data', s_prev['data']['LIDAR_TOP'])
        ts_prev = sd_prev['timestamp'] / 1e6
        boxes3d_prev, velos_prev = self.load_obj_boxes3d_from_sample(
            s_prev, load_global_coords=True, load_velocity=True)

        s_next = frame['s_next']
        sd_next = self.nusc.get('sample_data', s_next['data']['LIDAR_TOP'])
        ts_next = sd_next['timestamp'] / 1e6
        boxes3d_next, velos_next = self.load_obj_boxes3d_from_sample(
            s_next, load_global_coords=True, load_velocity=True)

        boxes3d = dict()
        for instance_token in boxes3d_prev:
            if instance_token not in boxes3d_next:
                continue

            box3d_prev = boxes3d_prev[instance_token]
            box3d_next = boxes3d_next[instance_token]
            heading_diff = np.abs(box3d_prev[-1]-box3d_next[-1])
            if box3d_prev[-1] < box3d_next[-1] and \
                    box3d_prev[-1]+2*np.pi - box3d_next[-1] < heading_diff:
                box3d_prev[-1] += 2*np.pi
            elif box3d_prev[-1] > box3d_next[-1] and \
                    box3d_next[-1]+2*np.pi - box3d_prev[-1] < heading_diff:
                box3d_next[-1] += 2*np.pi

            # Get previous location and velocity
            velo_prev = np.zeros_like(box3d_prev)
            velo_prev[:3] = velos_prev[instance_token]
            if np.isnan(velos_prev[instance_token]).any():
                grads_prev = [box3d_prev]
            else:
                grads_prev = [box3d_prev, velo_prev]

            # Get next location and velocity
            velo_next = np.zeros_like(box3d_next)
            velo_next[:3] = velos_next[instance_token]
            if np.isnan(velos_next[instance_token]).any():
                grads_next = [box3d_next]
            else:
                grads_next = [box3d_next, velo_next]

            # Compute a cubic Bezier curve interpolator
            # This has the same effect as Hermite interpolation
            # but uses Berstein basis instaed of Hermite functions.
            box3d = BPoly.from_derivatives(
                [ts_prev, ts_next], [grads_prev, grads_next])(ts)
            box3d[:3] = apply_transform(np.linalg.inv(
                frame['lidar_extrinsic']), box3d[None, :3])[0]
            boxes3d[instance_token] = box3d

        return boxes3d

    def clean_up(self, frame: dict) -> None:
        current_frame = frame
        # Only clean up key frames and the preceeding unlabeled frames
        if not current_frame['sd']['LIDAR_TOP']['is_key_frame']:
            return
        is_key_frame = False
        while not is_key_frame and current_frame is not None:
            prev_frame = current_frame.get('prev', None)
            super().clean_up(current_frame)
            current_frame = prev_frame
            if current_frame is not None and 'sd' in current_frame:
                is_key_frame = current_frame['sd']['LIDAR_TOP']['is_key_frame']
            else:
                is_key_frame = True




class Center2DRange(object):
    """Within 2D range"""

    def __init__(self, distance=2, coordinate='lidar'):
        # assert coordinate in ['camera', 'lidar', 'depth']
        self.distance = distance

    def __call__(self, bboxes1, bboxes2, mode='iou', is_aligned=False):
        """assumes xy are the first two coordinates"""
        iou = torch.cdist(bboxes1[:,:2],bboxes2[:,:2],p=2) 
        # idx1,idx2 = torch.where(iou < self.distance)
        return iou #, idx1, idx2

    def __repr__(self):
        """str: Return a string that describes the module."""
        repr_str = self.__class__.__name__
        repr_str += f'(coordinate={self.coordinate}'
        return repr_str

from mmdet3d.core.bbox.iou_calculators import BboxOverlapsNearest3D
from scipy.optimize import linear_sum_assignment
import torch



class NuScenesAggregatorFromDetector(NuScenesAggregator):

    CLS_TO_IDX = {
                'car':0, 'truck':1, 
                'construction_vehicle':2, 'bus':3, 
                'trailer':4, 'barrier':5, 
                'motorcycle':6, 'bicycle':7, 
                'pedestrian':8, 'traffic_cone':9
            }

    IDX_TO_CLS = {v:k for k,v in CLS_TO_IDX.items()}

    NameMapping = {
        "movable_object.barrier": "barrier",
        "vehicle.bicycle": "bicycle",
        "vehicle.bus.bendy": "bus",
        "vehicle.bus.rigid": "bus",
        "vehicle.car": "car",
        "vehicle.construction": "construction_vehicle",
        "vehicle.motorcycle": "motorcycle",
        "human.pedestrian.adult": "pedestrian",
        "human.pedestrian.child": "pedestrian",
        "human.pedestrian.construction_worker": "pedestrian",
        "human.pedestrian.police_officer": "pedestrian",
        "movable_object.trafficcone": "traffic_cone",
        "vehicle.trailer": "trailer",
        "vehicle.truck": "truck",
    }

    
    def __init__(self, 
                 merged_global_dets_path, 
                 merged_local_dets_path,
                 tp_threshold=0.1, 
                 confidence_threshold=0.01, 
                 bbox_scale=1,
                 skip_non_keyframe=True,
                 *args, 
                 **kwargs):
        super().__init__(*args, **kwargs)
        self.merged_global_dets_ = torch.load(merged_global_dets_path)
        self.merged_local_dets_ = torch.load(merged_local_dets_path)
        self.ioucal = BboxOverlapsNearest3D('lidar')
        # self.ioucal = Center2DRange(distance=2)
        self.confidence_threshold = confidence_threshold
        self.tp_threshold = tp_threshold
        self.bbox_scale = bbox_scale
        self.fp_labels = dict()
        self.fp_keys = dict()
        self.skip_non_keyframe = skip_non_keyframe

        self.obj_id_to_sample = dict()






    def get_obj_infos(self, results: dict) -> None:
        obj_pts_fields = list(filter(startswith('obj_pts_'), results.keys()))
        obj_labels = results.get('obj_labels', dict())
        obj_boxes3d = results.get('obj_boxes3d', dict())
        obj_images = results.get('obj_images', dict())
        obj_visibility = results.get('obj_visibility', dict())

        obj_ids = []
        if len(obj_pts_fields) > 0:
            obj_ids.extend(list(results[obj_pts_fields[0]].keys()))
        obj_ids.extend(list(obj_labels.keys()))
        obj_ids.extend(list(obj_boxes3d.keys()))
        obj_ids = set(obj_ids)
        obj_infos = dict()
        for obj_id in obj_ids:
            class_name = list(
                filter(is_not_none, obj_labels.get(obj_id, ['unknown'])))[0]
            obj_boxes3d_i = np.asarray(list(
                filter(is_not_none, obj_boxes3d.get(obj_id, [np.zeros(7)]))))
            obj_infos[obj_id] = dict(
                id=obj_id,
                class_name=class_name,
                size=obj_boxes3d_i[:, 3:6].mean(0).astype(np.float32),
                pts_data={k[4:]: results[k][obj_id] for k in obj_pts_fields},
                img_crops=obj_images.get(obj_id, [None]),
                visibility=obj_visibility.get(obj_id, [None]),)

            # TODO: save object images
            # img_data=dict(images=results.get('obj_images', None),
            #               img_labels=results.get('obj_img_labels', None)))

        return obj_infos



    



    def get_predictions(self,sample_token,load_global_coords,):
        if load_global_coords:
            dets = self.merged_global_dets_[sample_token]
            det_bbox = torch.from_numpy(dets['boxes_3d'])
            det_label = torch.from_numpy(dets['labels_3d'])
            det_score = torch.from_numpy(dets['scores_3d'])
        else:
            dets = self.merged_local_dets_[sample_token]
            det_bbox = dets['boxes_3d']
            det_label = dets['labels_3d']
            det_score = dets['scores_3d']

        #filter based on confidence
        filter_ = torch.where(det_score > self.confidence_threshold)[0]
        det_bbox = det_bbox[filter_]
        det_label = det_label[filter_]
        det_score = det_score[filter_]


        # print(sample_token,"-- bbox shapeL",det_bbox.shape)

        return det_bbox, det_label, det_score

    def load_obj_boxes3d(self, frame: dict) -> None:
        if frame['sd']['LIDAR_TOP']['is_key_frame']:
            frame['obj_boxes3d'], frame['obj_visibility'] = self.load_obj_boxes3d_from_sample(
                frame['s'])
        else:
            frame['obj_boxes3d'], frame['obj_visibility']  = self.interp_obj_boxes3d(frame)

    def load_obj_boxes3d_from_sample(
        self, sample: dict,
        load_global_coords: bool = False,
        load_velocity: bool = False,
        load_visibility: bool = True,
    ) -> None:
        box3d_list = self.nusc.get_sample_data(sample['data']['LIDAR_TOP'])[1]
        box3d_list = self.convert_box3d_format(box3d_list)
        boxes3d = dict()
        labels = dict()
        visibility = dict()
        if load_velocity:
            velos = dict()

        for ann_token, box in zip(sample['anns'], box3d_list):
            ann = self.nusc.get('sample_annotation', ann_token)
            instance_token = ann['instance_token']
            if load_global_coords:
                box[:3] = ann['translation']
            boxes3d[instance_token] = box
            if load_velocity:
                velos[instance_token] = self.nusc.box_velocity(ann_token)
            
            labels[instance_token] = NuScenesAggregatorFromDetector.CLS_TO_IDX.get(
                NuScenesAggregatorFromDetector.NameMapping.get(ann['category_name'],-1),-1
            )

            visibility[instance_token] = ann['visibility_token']


        det_bbox, det_label, det_score = self.get_predictions(sample_token=sample['token'],load_global_coords=load_global_coords,)

        gt_bboxes = np.array(list(boxes3d.values())).astype(np.float32)
        fp_det_idx, iou, tp_det_idx, tp_gt_idx = self.get_iou_idx(bbox=det_bbox[:,:7], 
                                                      gt_bboxes=gt_bboxes,
                                                      cls1=det_label,
                                                      cls2=torch.tensor(list(labels.values())),
                                                      device='cpu')

        keys = list(boxes3d.keys())
        out_boxes3d = dict()
        out_visibility = dict()
        if load_velocity:
            out_velos = dict()

        for i,x in enumerate(tp_gt_idx):
            k = keys[x]
            out_boxes3d[k] = det_bbox[tp_det_idx[i],:7].numpy() #replace gt bbox with det bbox
            # out_boxes3d[k][3:6] *= self.bbox_scale
            out_boxes3d[k][3:5] *= self.bbox_scale

            self.obj_id_to_sample[k] = sample['token']
            out_visibility[k] = visibility[k]

            if load_velocity:
                out_velos[k] = torch.cat([det_bbox[tp_det_idx[i],7:9],torch.zeros(1)],dim=0).numpy()


        for i in fp_det_idx:
            try:
                key_num = self.fp_keys[sample['token']].get(i.item(),"{:07d}".format(len(self.fp_labels)))
            except KeyError:
                key_num = "{:07d}".format(len(self.fp_labels))
                self.fp_keys[sample['token']] = dict()

            self.fp_keys[sample['token']][i.item()] = key_num
            k = "FP_{}".format(key_num)

            sorted_label = self.fp_labels.get(k,-1)
            if sorted_label != -1:
                assert sorted_label == NuScenesAggregatorFromDetector.IDX_TO_CLS[det_label[i].item()], "exist_label:{},label:{},k:{},len(fp_labels):{}".format(sorted_label,
                NuScenesAggregatorFromDetector.IDX_TO_CLS[det_label[i].item()],k,len(self.fp_labels))

            self.fp_labels[k] = NuScenesAggregatorFromDetector.IDX_TO_CLS[det_label[i].item()]
            self.obj_id_to_sample[k] = sample['token'] 

            out_boxes3d[k] = det_bbox[i,:7].numpy()
            out_boxes3d[k][3:5] *= self.bbox_scale

            out_visibility[k] = None

            if load_velocity:
                out_velos[k] = torch.cat([det_bbox[i,7:9],torch.zeros(1)],dim=0).numpy()


        
        if load_velocity:
            return out_boxes3d, out_visibility, out_velos
        else:
            return out_boxes3d, out_visibility


    def load_obj_labels(self, frame: dict) -> None:
        frame['obj_labels'] = dict()
        for instance_token in frame.get('obj_boxes3d', dict()).keys():
            if instance_token.startswith('FP_'):
                # print('in obj labels')
                frame['obj_labels'][instance_token] = self.fp_labels[instance_token]
            else:
                instance = self.nusc.get('instance', instance_token)
                category_token = instance['category_token']
                category = self.nusc.get('category', category_token)
                frame['obj_labels'][instance_token] = category['name']



    def interp_obj_boxes3d(self, frame: dict) -> None:
        ts = frame['sd']['LIDAR_TOP']['timestamp'] / 1e6

        s_next = frame['s_next']
        sd_next = self.nusc.get('sample_data', s_next['data']['LIDAR_TOP'])
        ts_next = sd_next['timestamp'] / 1e6
        ts_prev = ts_next - 0.5
        boxes3d_next, visbility, velos_next = self.load_obj_boxes3d_from_sample(
            s_next, load_global_coords=True, load_velocity=True)


        boxes3d = dict()
        for instance_token in boxes3d_next:
            # print('Interpolating {}...'.format(instance_token))
            box3d_next = boxes3d_next[instance_token]

            # Get next location and velocity
            velo_next = np.zeros_like(box3d_next)
            velo_next[:3] = velos_next[instance_token]


            box3d_prev = box3d_next.copy()
            box3d_prev[:2] = box3d_prev[:2] - velo_next[:2] * 0.5
            box3d_prev[-1] = np.arctan2(*velo_next[:2]) - 0.5 * np.pi # get heading as velocity direction

            heading_diff = np.abs(box3d_prev[-1]-box3d_next[-1])
            if box3d_prev[-1] < box3d_next[-1] and \
                    box3d_prev[-1]+2*np.pi - box3d_next[-1] < heading_diff:
                box3d_prev[-1] += 2*np.pi
            elif box3d_prev[-1] > box3d_next[-1] and \
                    box3d_next[-1]+2*np.pi - box3d_prev[-1] < heading_diff:
                box3d_next[-1] += 2*np.pi

            diff = ts_next - ts_prev
            ts_diff = (ts - ts_prev) / diff
            xydiff = box3d_next[:2] - box3d_prev[:2]

            box3d = box3d_prev
            box3d[:2] = box3d_prev[:2] + ts_diff * xydiff
            box3d[-1] = box3d_prev[-1] + ts_diff * (box3d_next[-1] - box3d_prev[-1])

            box3d[:3] = apply_transform(np.linalg.inv(
                frame['lidar_extrinsic']), box3d[None, :3])[0]
            boxes3d[instance_token] = box3d

        return boxes3d, visbility


    def get_iou_idx(self,bbox,gt_bboxes,device,cls1=None,cls2=None,):
        if not isinstance(bbox, torch.Tensor):
            bbox = torch.tensor(bbox)
        if not isinstance(gt_bboxes, torch.Tensor):
            gt_bboxes = torch.tensor(gt_bboxes)

        
        if bbox.size(0) == 0 or gt_bboxes.size(0) == 0:
            return [],[],torch.tensor([],dtype=torch.int64,device=device),torch.tensor([],dtype=torch.int64,device=device)

        if cls1 != None and cls2 != None and len(cls1) > 0 and len(cls2) > 0:
            assert len(cls1) == bbox.size(0)
            assert len(cls2) == gt_bboxes.size(0)
            mask = torch.zeros((bbox.size(0),gt_bboxes.size(0),),dtype=torch.float32,device=device)
            cp = torch.cartesian_prod(torch.arange(0,bbox.size(0)), torch.arange(0,gt_bboxes.size(0)))
            mask[cp[:,0],cp[:,1]] = (cls1.long()[cp[:,0]] != cls2.long()[cp[:,1]]).float() * 10000.0
        else:
            mask = torch.zeros((bbox.size(0),gt_bboxes.size(0),),dtype=torch.float32,device=device)
            

        if type(self.ioucal) == Center2DRange:

            det_iou = self.ioucal(bbox,gt_bboxes)
            det_iou = det_iou + mask

            tp_det_idx, tp_gt_idx = linear_sum_assignment(det_iou.cpu().numpy())
            tp_det_idx = torch.from_numpy(tp_det_idx).to(device)
            tp_gt_idx = torch.from_numpy(tp_gt_idx).to(device)
        
            matches = torch.where(det_iou[tp_det_idx,tp_gt_idx] < self.ioucal.distance)
            tp_det_idx = tp_det_idx[matches]
            tp_gt_idx = tp_gt_idx[matches]

            fp_det_idx = torch.arange(bbox.shape[0])[~tp_det_idx]
            keep = torch.where(det_iou.min(dim=1).values[fp_det_idx] >= 3.0)
            fp_det_idx = fp_det_idx[keep]

        else:

            det_iou_ = self.ioucal(bbox,gt_bboxes) * -1 #bbox.tensor,gt_bboxes.tensor) * -1 
            det_iou = det_iou_ + mask #dont match objects from different classes
            
            tp_det_idx, tp_gt_idx = linear_sum_assignment(det_iou.cpu().numpy())
            tp_det_idx = torch.from_numpy(tp_det_idx).to(device)
            tp_gt_idx = torch.from_numpy(tp_gt_idx).to(device)

            matches = torch.where(det_iou[tp_det_idx,tp_gt_idx] < ( self.tp_threshold * -1))
            tp_det_idx = tp_det_idx[matches]
            tp_gt_idx = tp_gt_idx[matches]
            
            fp_det_idx = torch.arange(bbox.shape[0])[~tp_det_idx]
            keep = torch.where(det_iou_.min(dim=1).values[fp_det_idx] >= 0)
            fp_det_idx = fp_det_idx[keep]


        return fp_det_idx, det_iou[tp_det_idx,tp_gt_idx], tp_det_idx, tp_gt_idx




class NuScenesAggregatorFromDetectorImages(NuScenesAggregatorFromDetector):

    def __init__(self,*args,**kwargs):
        super().__init__(*args,**kwargs)


    def __getitem__(self, scene_id: Any):
        results = dict(scene_id=scene_id)
        frames = self.gather_frames(scene_id)
        for i, frame in enumerate(tqdm(frames, desc=f'[{scene_id}] Gathering sequence data', position=2*self.rank+1, leave=False)):

            # Data pipeline
            for method in self.data_pipeline:
                getattr(self, method)(frame)
                if not frame['sd']['LIDAR_TOP']['is_key_frame'] and self.skip_non_keyframe:
                    break
            
            # print(frame['obj_images'])

            # Add results from current frame to aggregated results
            for field in self.data_fields:
                if field not in results:
                    if isinstance(frame[field], dict):
                        results[field] = dict()
                    else:
                        results[field] = []

                if field in frame: 
                    if isinstance(frame[field], dict):
                        for k, v in results[field].items():
                            if k in frame[field]:
                                v.append(frame[field][k])
                            else:
                                v.append(None)
                        for k, v in frame[field].items():
                            if k not in results[field]:
                                results[field][k] = [None] * i
                                results[field][k].append(v)
                    else:
                        results[field].append(frame[field])
                else:
                    for k, v in results[field].items():
                        v.append(None)


            self.clean_up(frame)


        # print({k:v.shape for k,v in results['obj_images'].items()})

        obj_fields = list(filter(startswith('obj_'), results.keys()))
        if len(obj_fields) > 0 and self.obj_aggregator is not None:
            obj_ids = results[obj_fields[0]].keys()
            obj_iterator = zip(
                obj_ids, [{k.replace('obj_', ''): results[k].get(obj_id, None) for k in obj_fields} for obj_id in obj_ids])
            for i, (obj_id, obj_data) in tqdm(enumerate(obj_iterator), total=len(obj_ids), desc=f'[{scene_id}] Reconstructing objects', position=2*self.rank+1, leave=False):
                obj_data = self.obj_aggregator(**obj_data)
                for k, v in obj_data.items():
                    k = 'obj_' + k
                    if i == 0 or k not in results:
                        results[k] = dict()
                    results[k][obj_id] = v

        bg_fields = list(
            filter(startswith(['bg_', 'ego_pose']), results.keys()))
        # Add rejected points back to background
        for k in bg_fields:
            if k == 'ego_pose':
                continue
            l = k.replace('bg_', 'obj_rej_')
            for rej_data in results.get(l, dict()).values():
                for i, v in rej_data.items():
                    results[k][i] = np.concatenate([results[k][i], v])

        if len(bg_fields) > 0 and self.scene_aggregator is not None:
            for _ in tqdm(range(1), desc=f'[{scene_id}] Reconstructing scene', position=2*self.rank+1, leave=False):
                bg_data = {k.replace('bg_', ''): results[k] for k in bg_fields}
                bg_data = self.scene_aggregator(**bg_data)
                for k, v in bg_data.items():
                    k = 'bg_' + k
                    results[k] = v


        return self.process_results(results) 

