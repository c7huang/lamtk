import numpy as np
from typing import Any, List
from tqdm.autonotebook import tqdm
from scipy.spatial.transform import Rotation as R
from .pointcloud_aggregator import ObjectAggregator, SceneAggregator
from ..utils.transforms import apply_transform, affine_transform
from ..utils.box_np_ops import points_in_rbbox

DEFAULT_DATA_PIPELINE = [
    'load_frame_info',
    'load_images',
    'load_img_labels',
    'load_points',
    'load_pts_rgb',
    'load_pts_labels',
    'load_obj_labels',
    'load_obj_boxes',
    'load_obj_boxes3d',
    'extract_obj_points',
    'extract_obj_images'
]

DEFAULT_DATA_FIELDS = [
    'ego_pose', 'timestamp', 'obj_images',
    'obj_pts_xyz', 'obj_pts_feats', 'obj_pts_dir', 'obj_pts_range',
    'obj_pts_rgb', 'obj_pts_labels',
    'obj_labels', 'obj_boxes', 'obj_boxes3d',
    'bg_pts_xyz', 'bg_pts_feats', 'bg_pts_dir', 'bg_pts_range',
    'bg_pts_rgb', 'bg_pts_labels'
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


class DatasetAggregator():
    def __init__(self,
                 data_pipeline=DEFAULT_DATA_PIPELINE,
                 data_fields=DEFAULT_DATA_FIELDS,
                 obj_aggregator=ObjectAggregator(),
                 scene_aggregator=SceneAggregator(),
                 split_scene=False, max_range=106.35,
                 box3d_origin=(0.5, 0.5, 0.5), rank=0):
        self.data_pipeline = data_pipeline
        self.data_fields = data_fields
        self.obj_aggregator = obj_aggregator
        self.scene_aggregator = scene_aggregator
        self.split_scene = split_scene
        self.max_range = max_range
        self.box3d_origin = box3d_origin
        self.rank = rank
        self.scene_ids = []

    def gather_frames(self, scene_id: Any) -> List[Any]:
        return []

    def load_frame_info(self, frame: Any) -> None:
        frame['frame_id'] = 0
        frame['ego_pose'] = np.eye(4)
        frame['timestamp'] = 0.0

    def load_images(self, frame: Any) -> None:
        frame['images'] = dict()
        frame['cam_extrinsics'] = dict()
        frame['cam_intrinsics'] = dict()

    def load_img_labels(self, frame: Any) -> None:
        frame['img_labels'] = dict()

    def load_points(self, frame: Any) -> None:
        frame['pts_xyz'] = np.zeros((0, 3), dtype=np.float32)
        frame['pts_feats'] = np.zeros((0, 0), dtype=np.float32)
        frame['lidar_extrinsic'] = np.eye(4)

    def load_pts_rgb(self, frame: Any) -> None:
        frame['pts_rgb'] = np.zeros((0, 3), dtype=np.float32)

    def load_pts_labels(self, frame: Any) -> None:
        frame['pts_labels'] = np.zeros((0, 1), dtype=int)

    def load_obj_labels(self, frame: Any) -> None:
        frame['obj_labels'] = dict()

    def load_obj_boxes(self, frame: Any) -> None:
        frame['obj_boxes'] = dict()

    def load_obj_boxes3d(self, frame: Any) -> None:
        frame['obj_boxes3d'] = dict()

    def extract_obj_points(self, frame: Any) -> None:
        pts_fields = []
        for k in list(frame.keys()):
            if k.startswith('pts_'):
                pts_fields.append(k)
                frame[f'obj_{k}'] = dict()

        if len(frame.get('obj_boxes3d', dict())) > 0:
            obj_mask = points_in_rbbox(frame['pts_xyz'], np.stack(
                list(frame['obj_boxes3d'].values()))[:, :7],
                origin=self.box3d_origin)
            for i, id in enumerate(frame['obj_boxes3d'].keys()):
                for k in pts_fields:
                    frame[f'obj_{k}'][id] = frame[k][obj_mask[:, i]]
            bg_mask = ~obj_mask.any(-1)
            for k in pts_fields:
                frame[f'bg_{k}'] = frame[k][bg_mask]
        else:
            for k in pts_fields:
                frame[f'bg_{k}'] = frame[k]

    def extract_obj_images(self, frame: Any) -> None:
        frame['obj_images'] = dict()
        images = frame.get('images', None)
        obj_boxes = frame.get('obj_boxes', dict())
        if images is None:
            obj_boxes = dict()
        for obj_id, boxes_i in obj_boxes.items():
            if obj_id not in frame['obj_images']:
                frame['obj_images'][obj_id] = []
            for cam_id, box in boxes_i.items():
                frame['obj_images'][obj_id].append(
                    images[cam_id][box[0]:box[1], box[2]:box[3]])

    def clean_up(self, frame: Any) -> None:
        for field in list(frame.keys()):
            del frame[field]

    def get_obj_infos(self, results: dict) -> None:
        obj_pts_fields = list(filter(startswith('obj_pts_'), results.keys()))
        obj_labels = results.get('obj_labels', dict())
        obj_boxes3d = results.get('obj_boxes3d', dict())
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
                pts_data={k[4:]: results[k][obj_id] for k in obj_pts_fields})
            # TODO: save object images
            # img_data=dict(images=results.get('obj_images', None),
            #               img_labels=results.get('obj_img_labels', None)))
        return obj_infos

    def get_frame_infos(self, results: dict) -> None:
        frame_infos = dict()
        obj_boxes3d = results.get('obj_boxes3d', dict())
        for i in range(len(results['frame_id'])):
            frame_info = frame_infos[results['frame_id'][i]] = dict()
            frame_info['id'] = results['frame_id'][i]
            frame_info['scene_id'] = results['scene_id']
            frame_info['timestamp'] = results.get('timestamp', [0]*(i+1))[i]
            frame_info['ego_pose'] = results.get('ego_pose', [None]*(i+1))[i]
            frame_info['obj_poses'] = dict()
            for obj_id, boxes3d in obj_boxes3d.items():
                if boxes3d[i] is None:
                    continue
                rotmat = R.from_euler('z', -boxes3d[i][6]).as_matrix()
                obj_pose = affine_transform(matrix=rotmat,
                                            translation=boxes3d[i][:3])
                obj_pose = frame_info['ego_pose'] @ obj_pose
                frame_info['obj_poses'][obj_id] = obj_pose.astype(np.float32)
        return frame_infos

    def get_scene_info(self, results: dict) -> None:
        bg_pts_fields = list(filter(startswith('bg_pts_'), results.keys()))
        scene_info = dict(id=results['scene_id'],
                          frame_ids=results['frame_id'])
        scene_info['pts_data'] = {
            k[3:]: [] if self.split_scene else results[k] for k in bg_pts_fields}

        chunk_ids = None
        if self.split_scene:
            # Split scene
            if 'bg_pts_xyz' not in results:
                raise ValueError('split_scene requires pts_xyz to be loaded')
            chunk_ids = {frame_id: [] for frame_id in results['frame_id']}
            bg_pts_masks = []
            # Compute masks for in-range points for each frame
            pbar = tqdm(total=2*len(results.get('ego_pose', [])),
                        desc=f'[{results["scene_id"]}] Splitting scene',
                        leave=False)
            for ego_pose in results.get('ego_pose', []):
                center = apply_transform(ego_pose, np.zeros((1, 3)))[0, :2]
                bg_pts_masks.append(np.linalg.norm(
                    results['bg_pts_xyz'][:, :2] - center,
                    axis=-1) < self.max_range)
                pbar.update()
            num_chunks = 0
            for l in range(len(bg_pts_masks), 0, -1):
                for begin in range(0, len(bg_pts_masks)-l+1):
                    mask = np.logical_and.reduce(bg_pts_masks[begin:begin+l])
                    if mask.sum() == 0:
                        continue
                    for k in bg_pts_fields:
                        scene_info['pts_data'][k[3:]].append(results[k][mask])
                        results[k] = results[k][~mask]
                    num_chunks += 1
                    for i in range(begin, begin+l):
                        chunk_ids[results['frame_id'][i]].append(num_chunks)
                    for i in range(len(bg_pts_masks)):
                        bg_pts_masks[i] = bg_pts_masks[i][~mask]
                pbar.update()
            for k in bg_pts_fields:
                scene_info['pts_data'][k[3:]].insert(0, results[k])
        return scene_info, chunk_ids

    def process_results(self, results: dict) -> dict:
        obj_infos = self.get_obj_infos(results)
        frame_infos = self.get_frame_infos(results)
        scene_info, chunk_ids = self.get_scene_info(results)
        if chunk_ids is not None:
            for frame_id, chunks in chunk_ids.items():
                frame_infos[frame_id]['chunk_ids'] = chunks
        return dict(obj_infos=obj_infos, frame_infos=frame_infos,
                    scene_info=scene_info)

    def __getitem__(self, scene_id: Any):
        results = dict(scene_id=scene_id)
        frames = self.gather_frames(scene_id)
        for i, frame in enumerate(tqdm(frames, desc=f'[{scene_id}] Gathering sequence data', position=2*self.rank+1, leave=False)):

            # Data pipeline
            for method in self.data_pipeline:
                getattr(self, method)(frame)

            # Add results from current frame to aggregated results
            for field in self.data_fields:
                if field not in results:
                    if isinstance(frame[field], dict):
                        results[field] = dict()
                    else:
                        results[field] = []
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

            self.clean_up(frame)

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