import io
import glob
import numpy as np
import PIL
try:
    import tensorflow.compat.v1 as tf
    from waymo_open_dataset.utils import frame_utils
    from waymo_open_dataset import dataset_pb2
except:
    pass
from .dataset_aggregator import DatasetAggregator, DEFAULT_DATA_PIPELINE
from ..utils.transforms import apply_transform


class WaymoAggregator(DatasetAggregator):
    CAMERAS = ['UNKNOWN', 'FRONT', 'FRONT_LEFT',
               'FRONT_RIGHT', 'SIDE_LEFT', 'SIDE_RIGHT']
    LASERS = ['UNKNOWN', 'TOP', 'FRONT', 'SIDE_LEFT', 'SIDE_RIGHT', 'REAR']
    CLASSES = ['UNKNOWN', 'Vehicle', 'Pedestrian', 'Sign', 'Cyclist']

    def __init__(self, dataset_path, split='training', *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.dataset_path = dataset_path
        self.split = split
        tfrecords = list(
            sorted(glob.glob(f'{dataset_path}/{split}/*.tfrecord')))
        self.scene_ids = [
            x.split('-')[1][:-len('_with_camera_labels.tfrecord')] for x in tfrecords]

    def gather_frames(self, scene_id):
        frames = []
        tfrecord = f'{self.dataset_path}/{self.split}/segment-{scene_id}_with_camera_labels.tfrecord'
        for i, data in enumerate(tf.data.TFRecordDataset(tfrecord, compression_type='')):
            frame = dataset_pb2.Frame()
            frame.ParseFromString(bytearray(data.numpy()))
            frames.append(dict(idx=i, frame=frame, frame_id=f'{scene_id}_{i:0>3}'))
        return frames

    def load_frame_info(self, frame):
        frame['ego_pose'] = np.asarray(
            frame['frame'].pose.transform).reshape((4, 4))
        frame['timestamp'] = frame['frame'].timestamp_micros / 1e6

    def load_images(self, frame):
        frame['images'] = {self.CAMERAS[i.name]: np.asarray(
            PIL.Image.open(io.BytesIO(i.image))) for i in frame['frame'].images}

    def load_points(self, frame):
        f = frame['frame']
        calibs = sorted(f.context.laser_calibrations, key=lambda c: c.name)
        frame['lidar_extrinsics'] = {self.LASERS[c.name]:
                                     np.asarray(c.extrinsic.transform).reshape((4, 4)) for c in calibs}

        range_images, camera_projections, seg_labels, range_image_top_pose = \
            frame_utils.parse_range_image_and_camera_projection(f)
        points_0, cp_points_0 = frame_utils.convert_range_image_to_point_cloud(
            f, range_images, camera_projections, range_image_top_pose,
            ri_index=0, keep_polar_features=True)
        points_1, cp_points_1 = frame_utils.convert_range_image_to_point_cloud(
            f, range_images, camera_projections, range_image_top_pose,
            ri_index=1, keep_polar_features=True)
        pts_dir_0 = [points_0[i][:, 3:6]-apply_transform(extrinsic, np.asarray([[0, 0, 0]]))
                     for i, extrinsic in enumerate(frame['lidar_extrinsics'].values())]
        pts_dir_1 = [points_1[i][:, 3:6]-apply_transform(extrinsic, np.asarray([[0, 0, 0]]))
                     for i, extrinsic in enumerate(frame['lidar_extrinsics'].values())]

        # Add Lidar and # return label
        for i in range(len(points_0)):
            points_0[i] = np.concatenate([
                points_0[i], np.full((len(points_0[i]),1), i)], axis=1)
        for i in range(len(points_1)):
            points_1[i] = np.concatenate([
                points_1[i], np.full((len(points_1[i]),1), i+5)], axis=1)

        frame['range_images'] = range_images
        frame['seg_labels'] = seg_labels
        frame['points_0'] = points_0
        frame['points_1'] = points_1

        points = np.concatenate([*points_0, *points_1])
        frame['pts_xyz'] = points[:, [3, 4, 5]]
        frame['pts_feats'] = np.concatenate([
            points[:, [1, 2, 6]], np.tile(frame['idx'], (points.shape[0], 1)),
        ], axis=-1).astype(np.float32)
        frame['cp_points'] = np.concatenate([*cp_points_0, *cp_points_1])
        frame['pts_dir'] = np.concatenate([*pts_dir_0, *pts_dir_1])
        frame['pts_range'] = np.linalg.norm(
            frame['pts_dir'], axis=-1, keepdims=True)
        frame['pts_dir'] /= frame['pts_range']

    def load_pts_rgb(self, frame):
        cp_points = frame.get('cp_points', None)
        points_rgb = np.zeros((frame['pts_xyz'].shape[0], 3))
        if cp_points is not None:
            for name, image in frame['images'].items():
                valid_rgb_mask_0 = cp_points[:, 0] == self.CAMERAS.index(name)
                image_coords_0 = cp_points[valid_rgb_mask_0][:, [2, 1]]
                points_rgb[valid_rgb_mask_0] = image[tuple(image_coords_0.T)]
                valid_rgb_mask_1 = cp_points[:, 3] == self.CAMERAS.index(name)
                image_coords_1 = cp_points[valid_rgb_mask_1][:, [5, 4]]
                points_rgb[valid_rgb_mask_1] += image[tuple(image_coords_1.T)]
                points_rgb[valid_rgb_mask_0 & valid_rgb_mask_1] /= 2
        frame['pts_rgb'] = np.clip(points_rgb, 0, 255) / 255

    def load_pts_labels(self, frame):
        # Load semantic labels here
        range_images = frame.get('range_images', None)
        seg_labels = frame.get('seg_labels', [])
        points_0 = frame.get('points_0', None)
        points_1 = frame.get('points_1', None)
        pts_labels = np.zeros((frame['pts_xyz'].shape[0], 1)) - 1
        if len(seg_labels) > 0:
            n0 = sum([pts.shape[0] for pts in points_0])
            ri0_mask = np.asarray(range_images[1][0].data).reshape(
                range_images[1][0].shape.dims)[..., 0] > 0
            ri1_mask = np.asarray(range_images[1][1].data).reshape(
                range_images[1][1].shape.dims)[..., 0] > 0
            seg_labels_0 = np.asarray(seg_labels[1][0].data).reshape(
                seg_labels[1][0].shape.dims)[..., 1:2]
            seg_labels_1 = np.asarray(seg_labels[1][1].data).reshape(
                seg_labels[1][1].shape.dims)[..., 1:2]
            pts_labels[0:points_0[0].shape[0]] = seg_labels_0[ri0_mask]
            pts_labels[n0:n0+points_1[0].shape[0]] = seg_labels_1[ri1_mask]
        frame['pts_labels'] = pts_labels

    def load_obj_labels(self, frame):
        frame['obj_labels'] = dict()
        for l in frame['frame'].laser_labels:
            frame['obj_labels'][l.id] = self.CLASSES[l.type]

    def load_obj_boxes(self, frame):
        boxes = dict()
        for labels in frame['frame'].projected_lidar_labels:
            name = self.CAMERAS[labels.name]
            for label in labels.labels:
                obj_id = label.id[:-len(name)-1]
                if obj_id not in boxes:
                    boxes[obj_id] = dict()
                boxes[obj_id][name] = np.asarray([
                    np.floor(label.box.center_y-label.box.width/2),
                    np.ceil(label.box.center_y+label.box.width/2),
                    np.floor(label.box.center_x-label.box.length/2),
                    np.ceil(label.box.center_x+label.box.length/2)
                ]).astype(int)
        frame['obj_boxes'] = boxes

    def load_obj_boxes3d(self, frame):
        frame['obj_boxes3d'] = dict()
        for l in frame['frame'].laser_labels:
            frame['obj_boxes3d'][l.id] = np.asarray([
                l.box.center_x, l.box.center_y, l.box.center_z,
                l.box.length, l.box.width, l.box.height, -l.box.heading
            ])
