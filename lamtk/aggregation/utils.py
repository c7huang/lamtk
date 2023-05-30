import glob
import torch

import numpy as np
import torch.nn.functional as F

from torchvision.transforms import ToTensor
from tqdm.autonotebook import tqdm
from nuscenes.utils.geometry_utils import BoxVisibility

from functools import reduce

def combine_metadata(metadata_dicts):
    metadata = dict(scene_infos={}, obj_infos={}, frame_infos={})
    for metadata_i in metadata_dicts:
        metadata['scene_infos'].update(metadata_i['scene_infos'])
        metadata['obj_infos'].update(metadata_i['obj_infos'])
        metadata['frame_infos'].update(metadata_i['frame_infos'])
    return metadata

def waymo_create_frame_id_map(metadata, waymo_root='data/waymo', split='training'):
    tfrecords = list(sorted(glob.glob(f'{waymo_root}/{split}/*.tfrecord')))
    scene_idx_map = {x.split('-')[1][:-len('_with_camera_labels.tfrecord')]: i for i, x in enumerate(tfrecords)}
    metadata['frame_id_map'] = dict()
    frame_infos = metadata['frame_infos']
    for frame_id in list(frame_infos.keys()):
        scene_id = frame_infos[frame_id]['scene_id']
        if scene_id not in scene_idx_map:
            continue
        scene_idx = scene_idx_map[scene_id]
        frame_idx = int(frame_id[-3:])
        mmdet3d_idx = int(scene_idx * 1e3 + frame_idx)
        if split == 'validation':
            mmdet3d_idx = int(mmdet3d_idx + 1e6)
        metadata['frame_id_map'][mmdet3d_idx] = frame_id
    return metadata

def nus_create_frame_id_map(metadata, nus_root='data/nuscenes', version='v1.0-trainval'):
    from nuscenes import NuScenes
    nusc = NuScenes(version=version, dataroot=nus_root)
    metadata['frame_id_map'] = dict()
    frame_infos = metadata['frame_infos']
    for frame_id in tqdm(frame_infos.keys(), desc='Create frame id map'):
        sd = nusc.get('sample_data', frame_id)
        if sd['is_key_frame']:
            metadata['frame_id_map'][sd['sample_token']] = frame_id
    return metadata

def nus_add_metadata(metadata, nus_root='data/nuscenes', version='v1.0-trainval'):
    from nuscenes import NuScenes
    from nuscenes.utils.splits import train as train_split, val as val_split
    train_split = set(train_split)
    val_split = set(val_split)
    nusc = NuScenes(version=version, dataroot=nus_root)
    # Add scene name
    scene_infos = metadata['scene_infos']
    for scene_id in tqdm(scene_infos.keys(), desc='Add scene info'):
        scene = nusc.get('scene', scene_id)
        scene_infos[scene_id]['name'] = scene['name']
        if scene['name'] in train_split:
            split = 'train'
        elif scene['name'] in val_split:
            split = 'val'
        else:
            split = 'unknown'
        scene_infos[scene_id]['split'] = split
    frame_infos = metadata['frame_infos']
    for frame_id in tqdm(frame_infos.keys(), desc='Add frame info'):
        sd = nusc.get('sample_data', frame_id)
        frame_infos[frame_id]['is_key_frame'] = sd['is_key_frame']
    return metadata

def compute_obj_completeness(metadata, num_vert_bins=5, num_angle_bins=[15, 30, 60, 120]):
    obj_infos = metadata['obj_infos']
    for k in tqdm(obj_infos.keys(), desc='Compute object completeness'):
        xyz = np.fromfile(obj_infos[k]['path'] + '/pts_xyz.bin', dtype=np.float32).reshape(-1, 3)
        obj_infos[k]['num_pts'] = xyz.shape[0]

        completeness = []
        bottom = -obj_infos[k]['size'][2] / 2
        vert_height = obj_infos[k]['size'][2] / num_vert_bins
        for vi in range(num_vert_bins):
            mask = (xyz[:,2] >= bottom+vi*vert_height) & \
                   (xyz[:,2] <= bottom+(vi+1)*vert_height)
            xy = xyz[mask,:2]
            angles = np.arctan2(xy[:,0], xy[:,1])
            for num_bins in num_angle_bins:
                bins = np.linspace(-np.pi, np.pi, num_bins)
                completeness.append(np.unique(np.digitize(angles, bins)).shape[0] / num_bins)
        completeness = np.mean(completeness).astype(np.float32)
        obj_infos[k]['completeness'] = completeness
    return metadata

def filter_metadata_by_scene_ids(metadata, scene_ids):
    metadata_filtered = dict(scene_infos={}, obj_infos={}, frame_infos={})
    for scene_id in scene_ids:
        scene_info = metadata['scene_infos'][scene_id]
        metadata_filtered['scene_infos'][scene_id] = scene_info
        for frame_id in scene_info['frame_ids']:
            frame_info = metadata['frame_infos'][frame_id]
            metadata_filtered['frame_infos'][frame_id] = frame_info
            for obj_id in frame_info['obj_poses']:
                obj_info = metadata['obj_infos'][obj_id]
                metadata_filtered['obj_infos'][obj_id] = obj_info
    return metadata_filtered

def filter_metadata_by_frame_ids(metadata, frame_ids):
    metadata_filtered = dict(scene_infos={}, obj_infos={}, frame_infos={})
    for frame_id in frame_ids:
        frame_info = metadata['frame_infos'][frame_id]
        metadata_filtered['frame_infos'][frame_id] = frame_info
        scene_id = frame_info['scene_id']
        scene_info = metadata['scene_infos'][scene_id]
        metadata_filtered['scene_infos'][scene_id] = scene_info
        for obj_id in frame_info['obj_poses']:
            obj_info = metadata['obj_infos'][obj_id]
            metadata_filtered['obj_infos'][obj_id] = obj_info
    return metadata_filtered



def boxes_in_image(corners2d,corners3d,imsize,vis_level):
    visible =  reduce(torch.logical_and,[
       0 <= corners2d[...,0],  corners2d[...,0] <= imsize[0],
       0 <= corners2d[...,1],  corners2d[...,1] <= imsize[1],
    ])
    
    in_front = corners3d[...,2] > 0.1  # True if a corner is at least 0.1 meter in front of the camera.

    if vis_level == BoxVisibility.ALL:
        torch.logical_and( visible.all(1), in_front.all(1))
    elif vis_level == BoxVisibility.ANY:
        return torch.logical_and(visible.any(1), in_front.any(1))
    elif vis_level == BoxVisibility.NONE:
        return True
    else:
        raise ValueError("vis_level: {} not valid".format(vis_level))

def extract_bboxes(img, bboxes, output_size=(225, 225)):
    # Create a grid of size (N, H_out, W_out, 2)
    N, C, H, W = img.shape
    H_out, W_out = output_size
    grid = torch.zeros((N, H_out, W_out, 2), dtype=torch.float32)

    # Normalize the bounding box coordinates to the range [-1, 1]
    bboxes[:, :2] = 2 * bboxes[:, :2] / torch.tensor([W, H], dtype=torch.float32) - 1
    bboxes[:, 2:] = 2 * bboxes[:, 2:] / torch.tensor([W, H], dtype=torch.float32) - 1

    # Create the grid of sampling locations
    for i in range(N):
        x1, y1, x2, y2 = bboxes[i]
        grid[i, :, :, 0] = torch.linspace(x1, x2, W_out).view(1, 1, -1)
        grid[i, :, :, 1] = torch.linspace(y1, y2, H_out).view(1, -1, 1)
    
    return F.grid_sample(img, grid, mode='bilinear', padding_mode='zeros')

def get_corners_torch(box,rot):
    """
    Returns the bounding box corners.
    :param wlh_factor: Multiply w, l, h by a factor to scale the box.
    :return: <np.float: 3, 8>. First four corners are the ones facing forward.
        The last four are the ones facing backwards.
    """
    import pytorch3d
    w, l, h = box[:,3:4], box[:,4:5], box[:,5:6]

    # 3D bounding box corners. (Convention: x points forward, y to the left, z up.)
    x_corners = l / 2 * torch.tensor([1,  1,  1,  1, -1, -1, -1, -1]).unsqueeze(0)
    y_corners = w / 2 * torch.tensor([1, -1, -1,  1,  1, -1, -1,  1]).unsqueeze(0)
    z_corners = h / 2 * torch.tensor([1,  1, -1, -1,  1,  1, -1, -1]).unsqueeze(0)
    
    corners = torch.cat((x_corners.unsqueeze(1), y_corners.unsqueeze(1), z_corners.unsqueeze(1)),dim=1)
    
    rotation_matrix = pytorch3d.transforms.quaternion_to_matrix(rot)
    corners = torch.bmm(rotation_matrix,corners)
    corners = corners + box[:,:3].unsqueeze(2) # (b, 3, 8) = (b, 3, 8) + (b, 3, 1)
    
    return corners

def get_image_crops_batch(img,
                          l2c,
                          ci,
                          boxes_3d,
                          device,
                          crop_size=(224,224),
                          imsize=(1600,900),
                          visibility=BoxVisibility.ANY):
    import pytorch3d
    num_boxes = boxes_3d.size(0)
    axis_angle = torch.cat([torch.zeros(num_boxes,2).to(device),-boxes_3d[:,6:7] - torch.pi/2],dim=1)
    yaw_quat = pytorch3d.transforms.axis_angle_to_quaternion(axis_angle)
    
    l2c_quat = pytorch3d.transforms.matrix_to_quaternion(l2c[:3,:3])
    
    
    box_temp = boxes_3d.clone()
    box_temp[:,:3] = (torch.cat([box_temp[:,:3],torch.ones([num_boxes,1]).to(device)],dim=1) @ l2c.T)[:,:3]
    mul_quat = pytorch3d.transforms.quaternion_multiply(l2c_quat,yaw_quat)

    corners = get_corners_torch(box_temp,mul_quat).permute(0,2,1)
    
    
    corners_im = ci @ torch.cat([corners.reshape(-1,3),torch.ones([8*num_boxes,1]).to(device)],dim=1).T
    corners_im = corners_im.T[:,:3] / corners_im.T[:,2:3]
    corners_im = corners_im.reshape(num_boxes,8,3)[:,:,:2]
    
    inim = boxes_in_image(corners2d=corners_im,
                          corners3d=corners,
                          imsize=imsize,
                          vis_level=visibility)
    
    idx = torch.where(inim)[0]
    max_ = corners_im[inim].max(1).values
    min_ = corners_im[inim].min(1).values
    
    
    max_[imsize[0] <= max_[:,0] , 0] = imsize[0]
    max_[imsize[1] <= max_[:,1] , 1] = imsize[1]
    min_[min_[:,0] <= 0 , 0] = 0
    min_[min_[:,1] <= 0 , 1] = 0
    
    box2d = torch.cat([min_,max_],dim=1)
    img = ToTensor()(img)
    crops = extract_bboxes(img.unsqueeze(0).repeat(max_.size(0),1,1,1), 
                           box2d, 
                           output_size=crop_size)
    
    return box2d, idx, crops


def get_crops_per_image(img_list,
                        ci_list,
                        l2c_list,
                        boxes_3d,
                        device,
                        crop_size=(224,224),
                        imsize=(1600,900),
                        visibility=BoxVisibility.ANY):
    
    all_idx = []
    all_crops = []
    all_box2d = []
    for i,x in enumerate(img_list):
        l2c = torch.tensor(l2c_list[i]).to(device)
        ci = torch.tensor(ci_list[i]).to(device)
        box2d, idx, crops = get_image_crops_batch(img=img_list[i],
                                                  l2c=l2c,
                                                  ci=ci,
                                                  boxes_3d=boxes_3d,
                                                  device=device,
                                                  crop_size=crop_size,
                                                  imsize=(1600,900),
                                                  visibility=visibility)
        all_idx.append(idx)
        all_crops.append(crops)
        all_box2d.append(box2d)
        
    all_crops, all_idx, all_box2d = torch.cat(all_crops), torch.cat(all_idx), torch.cat(all_box2d)
    unique, counts = torch.unique(all_idx,return_counts=True)

    #assert there is one bbox per crop
    if unique.size(0) < boxes_3d.size(0):
        print("[Waterning in get_crops_per_image] There were fewer unique crops ({}) than boxes ({}). This is probably because of the visibility parameter.".format(unique.size(0),boxes_3d.size(0)))


    before_shape = all_idx.shape# print(,counts)
    duplicates = torch.where(counts > 1)[0]
    mask = torch.zeros_like(all_idx)
    keep_list = []
    # idx_keep_list = []
    for i in duplicates:
        idx = unique[i]
        u_pos = torch.where(all_idx == idx)[0]
        mask[u_pos] = 1
        u_box = all_box2d[u_pos]
        areas = (u_box[:,3]-u_box[:,1]) * (u_box[:,2]-u_box[:,0])
        keep_idx = torch.argmax(areas,0)
        if keep_idx.nelement() > 1:
            print(areas)
            print(keep_idx)
            raise ValueError('keep_idx should be a single value')
        keep_list.append(all_crops[u_pos[keep_idx],...].unsqueeze(0))
        # idx_keep_list.append(all_idx[u_pos[keep_idx]].unsqueeze(0))
        
    if len(keep_list) > 0:
        all_crops = torch.cat([torch.cat(keep_list),all_crops[mask == 0] ])
        all_idx = torch.cat([unique[duplicates],all_idx[mask == 0]])
    else:
        all_crops = all_crops[mask == 0]
        all_idx = all_idx[mask == 0]
    
    assert before_shape[0] == all_idx.shape[0] + counts[duplicates].sum() - duplicates.size(0)

    assert all_idx.size(0) <= boxes_3d.size(0)
    return all_crops, all_idx

 