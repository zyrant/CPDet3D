'''
Modify from https://github.com/AIR-DISCOVER/DQS3D --zyrant
'''

from math import floor
from matplotlib import pyplot as plt
import torch
import random
import numpy as np
from copy import deepcopy
import MinkowskiEngine as ME
from mmcv.parallel import MMDistributedDataParallel
from torch import nn
from mmcv.ops import nms3d, nms3d_normal


import mmcv
import time
import os.path as osp
from mmdet3d.models import DETECTORS, build_detector
from mmdet3d.core.bbox.structures.box_3d_mode import Box3DMode
from mmdet3d.core.bbox.structures.depth_box3d import DepthInstance3DBoxes
from mmdet3d.core.evaluation.indoor_eval import indoor_eval
from mmdet3d.models import build_backbone, build_head
from mmdet3d.core import bbox3d2result
from mmdet3d.core.bbox.structures import rotation_3d_in_axis
from mmdet3d.models.detectors import Base3DDetector
from mmdet3d.models.losses.rotated_iou_loss import diff_iou_rotated_3d
from mmdet3d.core.bbox.iou_calculators import axis_aligned_bbox_overlaps_3d, bbox_overlaps_3d
from mmcv.utils import get_logger


def get_module(module):
    if isinstance(module, MMDistributedDataParallel):
        return module.module
    return module


@DETECTORS.register_module()
class CPDet3D_two_stage(Base3DDetector):
    
    def __init__(self,
                model_cfg,
                transformation=dict(),
                alpha=0.99,
                pretrained=None,
                iter_pretrained = None,
                eval_teacher=False,
                train_cfg=None,
                test_cfg=None,
                pesudo_nms_pre=1000, 
                pesudo_iou_thr=0.9, 
                pesudo_score_thr=0.9,
                pesudo_labeled_thr = 0.5,
                ):
        super(CPDet3D_two_stage, self).__init__()
        self.model_cfg = model_cfg
        self.student = build_detector(model_cfg, train_cfg=train_cfg, test_cfg=test_cfg)
        self.teacher = build_detector(deepcopy(model_cfg))

        self.pesudo_nms_pre = pesudo_nms_pre
        self.pesudo_iou_thr = pesudo_iou_thr
        self.pesudo_score_thr = pesudo_score_thr
        self.pesudo_labeled_thr = pesudo_labeled_thr
        
        self.voxel_size = self.student.voxel_size
        
        self.alpha = alpha
        self.transformation = transformation
        
        self.eval_teacher = eval_teacher
        self.pretrained = pretrained
        self.iter_pretrained = iter_pretrained

        self.local_iter = self.student.head.local_iter


    def get_model(self):
        return get_module(self.student)

    def get_t_model(self):
        return get_module(self.teacher)


    def _generate_transformation(self, gathered_img_metas):
        """A stochastic transformation.
        """
        
        transformation = {
            "flipping": [],
            "rotation_angle": [],
            "translation_offset": [],
            "scaling_factor": [],
            "half_mix": [],
            # TODO: [c7w] implement color jittor for RGBs
        }
        
        for batch_id in range(len(gathered_img_metas)):
            # Flipping
            if self.transformation.get("flipping", False):
                transformation["flipping"].append(
                    [random.choice([True, False]) for _ in range(2)]
                )
            else:
                transformation["flipping"].append([False, False])
        
            # Rotation Angle
            if self.transformation.get("rotation_angle") is None:
                transformation["rotation_angle"].append(0.)
            
            elif self.transformation.get("rotation_angle") == "orthogonal":
                transformation["rotation_angle"].append(
                    random.choice(
                        [np.pi / 2 * k for k in range(4)]
                    )
                )
            
            else:
                delta_angle = self.transformation.get("rotation_angle")
                assert isinstance(delta_angle, float)
                transformation["rotation_angle"].append(
                    random.choice(
                        [np.pi / 2 * k for k in range(4)]
                    )
                )
                transformation["rotation_angle"][-1] += np.random.random() * delta_angle * 2 - delta_angle
            
            # translation_offset
            if self.transformation.get("translation_offset") is None:
                transformation["translation_offset"].append(
                    np.array([0, 0, 0])
                )
                
            else:
                
                delta_translation = self.transformation.get("translation_offset")                
                
                def generate_translation():
                    voxel_size = self.voxel_size 
                    upsampled_voxel_size = self.voxel_size * 8
                    max_K = floor(delta_translation / upsampled_voxel_size)
                    K = np.random.randint(-max_K, max_K + 1)
                    # 0-1 -> 0-2 * voxel_size -> - voxel_size - voxel_size ->  (k*8-1) * voxel_size - (k*8+1)
                    return np.random.random() * voxel_size * 2 - voxel_size + K * upsampled_voxel_size
                
                transformation["translation_offset"].append(
                    np.array([generate_translation() for _ in range(3)])
                )
            
            # scaling factor
            if self.transformation.get("scaling_factor") is None:
                transformation["scaling_factor"].append(np.array([1.0, 1.0, 1.0]))
            
            else:
                scaling_offset = self.transformation.get("scaling_factor")
                transformation["scaling_factor"].append(
                    [1.0 + np.random.random() * scaling_offset * 2 - scaling_offset for _ in range(3)]
                    )
                
        
        return transformation
    
    
    def _apply_transformation_pc(self, gathered_points_list, transformation):

        # transformation = {
        #     "flipping": [],
        #     "rotation_angle": [],
        #     "translation_offset": [],
        #     "scaling_factor": [],
        # }

        new_points_list = []
        for i, points in enumerate(gathered_points_list):
            
            # Flipping
            flipping = np.array(transformation["flipping"][i:i+1])
            flipping_X, flipping_Y = flipping[:, 0][:, None], flipping[:, 1][:, None]
            flipping_X = torch.tensor(flipping_X).to(points.device) # [1, 1]
            flipping_Y = torch.tensor(flipping_Y).to(points.device) # [1, 1]
            
            pts_flip_x = points.clone()
            pts_flip_x[..., 0] = pts_flip_x[..., 0] * -1
            
            pts_flip_y = points.clone()
            pts_flip_y[..., 1] = pts_flip_y[..., 1] * -1
            
            points = flipping_X * pts_flip_x + ~flipping_X * points
            points = flipping_Y * pts_flip_y + ~flipping_Y * points
            
            # Rotation_angle
            rotation_angle = torch.tensor(transformation["rotation_angle"][i:i+1]).to(points.device)

            # In mmdet3d_1.0 version, the rotation angle definition of the ground truth (GT) box differs from the previous version by a sign.
            # https://github.com/AIR-DISCOVER/DQS3D/blob/0df1bf0666724eb6f95b1dee9a996184a181e82d/mmdet3d/models/detectors/semi_single_stage_sparse.py#L335
            points[..., :3] = rotation_3d_in_axis(points[..., :3], rotation_angle, axis=-1)

            # translation_offset
            translation_offset = np.stack(transformation["translation_offset"][i:i+1])
            translation_offset = torch.tensor(translation_offset).to(points.device)
            points[..., :3] += translation_offset
            
            # scaling factor
            scaling_factor = np.array(transformation["scaling_factor"][i:i+1])
            scaling_factor = torch.tensor(scaling_factor).to(points.device)
            points[..., :3] *= scaling_factor

            new_points_list.append(points)

        
        return new_points_list
        

    def _apply_transformation_bbox(self, gt_bboxes_3d, transformation, batch_id=None):
    
        bboxes = deepcopy(gt_bboxes_3d)

        if batch_id is None:
            # flipping
            for i in range(len(bboxes)):
                if transformation["flipping"][i][0]:
                    bboxes[i].flip("horizontal")
                if transformation["flipping"][i][1]:
                    bboxes[i].flip("vertical")
            
            
            # rotation_angle
            rot_angle = transformation["rotation_angle"]
            for i in range(len(bboxes)):
                bboxes[i].rotate(rot_angle[i])
            
                
            # translation_offset
            translation_offset = transformation["translation_offset"]
            for i in range(len(bboxes)):
                bboxes[i].tensor[:, :3] += translation_offset[i]
            
                    
            # scaling factor
            scaling_factor = np.array(transformation["scaling_factor"])
            for i in range(len(bboxes)):
                bboxes[i].tensor[:, :3] *= scaling_factor[i]
                bboxes[i].tensor[:, 3:6] *= scaling_factor[i]
        
        else:
             # flipping
            if transformation["flipping"][batch_id][0]:
                bboxes.flip("horizontal")
            if transformation["flipping"][batch_id][1]:
                bboxes.flip("vertical")
            
            
            # rotation_angle
            rot_angle = transformation["rotation_angle"]
            bboxes.rotate(rot_angle[batch_id])
            
                
            # translation_offset
            translation_offset = transformation["translation_offset"]
            bboxes.tensor[:, :3] += translation_offset[batch_id]
            
                    
            # scaling factor
            scaling_factor = np.array(transformation["scaling_factor"])
            bboxes.tensor[:, :3] *= scaling_factor[batch_id]
            bboxes.tensor[:, 3:6] *= scaling_factor[batch_id]

        return bboxes
    
    @staticmethod
    def _bbox_pred_to_bbox(points, bbox_pred):
        """Transform predicted bbox parameters to bbox.
        Args:
            points (Tensor): Final locations of shape (N, 3)
            bbox_pred (Tensor): Predicted bbox parameters of shape (N, 6)
                or (N, 8).
        Returns:
            Tensor: Transformed 3D box of shape (N, 6) or (N, 7).
        """
        if bbox_pred.shape[0] == 0:
            return bbox_pred

        x_center = points[:, 0] + bbox_pred[:, 0]
        y_center = points[:, 1] + bbox_pred[:, 1]
        z_center = points[:, 2] + bbox_pred[:, 2]
        base_bbox = torch.stack([
            x_center,
            y_center,
            z_center,
            bbox_pred[:, 3],
            bbox_pred[:, 4],
            bbox_pred[:, 5]], -1)

        # axis-aligned case
        if bbox_pred.shape[1] == 6:
            return base_bbox

        # rotated case: ..., sin(2a)ln(q), cos(2a)ln(q)
        scale = bbox_pred[:, 3] + bbox_pred[:, 4]
        q = torch.exp(
            torch.sqrt(
                torch.pow(bbox_pred[:, 6], 2) + torch.pow(bbox_pred[:, 7], 2)))
        alpha = 0.5 * torch.atan2(bbox_pred[:, 6], bbox_pred[:, 7])
        return torch.stack(
            (x_center, y_center, z_center, scale / (1 + q), scale /
             (1 + q) * q, bbox_pred[:, 5] + bbox_pred[:, 4], alpha),  
            dim=-1)


    @staticmethod
    def _bbox_to_loss(bbox):
        """Transform box to the axis-aligned or rotated iou loss format.

        Args:
            bbox (Tensor): 3D box of shape (N, 6) or (N, 7).

        Returns:
            Tensor: Transformed 3D box of shape (N, 6) or (N, 7).
        """
        # rotated iou loss accepts (x, y, z, w, h, l, heading)
        if bbox.shape[-1] != 6:
            return bbox

        # axis-aligned case: x, y, z, w, h, l -> x1, y1, z1, x2, y2, z2
        return torch.stack(
            (bbox[..., 0] - bbox[..., 3] / 2, bbox[..., 1] - bbox[..., 4] / 2,
             bbox[..., 2] - bbox[..., 5] / 2, bbox[..., 0] + bbox[..., 3] / 2,
             bbox[..., 1] + bbox[..., 4] / 2, bbox[..., 2] + bbox[..., 5] / 2),
            dim=-1)
    
    def _get_pesudo_bboxes_single(self, bbox_preds, cls_preds, points, student_gt_bbox_3d, img_meta, transformations, batch_id):
        """Generate boxes for a single scene.

        Args:
            center_preds (list[Tensor]): Centerness predictions for all levels.
            bbox_preds (list[Tensor]): Bbox predictions for all levels.
            cls_preds (list[Tensor]): Classification predictions for all
                levels.
            points (list[Tensor]): Final location coordinates for all levels.
            img_meta (dict): Scene meta info.

        Returns:
            tuple[Tensor]: Predicted bounding boxes, scores and labels.
        """

        mlvl_bboxes, mlvl_scores = [], []
        for bbox_pred, cls_pred, point in zip(
                bbox_preds, cls_preds, points):

            scores = cls_pred.sigmoid()

            max_scores, _ = scores.max(dim=1)


            if len(scores) > self.pesudo_nms_pre > 0:
                _, ids = max_scores.topk(self.pesudo_nms_pre)
                bbox_pred = bbox_pred[ids]
                scores = scores[ids]
                point = point[ids]

            bboxes = self._bbox_pred_to_bbox(point, bbox_pred)
            mlvl_bboxes.append(bboxes)
            mlvl_scores.append(scores)

        bboxes = torch.cat(mlvl_bboxes)
        scores = torch.cat(mlvl_scores)

        # NMS
        bboxes, scores, labels, yaw_flag = self._single_scene_nms(
            bboxes, scores, img_meta)
        
        if yaw_flag:
            with_yaw = True
            box_dim = 7
        else:
            with_yaw = False
            box_dim = 6
            bboxes = bboxes[:, :6]
        
        bboxes = img_meta['box_type_3d'](
            bboxes.detach().clone().cpu(),
            box_dim=box_dim,
            with_yaw=with_yaw,
            origin=(.5, .5, .5))
        

        # remove labeled part 
        # correct the bbox according to the transformations
        bboxes = self._apply_transformation_bbox(bboxes, transformations, batch_id)
        scores = scores
        labels = labels


         # collision detection
        bboxes = bboxes.to(scores.device)
        bboxes =  torch.cat((bboxes.gravity_center, bboxes.tensor[:, 3:]),
                          dim=1)
        
        student_gt_bbox_3d = student_gt_bbox_3d.to(scores.device)
        student_gt_bbox_3d = torch.cat((student_gt_bbox_3d.gravity_center, student_gt_bbox_3d.tensor[:, 3:]),
                          dim=1)
        if yaw_flag:
            with_yaw = True
            box_dim = 7
            bboxes_iou = self._bbox_to_loss(bboxes)
            student_gt_bbox_3d_iou = self._bbox_to_loss(student_gt_bbox_3d)
            ious_ts = bbox_overlaps_3d(bboxes_iou, student_gt_bbox_3d_iou, coordinate='depth') # TBD

        else:
            with_yaw = False
            box_dim = 6
            bboxes = bboxes[:, :6]
            student_gt_bbox_3d = student_gt_bbox_3d[:, :6]
            bboxes_iou = self._bbox_to_loss(bboxes)
            student_gt_bbox_3d_iou = self._bbox_to_loss(student_gt_bbox_3d)
            ious_ts = axis_aligned_bbox_overlaps_3d(bboxes_iou, student_gt_bbox_3d_iou, is_aligned=False)

        if len(bboxes) > 0 and len(student_gt_bbox_3d_iou) > 0:
            if len(student_gt_bbox_3d_iou) > 0:
                max_ious_ts, max_ids_ts = ious_ts.max(dim=-1)
                select_ids_ts = torch.where(max_ious_ts < self.pesudo_labeled_thr)
            else:
                select_ids_ts = []

            bboxes = bboxes[select_ids_ts]
            scores = scores[select_ids_ts]
            labels = labels[select_ids_ts]
            


        # generate pseudo labels
        bboxes = img_meta['box_type_3d'](
            bboxes.detach().clone().cpu(),
            box_dim=box_dim,
            with_yaw=with_yaw,
            origin=(.5, .5, .5))


        return bboxes, scores, labels
    
    def _single_scene_nms(self, bboxes, scores, img_meta):
        """Multi-class nms for a single scene.

        Args:
            bboxes (Tensor): Predicted boxes of shape (N_boxes, 6) or
                (N_boxes, 7).
            scores (Tensor): Predicted scores of shape (N_boxes, N_classes).
            img_meta (dict): Scene meta data.

        Returns:
            Tensor: Predicted bboxes.
            Tensor: Predicted scores.
            Tensor: Predicted labels.
        """

        yaw_flag = bboxes.shape[1] == 7

        max_scores, lables = scores.max(dim=1)
        ids = max_scores > self.pesudo_score_thr

        bboxes = bboxes[ids]
        max_scores = max_scores[ids]
        lables = lables[ids]
        
        if yaw_flag:
            nms_function = nms3d
        else:
            bboxes = torch.cat(
                (bboxes, torch.zeros_like(bboxes[:, :1])),
                dim=1)
            nms_function = nms3d_normal

        if len(bboxes) > 0:
            nms_ids = nms_function(bboxes, max_scores,
                                    self.pesudo_iou_thr)
        else:
            nms_ids = []
            
        nms_bboxes = bboxes[nms_ids]
        nms_scores = max_scores[nms_ids]
        nms_labels = lables[nms_ids]

        return nms_bboxes, nms_scores, nms_labels, yaw_flag
    
    @torch.no_grad()
    def _get_pesudo_bboxes(self, teacher_predict, student_gt_bboxes_3d, transformations, img_metas):
        """Generate boxes for all scenes.

        Args:
            bbox_preds (list[list[Tensor]]): Bbox predictions for all scenes.
            cls_preds (list[list[Tensor]]): Classification predictions for all
                scenes.
            points (list[list[Tensor]]): Final location coordinates for all
                scenes.
            img_metas (list[dict]): Meta infos for all scenes.

        Returns:
            list[tuple[Tensor]]: Predicted bboxes, scores, and labels for
                all scenes.
        """
        bboxes = []
        scores =  []
        labels = []
        bbox_preds = teacher_predict[0]
        cls_preds = teacher_predict[1]
        points = teacher_predict[2]

        for i in range(len(img_metas)):
            bbox, score, label = self._get_pesudo_bboxes_single(
                bbox_preds=[x[i] for x in bbox_preds],
                cls_preds=[x[i] for x in cls_preds],
                points=[x[i] for x in points],
                student_gt_bbox_3d = student_gt_bboxes_3d[i],
                img_meta=img_metas[i],
                transformations =  transformations,
                batch_id = i,
                )
            bboxes.append(bbox)
            scores.append(score)
            labels.append(label)
        

        return bboxes, labels
           

    # Return losses
    def forward_train(self,
                      points,
                      gt_bboxes_3d,
                      gt_labels_3d,
                      img_metas,
                      **kwargs):
        
        
        logger = get_logger('sparse')
        # more aug for student input
        # Transform gathered_points and gathered_img_metas to student input
        transformation = self._generate_transformation(img_metas)
        
        # 1. Transform on student input
        student_input_points_ = self._apply_transformation_pc(points, transformation)
        student_gt_bboxes_3d_ = self._apply_transformation_bbox(gt_bboxes_3d, transformation)
        
        student_input_points = student_input_points_
        student_gt_bboxes_3d = student_gt_bboxes_3d_
        gt_labels_3d_ = gt_labels_3d

        for i in range(len(points)):
            point_min, point_max = student_input_points[i].min(0)[0][:3], student_input_points[i].max(0)[0][:3]
            img_metas[i].update(input_ranges=(point_min, point_max))
        
        
        # 2. Get Models
        model = self.get_model()
        ema_model = self.get_t_model()

        # if self.pretrained is not None and self.local_iter == 0:
        #     pretrain_dict = torch.load(self.pretrained)['state_dict']
        #     model.load_state_dict(pretrain_dict)
        #     logger.info(
        #             f'student pretrained loaded: {self.pretrained}')


        # 3. Make teacher Predictions
        with torch.no_grad():
            if self.local_iter == 0:
                if self.pretrained is not None:
                    pretrain_dict = torch.load(self.pretrained)['state_dict']
                    ema_model.load_state_dict(pretrain_dict)
                    logger.info(
                        f'teacher pretrained loaded: {self.pretrained}')        
                if self.iter_pretrained is not None:
                    iter_pretrain_dict = torch.load(self.iter_pretrained)['state_dict']
                    new_dict = {}
                    for k, v in iter_pretrain_dict.items():
                        if 'student' in k:
                            new_k = k.split('.', 1)[-1]
                            new_dict[new_k] = v
                    ema_model.load_state_dict(new_dict)
                    logger.info(
                        f'teacher iter pretrained loaded: {self.iter_pretrained}')                           
            
            teacher_feat = ema_model.extract_feats(points)
            teacher_predict = list(ema_model.head.forward(teacher_feat))

        # 4. Make student Predictions
        student_feat = model.extract_feats(student_input_points)
        student_predict = list(model.head.forward(student_feat))
        
        # self.local_iter += 1

        # 5. Loss calculation
        log_dict = {}

        # 5.1 generate pesudo GTs for unlabeled areas
        with torch.no_grad():
            pesudo_bboxs_unlabeled,  pesudo_labels_unlabeled = self._get_pesudo_bboxes(teacher_predict, 
                                                                                        student_gt_bboxes_3d,
                                                                                        transformation, 
                                                                                        img_metas)            

            for i in range(len(img_metas)):
                student_gt_bboxes_3d[i] = student_gt_bboxes_3d[i].cat([student_gt_bboxes_3d[i], pesudo_bboxs_unlabeled[i]])
                gt_labels_3d[i] = torch.cat((gt_labels_3d[i], pesudo_labels_unlabeled[i].long()), dim=0)


        # 5.2. Supervised Loss (including sparse label)
        supervised_loss = self._supervised_loss(
            student_predict, student_gt_bboxes_3d, gt_labels_3d, img_metas
        )
        

        for k, v in supervised_loss.items():
            log_dict[k + '_all'] = v 
 
        
        return log_dict


    def _supervised_loss(self, student_predict, gt_bboxes_3d, gt_labels_3d, img_metas):
        
        model = self.get_model()
        supervised_loss = model.head._loss(*student_predict,
                                  gt_bboxes_3d, gt_labels_3d, img_metas)
        
        return supervised_loss
    


    def simple_test(self, points, img_metas, *args, **kwargs):
        if self.eval_teacher:
            model = self.get_t_model()  # teacher
        else:
            model = self.get_model()  # student

        x = model.extract_feats(points)
        bbox_list = model.head.forward_test(x, img_metas)      
        bbox_results = [
            bbox3d2result(bboxes, scores, labels)
            for bboxes, scores, labels in bbox_list
        ]
        return bbox_results


    def aug_test(self, points, img_metas, **kwargs):
        assert NotImplementedError, "aug test not implemented"
        # TODO: [c7w] aug_test
        pass


    def extract_feat(self, points, img_metas):
        assert NotImplementedError, "cannot directly use extract_feat in ensembled model"
        pass


    # Visualization functions
    @staticmethod
    def _write_obj(points, out_filename):
        """Write points into ``obj`` format for meshlab visualization.

        Args:
            points (np.ndarray): Points in shape (N, dim).
            out_filename (str): Filename to be saved.
        """
        N = points.shape[0]
        fout = open(out_filename, 'w')
        for i in range(N):
            if points.shape[1] == 6:
                c = points[i, 3:].astype(int)
                fout.write(
                    'v %f %f %f %d %d %d\n' %
                    (points[i, 0], points[i, 1], points[i, 2], c[0], c[1], c[2]))

            else:
                fout.write('v %f %f %f\n' %
                        (points[i, 0], points[i, 1], points[i, 2]))
        fout.close()
    
    @staticmethod
    def _write_oriented_bbox(corners, labels, out_filename):
        """Export corners and labels to .obj file for meshlab.

        Args:
            corners(list[ndarray] or ndarray): [B x 8 x 3] corners of
                boxes for each scene
            labels(list[int]): labels of boxes for each scene
            out_filename(str): Filename.
        """
        colors = np.multiply([
            plt.cm.get_cmap('nipy_spectral', 19)((i * 5 + 11) % 18 + 1)[:3] for i in range(18)
        ], 255).astype(np.uint8).tolist()
        with open(out_filename, 'w') as file:
            for i, (corner, label) in enumerate(zip(corners, labels)):
                c = colors[label]
                for p in corner:
                    file.write(f'v {p[0]} {p[1]} {p[2]} {c[0]} {c[1]} {c[2]}\n')
                j = i * 8 + 1
                for k in [[0, 1, 2, 3], [4, 5, 6, 7], [0, 1, 5, 4],
                        [2, 3, 7, 6], [3, 0, 4, 7], [1, 2, 6, 5]]:
                    file.write('f')
                    for l in k:
                        file.write(f' {j + l}')
                    file.write('\n')
        return

    @staticmethod
    def show_result(points=None,
                    gt_bboxes=None,
                    gt_labels=None,
                    pred_bboxes=None,
                    pred_labels=None,
                    fake_points = None,
                    out_dir=None,
                    filename=None):
        """Convert results into format that is directly readable for meshlab.

        Args:
            points (np.ndarray): Points.
            gt_bboxes (np.ndarray): Ground truth boxes.
            pred_bboxes (np.ndarray): Predicted boxes.
            out_dir (str): Path of output directory
            filename (str): Filename of the current frame.
            show (bool): Visualize the results online. Defaults to False.
            snapshot (bool): Whether to save the online results. Defaults to False.
        """
        result_path = osp.join(out_dir, filename)
        mmcv.mkdir_or_exist(result_path)

        if points is not None:
            CPDet3D_two_stage._write_obj(points, osp.join(result_path, f'{filename}_points.obj'))
        
        if fake_points is not None:
            CPDet3D_two_stage._write_obj(fake_points, osp.join(result_path, f'{filename}_fake_points.obj'))


        if gt_bboxes is not None:
            CPDet3D_two_stage._write_oriented_bbox(gt_bboxes, gt_labels,
                                osp.join(result_path, f'{filename}_gt.obj'))

        if pred_bboxes is not None:
            CPDet3D_two_stage._write_oriented_bbox(pred_bboxes, pred_labels,
                                osp.join(result_path, f'{filename}_pred.obj'))
            

    