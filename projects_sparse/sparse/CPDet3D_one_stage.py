#  modify from https://github.com/SamsungLabs/tr3d and https://github.com/tfzhou/ProtoSeg --zyrant

try:
    import MinkowskiEngine as ME
except ImportError:
    import warnings
    warnings.warn(
        'Please follow `getting_started.md` to install MinkowskiEngine.`')

import torch
from mmcv.cnn import bias_init_with_prob
from mmcv.ops import nms3d, nms3d_normal
from mmcv.runner import BaseModule
from torch import nn

from mmdet3d.models.builder import HEADS, build_loss
from mmdet.core.bbox.builder import BBOX_ASSIGNERS, build_assigner
from mmdet3d.models.dense_heads import TR3DHead

from einops import rearrange, repeat
import torch.nn.functional as F
from .prototype.sinkhorn import distributed_sinkhorn
from .prototype.contrast import momentum_update, l2_normalize, ProjectionHead
from mmdet.core import reduce_mean
from timm.models.layers import trunc_normal_
import numpy as np
from matplotlib import pyplot as plt

@HEADS.register_module()
class CPDet3D_one_stage(TR3DHead):
    def __init__(self,
                 n_classes,
                 in_channels,
                 n_reg_outs,
                 voxel_size,
                 assigner,
                 bbox_loss=dict(type='AxisAlignedIoULoss', reduction='none'),
                 cls_loss=dict(type='FocalLoss', reduction='none'),
                 train_cfg=None,
                 test_cfg=None,
                 num_prototype = 10,
                 update_prototype = True,
                 gamma = 0.999,
                 warm_up = 2000,
                 sim_thr = 0.7,
                 score_thr = 0.1,
                 ):
        super().__init__(
                 n_classes,
                 in_channels,
                 n_reg_outs,
                 voxel_size,
                 assigner,
                 bbox_loss,
                 cls_loss,
                 train_cfg,
                 test_cfg)

        self.n_classes = n_classes
        self.sim_thr = sim_thr
        self.score_thr = score_thr

        self.num_prototype = num_prototype     # to add
        self.update_prototype = update_prototype    # to add
        self.gamma = gamma  # to add
        self.warm_up = warm_up
        self.local_iter = nn.Parameter(torch.zeros(1), requires_grad=False)
        self.prototypes = nn.Parameter(torch.zeros(self.n_classes, self.num_prototype, in_channels),
                                        requires_grad=False)
        
        self.feat_norm = nn.LayerNorm(in_channels)

        self.proj_head = nn.Sequential(
            nn.Linear(in_channels, in_channels),
            nn.BatchNorm1d(in_channels),
            nn.ReLU(inplace=True),
            nn.Linear(in_channels, in_channels))

        trunc_normal_(self.prototypes, std=0.02)

    def l2_normalize(self, x):
        return F.normalize(x, p=2, dim=-1)

    def prototype_learning(self, out_feat, pred, label, feat_proto_sim):

        """
        :param out_feat: [h*w, dim] proposal feature
        :param pred: pred [hw, cls_num] 
        :param label :  [h*w] segmentation label
        :param feat_proto_sim: [h*w, sub_cluster, cls_num]
        """

        pred_score, pred_seg = torch.max(pred, 1)
        mask = label == pred_seg.view(-1)
        mask = mask
        
        
        cosine_similarity = feat_proto_sim

        proto_logits = cosine_similarity
        proto_target = label.clone().float()

        # clustering for each class
        protos = self.prototypes.data.clone()
        
        for k in range(self.n_classes):
            
            # feat_proto_sim used for matching prototype

            # class aware feat_proto_sim
            feat_proto_sim_transpose = feat_proto_sim.view(-1, self.n_classes, self.num_prototype).transpose(1, 2)
            init_q = feat_proto_sim_transpose[..., k] # [n, num_prototype]
            # correct class aware feat_proto_sim
            init_q = init_q[label == k, ...] # [m, num_prototype]

            # debug
            # print('proposal to prototype similar: ', init_q)

            # no such class --> continue
            if init_q.shape[0] == 0:  
                continue

            # indexs: hard proposal to prototype mapping (arg_max)
            # q: one hot proposal to prototype mapping (gumbel_softmax)
            q, indexs = distributed_sinkhorn(init_q) # [m, num_prototype], [m]

            # class aware mask
            m_k = mask[label == k] # [m]
            # class aware feat
            c_k = out_feat[label == k, ...] # [m, d]

            m_k_tile = repeat(m_k, 'n -> n tile', tile=self.num_prototype) # [m, num_prototype]

            m_q = q * m_k_tile  # m x self.num_prototype

            c_k_tile = repeat(m_k, 'n -> n tile', tile=c_k.shape[-1]) # [m, d]

            c_q = c_k * c_k_tile  # n x embedding_dim


            # debug 
            # test same class embedding similar
            # similar = torch.mm(c_q, c_q.t())
            # print('same class similar: ', similar)
            # for i in range(self.n_classes):
            #     if i != k:
            #         similar_different_pro_i = torch.mm(c_q, protos[i].t())
            #         print('similar_different_pro: ' + str(i), similar_different_pro_i)    
            

            f = m_q.transpose(0, 1) @ c_q  # self.num_prototype x embedding_dim

            n = torch.sum(m_q, dim=0)

            # update prototype
            if torch.sum(n) > 0 and self.update_prototype is True:
                f = F.normalize(f, p=2, dim=-1)

                new_value = momentum_update(old_value=protos[k, n != 0, :], new_value=f[n != 0, :],
                                            momentum=self.gamma, iter = self.local_iter, warm_up = self.warm_up, debug=False)
                protos[k, n != 0, :] = new_value

            proto_target[label == k] = indexs.float() + (self.num_prototype * k)

        self.prototypes = nn.Parameter(self.l2_normalize(protos), requires_grad=False)

        return proto_logits, proto_target

    # per level
    def forward_single(self, x):

        # prototype learning
        out_feat = x.features.detach()

        out_feat = self.proj_head(out_feat)
        out_feat = self.feat_norm(out_feat)
        out_feat = self.l2_normalize(out_feat)
        
        self.prototypes.data.copy_(self.l2_normalize(self.prototypes))

        # cosine sim
        # n: h*w, k: num_class, m: num_prototype
        out_feats, feat_proto_sims = [], []
        for permutation in x.decomposition_permutations:
            
            out_feats.append(out_feat[permutation])

            feat_proto_sim = torch.mm(out_feat[permutation], self.prototypes.view(-1, self.prototypes.shape[-1]).t())  #  (n, self.num_class*self.num_proto)

            feat_proto_sims.append(feat_proto_sim)

        return out_feats, feat_proto_sims

    def forward(self, x):

        bbox_preds, cls_preds, points = super().forward(x)

        if self.training:
            out_feats, feat_proto_sims = [], []
            for i in range(len(x)):

                out_feat, feat_proto_sim = self.forward_single(x[i])
                out_feats.append(out_feat)
                feat_proto_sims.append(feat_proto_sim)
            
            self.local_iter += 1
            return bbox_preds, cls_preds, points, out_feats, feat_proto_sims
        
        return bbox_preds, cls_preds, points
        
    
    def forward_train(self, x, gt_bboxes, gt_labels, img_metas):
        bbox_preds, cls_preds, points, out_feats, feat_proto_sims = self(x)
        return self._loss(bbox_preds, cls_preds, points,  out_feats, feat_proto_sims,
                          gt_bboxes, gt_labels, img_metas)
    
    def forward_test(self, x, img_metas):
        bbox_preds, cls_preds, points = self(x)
        return self._get_bboxes(bbox_preds, cls_preds, points, img_metas)
    
    # per scene
    def _loss_single(self,
                     bbox_preds,
                     cls_preds,
                     points,
                     out_feats,
                     feat_proto_sims,
                     gt_bboxes,
                     gt_labels,
                     img_meta):
        
        point_levels = torch.cat([cls_preds[i].new_tensor(i, dtype=torch.long).expand(len(cls_preds[i]))
                            for i in range(len(cls_preds))])
        
        assigned_ids = self.assigner.assign(points, gt_bboxes, gt_labels, img_meta)
        bbox_preds = torch.cat(bbox_preds)
        cls_preds = torch.cat(cls_preds)
        points = torch.cat(points)
        
        # cls loss
        n_classes = cls_preds.shape[1]
        pos_mask = assigned_ids >= 0

        if len(gt_labels) > 0:
            cls_targets = torch.where(pos_mask, gt_labels[assigned_ids], n_classes)
        else:
            cls_targets = gt_labels.new_full((len(pos_mask),), n_classes)
        
        cls_loss = self.cls_loss(cls_preds, cls_targets)

        # bbox loss
        pos_bbox_preds = bbox_preds[pos_mask]
        if pos_mask.sum() > 0:
            pos_points = points[pos_mask]
            pos_bbox_preds = bbox_preds[pos_mask]
            bbox_targets = torch.cat((gt_bboxes.gravity_center, gt_bboxes.tensor[:, 3:]), dim=1)
            pos_bbox_targets = bbox_targets.to(points.device)[assigned_ids][pos_mask]
            if pos_bbox_preds.shape[1] == 6:
                pos_bbox_targets = pos_bbox_targets[:, :6]
            bbox_loss = self.bbox_loss(
                self._bbox_to_loss(
                    self._bbox_pred_to_bbox(pos_points, pos_bbox_preds)),
                self._bbox_to_loss(pos_bbox_targets))            
        else:
            bbox_loss = None

        out_feats = torch.cat(out_feats)
        feat_proto_sims = torch.cat(feat_proto_sims)

        # prototype learning
        contrast_logits, contrast_targets = self.prototype_learning(out_feats, cls_preds, 
                                                                        cls_targets, feat_proto_sims)
        
 
        proto_loss = F.cross_entropy(contrast_logits / 0.07, contrast_targets.long(), ignore_index=self.n_classes) * 0.1

        # fake cls label
        if self.local_iter > self.warm_up:

            # exist_mask
            exist_mask = gt_bboxes.points_in_boxes_all(points).sum(dim=-1) > 0
            exist_mask = ~ ((pos_mask + exist_mask) > 0)

            feat_proto_sim_transpose = feat_proto_sims.view(-1, self.n_classes, self.num_prototype)

            cosine_similarity = feat_proto_sim_transpose

            max_sim_pro, _ = cosine_similarity.max(dim=-1)
            max_sim_pro = F.relu(max_sim_pro, inplace=True)

            max_sim_pro, fake_cls_targets = (max_sim_pro * cls_preds.sigmoid()).max(dim=-1)
            
            # exist_mask & range_mask & level_mask  & score_mask
            # level mask
            label2level = point_levels.new_tensor(self.assigner.label2level)
            label_levels = label2level[fake_cls_targets]
            level_mask = label_levels == point_levels

            # range mask
            point_min, point_max = img_meta['input_ranges'][0], img_meta['input_ranges'][1]
            min_mask = (points[..., 0] >= point_min[0]) * (points[..., 1] >= point_min[1]) * (points[..., 2] >= point_min[2])
            max_mask = (points[..., 0] <= point_max[0]) * (points[..., 1] <= point_max[1]) * (points[..., 2] <= point_max[2])
            range_mask = min_mask * max_mask

            # score mask
            max_scores, _ = cls_preds.sigmoid().max(dim=1)
            score_mask = max_scores > self.score_thr
            # sim_mask = max_cos > self.sim_thr
            # print('sim_mask: ', sim_mask.sum())

            mask = exist_mask * level_mask * score_mask * range_mask
            # mask = exist_mask * score_mask * range_mask

            # print('mask1: ', mask.sum())
            # top_mask = point_levels.new_zeros(mask.shape)
            # top_mask[max_scores.topk(self.top_thr).indices] = 1.
            # top_mask = top_mask.bool()
            # mask = mask * top_mask

            fake_cls_targets = torch.where(mask, fake_cls_targets, n_classes)

            fake_cls_loss = self.cls_loss(cls_preds, fake_cls_targets)

            # pos points
            # debug
            # fake_points = points[mask]
            # colors = np.multiply([
            #     plt.cm.get_cmap('nipy_spectral', 19)((i * 5 + 11) % 18 + 1)[:3] for i in range(18)
            # ], 255).astype(np.uint8)
            # colors = cls_preds.new_tensor(colors)
            # ture_class = fake_cls_targets[mask]
            # fake_colors = colors[ture_class]
            # fake_points = torch.cat((fake_points, fake_colors), dim=1)

        #     return bbox_loss, cls_loss, pos_mask, proto_loss, fake_cls_loss, mask, fake_points

        # return bbox_loss, cls_loss, pos_mask, proto_loss, None, None, None

            return bbox_loss, cls_loss, pos_mask, proto_loss, fake_cls_loss, mask

        return bbox_loss, cls_loss, pos_mask, proto_loss, None, None
    
    
    # debug
    # def _loss(self, bbox_preds, cls_preds, points, out_feats, feat_proto_sims,
    #           gt_bboxes, gt_labels, img_metas):
        
    #     bbox_losses, cls_losses, pos_masks, proto_losses, fake_cls_losses, new_pos_masks, fake_points = [], [], [], [], [], [], []
    #     for i in range(len(img_metas)):
    #         bbox_loss, cls_loss, pos_mask, proto_loss, fake_cls_loss, new_pos_mask, fake_point = self._loss_single(
    #         bbox_preds=[x[i] for x in bbox_preds],
    #         cls_preds=[x[i] for x in cls_preds],
    #         points=[x[i] for x in points],
    #         out_feats =[x[i] for x in out_feats],
    #         feat_proto_sims = [x[i] for x in feat_proto_sims],
    #         img_meta=img_metas[i],
    #         gt_bboxes=gt_bboxes[i],
    #         gt_labels=gt_labels[i])

    #         if bbox_loss is not None:
    #             bbox_losses.append(bbox_loss)
    #         if fake_cls_loss is not None:
    #             fake_cls_losses.append(fake_cls_loss)
    #             new_pos_masks.append(new_pos_mask)
    #             fake_points.append(fake_point)
    #         cls_losses.append(cls_loss)
    #         pos_masks.append(pos_mask)
    #         proto_losses.append(proto_loss)
        
    #     bbox_loss=torch.mean(torch.cat(bbox_losses))
    #     cls_loss=torch.sum(torch.cat(cls_losses)) / torch.sum(torch.cat(pos_masks))
    #     proto_loss=torch.mean(torch.stack(proto_losses))

    #     if len(fake_cls_losses) > 0:
    #         fake_cls_loss = torch.sum(torch.cat(fake_cls_losses)) / torch.sum(torch.cat(new_pos_masks))
    #         return dict(
    #             bbox_loss=bbox_loss,
    #             cls_loss=cls_loss,
    #             proto_loss = proto_loss,
    #             fake_cls_loss = fake_cls_loss,
    #             fake_points = fake_points)
    #     else:
    #         return dict(
    #             bbox_loss=bbox_loss,
    #             cls_loss=cls_loss,
    #             proto_loss = proto_loss)

    def _loss(self, bbox_preds, cls_preds, points, out_feats, feat_proto_sims,
              gt_bboxes, gt_labels, img_metas):
        
        bbox_losses, cls_losses, pos_masks, proto_losses, fake_cls_losses, new_pos_masks = [], [], [], [], [], []
        for i in range(len(img_metas)):
            bbox_loss, cls_loss, pos_mask, proto_loss, fake_cls_loss, new_pos_mask = self._loss_single(
            bbox_preds=[x[i] for x in bbox_preds],
            cls_preds=[x[i] for x in cls_preds],
            points=[x[i] for x in points],
            out_feats =[x[i] for x in out_feats],
            feat_proto_sims = [x[i] for x in feat_proto_sims],
            img_meta=img_metas[i],
            gt_bboxes=gt_bboxes[i],
            gt_labels=gt_labels[i])

            if bbox_loss is not None:
                bbox_losses.append(bbox_loss)
            if fake_cls_loss is not None and new_pos_mask.sum() > 0:
                fake_cls_losses.append(fake_cls_loss)
                new_pos_masks.append(new_pos_mask)

            cls_losses.append(cls_loss)
            pos_masks.append(pos_mask)
            proto_losses.append(proto_loss)
        

        bbox_loss=torch.mean(torch.cat(bbox_losses))
        cls_loss=torch.sum(torch.cat(cls_losses)) / torch.sum(torch.cat(pos_masks))
        proto_loss=torch.mean(torch.stack(proto_losses))

        if len(fake_cls_losses) > 0:
            fake_cls_loss = torch.sum(torch.cat(fake_cls_losses)) / torch.sum(torch.cat(new_pos_masks))
            return dict(
                bbox_loss=bbox_loss,
                cls_loss=cls_loss,
                proto_loss = proto_loss,
                fake_cls_loss = fake_cls_loss)
        else:
            return dict(
                bbox_loss=bbox_loss,
                cls_loss=cls_loss,
                proto_loss = proto_loss)
        

    # def _get_bboxes_single(self, bbox_preds, cls_preds, points, feat_proto_sims, img_meta):
    #     scores = torch.cat(cls_preds).sigmoid()
    #     bbox_preds = torch.cat(bbox_preds)
    #     points = torch.cat(points)
    #     feat_proto_sims = torch.cat(feat_proto_sims)
    #     max_scores, _ = scores.max(dim=1)
    #     cosine_similarity = feat_proto_sims.view(-1, self.n_classes, self.num_prototype)

    #     max_sim_pro, _ = cosine_similarity.max(dim=-1)
    #     max_sim_pro = F.relu(max_sim_pro, inplace=True)

    #     if len(scores) > self.test_cfg.nms_pre > 0:
    #         _, ids = max_scores.topk(self.test_cfg.nms_pre)
    #         bbox_preds = bbox_preds[ids]
    #         scores = scores[ids]
    #         points = points[ids]

    #     boxes = self._bbox_pred_to_bbox(points, bbox_preds)
    #     boxes, scores, labels = self._nms(boxes, scores, img_meta)
    #     return boxes, scores, labels

    # def _get_bboxes(self, bbox_preds, cls_preds, points, feat_proto_sims, img_metas):
    #     results = []
    #     for i in range(len(img_metas)):
    #         result = self._get_bboxes_single(
    #             bbox_preds=[x[i] for x in bbox_preds],
    #             cls_preds=[x[i] for x in cls_preds],
    #             points=[x[i] for x in points],
    #             feat_proto_sims = [x[i] for x in feat_proto_sims],
    #             img_meta=img_metas[i])
    #         results.append(result)

    

    






