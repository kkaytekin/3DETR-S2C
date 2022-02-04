# Copyright (c) Facebook, Inc. and its affiliates.
# 
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import torch
import torch.nn as nn
import numpy as np
import sys
import os

sys.path.append(os.path.join(os.getcwd(), "lib")) # HACK add the lib folder
from utils.nn_distance import nn_distance, huber_loss
from lib.config import CONF
from tridetr.utils.dist import reduce_dict

# The following values are used to calculate scores the same way as Scan2Cap
FAR_THRESHOLD = 0.6
NEAR_THRESHOLD = 0.3

# Adjust the coefficients of various loss function groups
DETECTION_COEFF = 1
CAP_COEFF = 1
ORI_COEFF = 1
DIST_COEFF = 1

def compute_cap_loss(data_dict, config, weights):
    """ Compute cluster caption loss

    Args:
        data_dict: dict (read-only)

    Returns:
        cap_loss, cap_acc
    """

    # unpack
    pred_caps = data_dict["lang_cap"] # (B, num_words - 1, num_vocabs)
    num_words = data_dict["lang_len"].max()
    target_caps = data_dict["lang_ids"][:, 1:num_words] # (B, num_words - 1)
    
    _, _, num_vocabs = pred_caps.shape

    # caption loss
    criterion = nn.CrossEntropyLoss(ignore_index=0, reduction="none")
    cap_loss = criterion(pred_caps.reshape(-1, num_vocabs), target_caps.reshape(-1))

    # mask out bad boxes
    good_bbox_masks = data_dict["good_bbox_masks"].unsqueeze(1).repeat(1, num_words-1) # (B, num_words - 1)
    good_bbox_masks = good_bbox_masks.reshape(-1) # (B * num_words - 1)
    cap_loss = torch.sum(cap_loss * good_bbox_masks) / (torch.sum(good_bbox_masks) + 1e-6)

    num_good_bbox = data_dict["good_bbox_masks"].sum()
    if num_good_bbox > 0: # only apply loss on the good boxes
        pred_caps = pred_caps[data_dict["good_bbox_masks"]] # num_good_bbox
        target_caps = target_caps[data_dict["good_bbox_masks"]] # num_good_bbox

        # caption acc
        pred_caps = pred_caps.reshape(-1, num_vocabs).argmax(-1) # num_good_bbox * (num_words - 1)
        target_caps = target_caps.reshape(-1) # num_good_bbox * (num_words - 1)
        masks = target_caps != 0
        masked_pred_caps = pred_caps[masks]
        masked_target_caps = target_caps[masks]
        cap_acc = (masked_pred_caps == masked_target_caps).sum().float() / masks.sum().float()
    else: # zero placeholder if there is no good box
        cap_acc = torch.zeros(1)[0].cuda()
    
    return cap_loss, cap_acc

def radian_to_label(radians, num_bins=6):
    """
        convert radians to labels

        Arguments:
            radians: a tensor representing the rotation radians, (batch_size)
            radians: a binary tensor representing the valid masks, (batch_size)
            num_bins: number of bins for discretizing the rotation degrees

        Return:
            labels: a long tensor representing the discretized rotation degree classes, (batch_size)
    """

    boundaries = torch.arange(np.pi / num_bins, np.pi-1e-8, np.pi / num_bins).cuda()
    labels = torch.bucketize(radians, boundaries)

    return labels

def compute_node_orientation_loss(data_dict, num_bins=6):
    object_assignment = data_dict["object_assignment"]
    edge_indices = data_dict["edge_index"]
    edge_preds = data_dict["edge_orientations"]
    num_sources = data_dict["num_edge_source"]
    num_targets = data_dict["num_edge_target"]
    batch_size, num_proposals = object_assignment.shape

    object_rotation_matrices = torch.gather(
        data_dict["scene_object_rotations"], 
        1, 
        object_assignment.view(batch_size, num_proposals, 1, 1).repeat(1, 1, 3, 3)
    ) # batch_size, num_proposals, 3, 3
    object_rotation_masks = torch.gather(
        data_dict["scene_object_rotation_masks"], 
        1, 
        object_assignment
    ) # batch_size, num_proposals
    
    preds = []
    labels = []
    masks = []
    for batch_id in range(batch_size):
        batch_rotations = object_rotation_matrices[batch_id] # num_proposals, 3, 3
        batch_rotation_masks = object_rotation_masks[batch_id] # num_proposals

        batch_num_sources = num_sources[batch_id]
        batch_num_targets = num_targets[batch_id]
        batch_edge_indices = edge_indices[batch_id, :batch_num_sources * batch_num_targets]

        source_indices = edge_indices[batch_id, 0, :batch_num_sources*batch_num_targets].long()
        target_indices = edge_indices[batch_id, 1, :batch_num_sources*batch_num_targets].long()

        source_rot = torch.index_select(batch_rotations, 0, source_indices)
        target_rot = torch.index_select(batch_rotations, 0, target_indices)

        relative_rot = torch.matmul(source_rot, target_rot.transpose(2, 1))
        relative_rot = torch.acos(torch.clamp(0.5 * (torch.diagonal(relative_rot, dim1=-2, dim2=-1).sum(-1) - 1), -1, 1))
        assert torch.isfinite(relative_rot).sum() == source_indices.shape[0]

        source_masks = torch.index_select(batch_rotation_masks, 0, source_indices)
        target_masks = torch.index_select(batch_rotation_masks, 0, target_indices)
        batch_edge_masks = source_masks * target_masks
        
        batch_edge_labels = radian_to_label(relative_rot, num_bins)
        batch_edge_preds = edge_preds[batch_id, :batch_num_sources * batch_num_targets]

        preds.append(batch_edge_preds)
        labels.append(batch_edge_labels)
        masks.append(batch_edge_masks)

    # aggregate
    preds = torch.cat(preds, dim=0)
    labels = torch.cat(labels, dim=0)
    masks = torch.cat(masks, dim=0)

    criterion = nn.CrossEntropyLoss(reduction="none")
    loss = criterion(preds, labels)
    loss = (loss * masks).sum() / (masks.sum() + 1e-8)

    preds = preds.argmax(-1)
    acc = (preds[masks==1] == labels[masks==1]).sum().float() / (masks.sum().float() + 1e-8)

    return loss, acc

def compute_node_distance_loss(data_dict):
    gt_center = data_dict["center_label"][:,:,0:3]
    object_assignment = data_dict["object_assignment"]
    
    gt_center = torch.gather(gt_center, 1, object_assignment.unsqueeze(-1).repeat(1, 1, 3))
    batch_size, _, _ = gt_center.shape

    edge_indices = data_dict["edge_index"]
    edge_preds = data_dict["edge_distances"]
    num_sources = data_dict["num_edge_source"]
    num_targets = data_dict["num_edge_target"]

    preds = []
    labels = []
    for batch_id in range(batch_size):
        batch_gt_center = gt_center[batch_id]

        batch_num_sources = num_sources[batch_id]
        batch_num_targets = num_targets[batch_id]
        batch_edge_indices = edge_indices[batch_id, :batch_num_sources * batch_num_targets]

        source_indices = edge_indices[batch_id, 0, :batch_num_sources*batch_num_targets].long()
        target_indices = edge_indices[batch_id, 1, :batch_num_sources*batch_num_targets].long()

        source_centers = torch.index_select(batch_gt_center, 0, source_indices)
        target_centers = torch.index_select(batch_gt_center, 0, target_indices)

        batch_edge_labels = torch.norm(source_centers - target_centers, dim=1)
        batch_edge_preds = edge_preds[batch_id, :batch_num_sources * batch_num_targets]

        preds.append(batch_edge_preds)
        labels.append(batch_edge_labels)

    # aggregate
    preds = torch.cat(preds, dim=0)
    labels = torch.cat(labels, dim=0)

    criterion = nn.MSELoss()
    loss = criterion(preds, labels)

    return loss

def compute_object_cls_loss(data_dict, weights):
    """ Compute object classification loss

    Args:
        data_dict: dict (read-only)

    Returns:
        cls_loss, cls_acc
    """

    # unpack
    preds = data_dict["enc_preds"] # (B, num_cls)
    targets = data_dict["object_cat"] # (B,)
    
    # classification loss
    criterion = nn.CrossEntropyLoss(weight=torch.FloatTensor(weights).cuda())
    cls_loss = criterion(preds, targets)

    # classification acc
    preds = preds.argmax(-1) # (B,)
    cls_acc = (preds == targets).sum().float() / targets.shape[0]

    return cls_loss, cls_acc

# The main loss calculation function
def get_detr_and_cap_loss(data_dict, device, config, weights, tridetrcriterion,
    detection=True, caption=True, orientation=False, distance=False, num_bins=CONF.TRAIN.NUM_BINS):
    """ Loss functions

    Args:
        data_dict: dict
        config: dataset config instance
        reference: flag (False/True)
    Returns:
        loss: pytorch scalar tensor
        data_dict: dict
    """
    loss = 0.0
    # Obj loss
    if detection:
        if tridetrcriterion is not None:
            loss , loss_dict = tridetrcriterion(data_dict)
            loss = DETECTION_COEFF * loss
            loss_dict_reduced = reduce_dict(loss_dict)
            for key in loss_dict_reduced:
                #track only final decoder losses
                if not key[-1].isdigit():
                    data_dict[key] = loss_dict[key]

        # Calculate pos / neg ratio and objectness accuracy for logging
        pred_center = data_dict["box_predictions"]["outputs"]["center_unnormalized"]
        #pred_center = rotate_preds(pred_center)
        gt_center = data_dict["center_label"][:,:,0:3]
        B = gt_center.shape[0]
        K = pred_center.shape[1]
        K2 = gt_center.shape[1]
        dist1, ind1, dist2, _ = nn_distance(pred_center, gt_center) # dist1: BxK, dist2: BxK2
        euclidean_dist1 = torch.sqrt(dist1+1e-6)
        objectness_label = torch.zeros((B,K), dtype=torch.long).cuda()
        objectness_mask = torch.zeros((B,K)).cuda()
        objectness_label[euclidean_dist1<NEAR_THRESHOLD] = 1
        objectness_mask[euclidean_dist1<NEAR_THRESHOLD] = 1
        objectness_mask[euclidean_dist1>FAR_THRESHOLD] = 1

        data_dict["pos_ratio"] = torch.sum(objectness_label.float().to(device))/float(K*B)
        data_dict["neg_ratio"] = torch.sum(objectness_mask.float())/float(K*B) - data_dict["pos_ratio"]
        obj_pred_val = data_dict["bbox_mask"]
        obj_acc = torch.sum((obj_pred_val==objectness_label.long()).float())/(K*B+1e-6)
        data_dict["obj_acc"] = obj_acc

        # used later in eval
        data_dict["object_assignment"] = ind1

    if caption:
        cap_loss, cap_acc = compute_cap_loss(data_dict, config, weights)
        # store
        data_dict["cap_loss"] = cap_loss
        data_dict["cap_acc"] = cap_acc

        loss += CAP_COEFF * data_dict["cap_loss"]
    else:
        # store
        data_dict["cap_loss"] = torch.zeros(1)[0].to(device)
        data_dict["cap_acc"] = torch.zeros(1)[0].to(device)
        data_dict["pred_ious"] =  torch.zeros(1)[0].to(device)

    if orientation:
        pred_center = data_dict["box_predictions"]["outputs"]["center_unnormalized"]
        gt_center = data_dict["gt_box_centers"][:,:,0:3]
        B = gt_center.shape[0]
        K = pred_center.shape[1]
        K2 = gt_center.shape[1]
        dist1, ind1, dist2, _ = nn_distance(pred_center, gt_center) # dist1: BxK, dist2: BxK2
        data_dict["object_assignment"] = ind1 # (B,K) with values in 0,1,...,K2-1

        ori_loss, ori_acc = compute_node_orientation_loss(data_dict, num_bins)

        # store
        data_dict["ori_loss"] = ori_loss
        data_dict["ori_acc"] = ori_acc

        loss += ORI_COEFF * data_dict["ori_loss"]
    else:
        # store
        data_dict["ori_loss"] = torch.zeros(1)[0].to(device)
        data_dict["ori_acc"] = torch.zeros(1)[0].to(device)

    if distance:
        dist_loss = compute_node_distance_loss(data_dict)

        # store
        data_dict["dist_loss"] = dist_loss
        loss += DIST_COEFF * data_dict["dist_loss"]
    else:
        # store
        data_dict["dist_loss"] = torch.zeros(1)[0].to(device)

    data_dict["loss"] = loss

    return data_dict
