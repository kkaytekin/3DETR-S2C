""" 
Modified from: https://github.com/facebookresearch/votenet/blob/master/models/proposal_module.py
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import os
import sys

sys.path.append(os.path.join(os.getcwd(), "lib")) # HACK add the lib folder
import lib.pointnet2.pointnet2_utils
from data.scannet.model_util_scannet import ScannetDatasetConfig
from lib.pointnet2.pointnet2_modules import PointnetSAModuleVotes
from utils.box_util import get_3d_box_batch

# constants
DC = ScannetDatasetConfig()

class ProposalModule(nn.Module):
    def __init__(self, num_class, num_heading_bin, num_size_cluster, mean_size_arr, num_proposal, sampling, seed_feat_dim=256):
        super().__init__() 

        self.num_class = num_class
        self.num_heading_bin = num_heading_bin
        self.num_size_cluster = num_size_cluster
        self.mean_size_arr = mean_size_arr
        self.num_proposal = num_proposal
        self.sampling = sampling
        self.seed_feat_dim = seed_feat_dim
        # Vote clustering
        # self.vote_aggregation = PointnetSAModuleVotes(
        #     npoint=self.num_proposal,
        #     radius=0.3,
        #     nsample=16,
        #     mlp=[self.seed_feat_dim, 128, 128, 128],
        #     use_xyz=True,
        #     normalize_xyz=True
        # )
        #
        # # Object proposal/detection
        # # Objectness scores (2), center residual (3),
        # # heading class+residual (num_heading_bin*2), size class+residual(num_size_cluster*4)
        # self.proposal = nn.Sequential(
        #     nn.Conv1d(128,128,1, bias=False),
        #     nn.BatchNorm1d(128),
        #     nn.ReLU(),
        #     nn.Conv1d(128,128,1, bias=False),
        #     nn.BatchNorm1d(128),
        #     nn.ReLU(),
        # )
        self.proposal = nn.Sequential(
            nn.Conv1d(256,128,1, bias=False),   # This is the only difference, changed in_channel to 256 # TODO: Understand relationship with num_proposal
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Conv1d(128,128,1, bias=False),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Conv1d(128,2+3+num_heading_bin*2+num_size_cluster*4+self.num_class+128,1) # Add 128 for bbox features
        )

    def forward(self, xyz, features, data_dict):
        """
        Old Args:
            xyz: (B,K,3)
            features: (B,C,K)
        New Args:
            xyz: query_xyz
            features: box_features --> This is the important part. It should take part in calculations.
        Returns:
            scores: (B,num_proposal,2+3+NH*2+NS*4) 
        """

        # Farthest point sampling (FPS) on votes
        # xyz, features, fps_inds = self.vote_aggregation(xyz, features)

        # sample_inds = fps_inds

        #data_dict['aggregated_vote_xyz'] = xyz # (batch_size, num_proposal, 3)
        #data_dict['aggregated_vote_features'] = features.permute(0, 2, 1).contiguous() # (batch_size, num_proposal, 128)
        #data_dict['aggregated_vote_inds'] = sample_inds # (batch_size, num_proposal,) # should be 0,1,2,...,num_proposal

        # ----- about input dims -----
        # box_features: num_layers x num_queries x batch x channel

        # box_features change to (num_layers x batch) x channel x num_queries
        # TODO: would hardly converge. do it like in 3detr
        features = features.permute(0, 2, 3, 1)
        num_layers, batch, channel, num_proposal = (
            features.shape[0],
            features.shape[1],
            features.shape[2],
            features.shape[3],
        )
        features = features.reshape(num_layers * batch, channel, num_proposal)
        # --------- PROPOSAL GENERATION ---------
        # mlp outputs: ( (num_layers * batch), 2+3+NH*2+NS*4+128, num_outputs )
        net = self.proposal(features)
        net = net.reshape(num_layers , batch, num_proposal, -1)

        #Compute each decoder proposal in training
        if self.training:
            for i in range(num_layers):
                data_dict["decoder{}_proposal".format(i)] = self.decode_scores(net[i,...].reshape(batch,num_proposal,-1),
                                                                               data_dict,
                                                                               xyz,
                                                                               self.num_class,
                                                                               self.num_heading_bin,
                                                                               self.num_size_cluster,
                                                                               self.mean_size_arr)
                if i == num_layers-1:
                    data_dict = self.decode_scores(net[i,...].reshape(batch,num_proposal,-1),
                                                                               data_dict,
                                                                               xyz,
                                                                               self.num_class,
                                                                               self.num_heading_bin,
                                                                               self.num_size_cluster,
                                                                               self.mean_size_arr)
        #store only the final decoder output in evaluation mode
        else:
            data_dict = self.decode_scores(net[num_layers-1,...].reshape(batch,num_proposal,-1),
                                                                               data_dict,
                                                                               xyz,
                                                                               self.num_class,
                                                                               self.num_heading_bin,
                                                                               self.num_size_cluster,
                                                                               self.mean_size_arr)
        data_dict["num_decoder_layers"] = num_layers
        return data_dict

    def decode_pred_box(self, data_dict):
        # predicted bbox
        pred_center = data_dict["center"].detach().cpu().numpy() # (B,K,3)
        pred_heading_class = torch.argmax(data_dict["heading_scores"], -1) # B,num_proposal
        pred_heading_residual = torch.gather(data_dict["heading_residuals"], 2, pred_heading_class.unsqueeze(-1)) # B,num_proposal,1
        pred_heading_class = pred_heading_class.detach().cpu().numpy() # B,num_proposal
        pred_heading_residual = pred_heading_residual.squeeze(2).detach().cpu().numpy() # B,num_proposal
        pred_size_class = torch.argmax(data_dict["size_scores"], -1) # B,num_proposal
        pred_size_residual = torch.gather(data_dict["size_residuals"], 2, pred_size_class.unsqueeze(-1).unsqueeze(-1).repeat(1,1,1,3)) # B,num_proposal,1,3
        pred_size_class = pred_size_class.detach().cpu().numpy()
        pred_size_residual = pred_size_residual.squeeze(2).detach().cpu().numpy() # B,num_proposal,3

        batch_size, num_proposals, _ = pred_center.shape
        pred_bboxes = []
        for i in range(batch_size):
            # convert the bbox parameters to bbox corners
            pred_obb_batch = DC.param2obb_batch(pred_center[i, :, 0:3], pred_heading_class[i], pred_heading_residual[i],
                        pred_size_class[i], pred_size_residual[i])
            pred_bbox_batch = get_3d_box_batch(pred_obb_batch[:, 3:6], pred_obb_batch[:, 6], pred_obb_batch[:, 0:3])
            pred_bboxes.append(torch.from_numpy(pred_bbox_batch).cuda().unsqueeze(0))

        pred_bboxes = torch.cat(pred_bboxes, dim=0) # batch_size, num_proposals, 8, 3

        return pred_bboxes

    def decode_scores(self, net, data_dict, xyz, num_class, num_heading_bin, num_size_cluster, mean_size_arr):
        """
        decode the predicted parameters for the bounding boxes

        """
        #net_transposed = net.transpose(1,2).contiguous() # (batch_size, 1024, ..)
        batch_size = net.shape[0]
        num_proposal = net.shape[1]

        objectness_scores = net[:,:,0:2]

        #base_xyz = data_dict['aggregated_vote_xyz'] # (batch_size, num_proposal, 3)
        # base_xyz = query_xyz ???
        center = xyz + net[:,:,2:5] # (batch_size, num_proposal, 3)

        heading_scores = net[:,:,5:5+num_heading_bin]
        heading_residuals_normalized = net[:,:,5+num_heading_bin:5+num_heading_bin*2]
        
        size_scores = net[:,:,5+num_heading_bin*2:5+num_heading_bin*2+num_size_cluster]
        size_residuals_normalized = net[:,:,5+num_heading_bin*2+num_size_cluster:5+num_heading_bin*2+num_size_cluster*4].view([batch_size, num_proposal, num_size_cluster, 3]) # Bxnum_proposalxnum_size_clusterx3
        
        sem_cls_scores = net[:,:,5+num_heading_bin*2+num_size_cluster*4:5+num_heading_bin*2+num_size_cluster*4+num_class] # Bxnum_proposalx18
        bbox_feature = net[:,:,5+num_heading_bin*2+num_size_cluster*4+num_class:]

        # store
        data_dict['objectness_scores'] = objectness_scores
        data_dict['center'] = center
        data_dict['heading_scores'] = heading_scores # Bxnum_proposalxnum_heading_bin
        data_dict['heading_residuals_normalized'] = heading_residuals_normalized # Bxnum_proposalxnum_heading_bin (should be -1 to 1)
        data_dict['heading_residuals'] = heading_residuals_normalized * (np.pi/num_heading_bin) # Bxnum_proposalxnum_heading_bin
        data_dict['size_scores'] = size_scores
        data_dict['size_residuals_normalized'] = size_residuals_normalized
        data_dict['size_residuals'] = size_residuals_normalized * torch.from_numpy(mean_size_arr.astype(np.float32)).cuda().unsqueeze(0).unsqueeze(0)
        data_dict['sem_cls_scores'] = sem_cls_scores
        # processed box info
        data_dict["bbox_corner"] = self.decode_pred_box(data_dict) # bounding box corner coordinates
        #data_dict["bbox_feature"] = data_dict["aggregated_vote_features"]
        data_dict["bbox_feature"] = bbox_feature
        data_dict["query_xyz"] = xyz # takes place of data_dict["aggregated_vote_xyz"]
        data_dict["bbox_mask"] = objectness_scores.argmax(-1)
        data_dict['bbox_sems'] = sem_cls_scores.argmax(-1)
        data_dict['sem_cls'] = sem_cls_scores.argmax(-1)

        return data_dict

