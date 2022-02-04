import torch
import torch.nn as nn
import numpy as np
import sys
import os

sys.path.append(os.path.join(os.getcwd(), "lib")) # HACK add the lib folder
# from models.backbone_module import Pointnet2Backbone
# from models.voting_module import VotingModule
from models.proposal_module import ProposalModule
from models.graph_module import GraphModule
from models.caption_module import SceneCaptionModule, TopDownSceneCaptionModule



class TridetrS2c(nn.Module):
    def __init__(self, num_class, vocabulary, embeddings, num_heading_bin,
                 num_size_cluster, mean_size_arr, tridetrmodel,
                 num_proposal=256, num_locals=-1,
                 no_caption=False,use_topdown=False, query_mode="corner",
                 graph_mode="graph_conv", num_graph_steps=0,
                 use_relation=False, graph_aggr="add",
                 use_orientation=False, num_bins=6, use_distance=False,
                 emb_size=300, hidden_size=512,):
        super().__init__()

        self.num_class = num_class
        self.num_heading_bin = num_heading_bin
        self.num_size_cluster = num_size_cluster
        self.mean_size_arr = mean_size_arr
        assert(mean_size_arr.shape[0] == self.num_size_cluster)
        self.num_proposal = num_proposal
        self.no_caption = no_caption
        self.num_graph_steps = num_graph_steps
        # --------- PROPOSAL GENERATION ---------
        self.tridetr = tridetrmodel
        # --------- Graph Module ---------
        if use_relation: assert use_topdown # only enable use_relation in topdown captioning module

        if num_graph_steps > 0:
            self.graph = GraphModule(128, 128, num_graph_steps, num_proposal, 128, num_locals, 
                query_mode, graph_mode, return_edge=use_relation, graph_aggr=graph_aggr, 
                return_orientation=use_orientation, num_bins=num_bins, return_distance=use_distance)

        # --------- Caption generation ---------
        if not no_caption:
            if use_topdown:
                self.caption = TopDownSceneCaptionModule(vocabulary, embeddings, emb_size, 128, 
                    hidden_size, num_proposal, num_locals, query_mode, use_relation)
            else:
                self.caption = SceneCaptionModule(vocabulary, embeddings, emb_size, 128, hidden_size, num_proposal)

    def forward(self, data_dict, use_tf=True, is_eval=False,
                encoder_only=False):
        """ Forward pass of the network

        Args:
            data_dict: dict
                {
                    point_clouds, 
                    lang_feat,
                }

                point_clouds: Variable(torch.cuda.FloatTensor)
                    (B, N, 3 + input_channels) tensor
                    Point cloud to run predicts on
                    Each point in the point-cloud MUST
                    be formatted as (x, y, z, features...)
        Returns:
            end_points: dict
        """

        #######################################
        #                                     #
        #           DETECTION BRANCH          #
        #                                     #
        #######################################
        box_predictions, _ = self.tridetr(data_dict,encoder_only)
        # For the following modules
        data_dict["bbox_feature"] = box_predictions["outputs"]["bbox_features"]
        data_dict["bbox_mask"] = box_predictions["outputs"]["bbox_mask"]
        data_dict["bbox_corner"] = box_predictions["outputs"]["box_corners"]
        # For loss calculation
        data_dict["box_predictions"] = box_predictions
        data_dict["query_xyz"] = box_predictions["outputs"]["query_xyz"]
        # Process data_dict to incorporate necessary tridetr outputs.

        #######################################
        #                                     #
        #           GRAPH ENHANCEMENT         #
        #                                     #
        #######################################

        if self.num_graph_steps > 0: data_dict = self.graph(data_dict)

        #######################################
        #                                     #
        #            CAPTION BRANCH           #
        #                                     #
        #######################################

        # --------- CAPTION GENERATION ---------
        if not self.no_caption:
            data_dict = self.caption(data_dict, use_tf, is_eval)

        return data_dict
