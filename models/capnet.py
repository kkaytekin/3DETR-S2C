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

#3DETR Imports:
from models.tridetr.model_3detr import Model3DETR
from models.tridetr.helpers import GenericMLP
from models.tridetr.position_embedding import PositionEmbeddingCoordsSine
from models.tridetr.transformer import (MaskedTransformerEncoder, TransformerDecoder,
                                TransformerDecoderLayer, TransformerEncoder,
                                TransformerEncoderLayer)


class CapNet(nn.Module):
    def __init__(self, num_class, vocabulary, embeddings, num_heading_bin,
                 num_size_cluster, mean_size_arr,
                 #3detr non-defaults
                 pre_encoder,encoder,decoder,dataset_config,
                 #CapNet defaults
                 input_feature_dim=0, num_proposal=256, num_locals=-1,
                 vote_factor=1, sampling="vote_fps",no_caption=False,
                 use_topdown=False, query_mode="corner",
                 graph_mode="graph_conv", num_graph_steps=0,
                 use_relation=False, graph_aggr="add",
                 use_orientation=False, num_bins=6, use_distance=False, use_new=False,
                 emb_size=300, hidden_size=512,
                 #3detr defaults:
                 encoder_dim=256,
                 decoder_dim=256,
                 position_embedding="fourier",
                 mlp_dropout=0.3,
                 num_queries=256,):
        super().__init__()

        self.num_class = num_class
        self.num_heading_bin = num_heading_bin
        self.num_size_cluster = num_size_cluster
        self.mean_size_arr = mean_size_arr
        assert(mean_size_arr.shape[0] == self.num_size_cluster)
        self.input_feature_dim = input_feature_dim
        self.num_proposal = num_proposal
        self.vote_factor = vote_factor
        self.sampling = sampling
        self.no_caption = no_caption
        self.num_graph_steps = num_graph_steps

        # --------- 3DETR ---------
        self.pre_encoder = pre_encoder
        self.encoder = encoder
        if hasattr(self.encoder, "masking_radius"):
            hidden_dims = [encoder_dim]
        else:
            hidden_dims = [encoder_dim, encoder_dim]
        self.encoder_to_decoder_projection = GenericMLP(
            input_dim=encoder_dim,
            hidden_dims=hidden_dims,
            output_dim=decoder_dim,
            norm_fn_name="bn1d",
            activation="relu",
            use_conv=True,
            output_use_activation=True,
            output_use_norm=True,
            output_use_bias=False,
        )
        self.pos_embedding = PositionEmbeddingCoordsSine(
            d_pos=decoder_dim, pos_type=position_embedding, normalize=True
        )
        self.query_projection = GenericMLP(
            input_dim=decoder_dim,
            hidden_dims=[decoder_dim],
            output_dim=decoder_dim,
            use_conv=True,
            output_use_activation=True,
            hidden_use_bias=True,
        )
        self.decoder = decoder
        self.build_mlp_heads(dataset_config, decoder_dim, mlp_dropout)

        self.num_queries = num_queries
        # TODO: Not including box_processor, use ProposalModule instead.
        # self.box_processor = BoxProcessor(dataset_config)
        """
        # --------- PROPOSAL GENERATION ---------
        # Backbone point feature learning
        self.backbone_net = Pointnet2Backbone(input_feature_dim=self.input_feature_dim)
        Hough voting
        self.vgen = VotingModule(self.vote_factor, 256)
        
        # Vote aggregation and """ # object proposal
        # TODO: Add MLP after decoder into Proposal Module
        self.proposal = ProposalModule(num_class, num_heading_bin, num_size_cluster, mean_size_arr, num_proposal, sampling)
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
                #argument for 3detr
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
        # --------- 3DETR Fwd ---------
        point_clouds = data_dict["point_clouds"]
        enc_xyz, enc_features, enc_inds = self.run_encoder(point_clouds)
        enc_features = self.encoder_to_decoder_projection(
            enc_features.permute(1, 2, 0)
        ).permute(2, 0, 1)
        # encoder features: npoints x batch x channel
        # encoder xyz: npoints x batch x 3

        if encoder_only:
            # return: batch x npoints x channels
            return enc_xyz, enc_features.transpose(0, 1)
        # TODO: Either put these hyperparameters into data_dict as initial input,
        #  set them within fwd pass, or handle them within get_query_embeddings().
        #  I checked if its possible to pass them into parser; but they are calculated
        #  within dataloader. So either modify current dataloader to have them within
        #  data_dict, or calculate them during fwd. pass. There might be some related
        #  arguments at parser. Check them and implement them jointly.
        point_cloud_dims = [
            data_dict["point_cloud_dims_min"],
            data_dict["point_cloud_dims_max"],
        ]
        # TODO: Pass enhanced data_dict instead of point_cloud_dims into following function
        query_xyz, query_embed = self.get_query_embeddings(enc_xyz, point_cloud_dims)
        # query_embed: batch x channel x npoint
        enc_pos = self.pos_embedding(enc_xyz, input_range=point_cloud_dims)

        # decoder expects: npoints x batch x channel
        enc_pos = enc_pos.permute(2, 0, 1)
        query_embed = query_embed.permute(2, 0, 1)
        tgt = torch.zeros_like(query_embed)
        box_features = self.decoder(
            tgt, enc_features, query_pos=query_embed, pos=enc_pos)[0]
        # TODO: Handle the following within proposal module.
        """
        Besides the following, lines, self.get_box_predictions() 
        method of 3DETR class should be put into proposal module.
        box_predictions = self.get_box_predictions(
            query_xyz, point_cloud_dims, box_features
        )
        return box_predictions
        """

        """
        # --------- HOUGH VOTING ---------
        data_dict = self.backbone_net(data_dict)
                
        # --------- HOUGH VOTING ---------
        xyz = data_dict["fp2_xyz"]
        features = data_dict["fp2_features"]
        data_dict["seed_inds"] = data_dict["fp2_inds"]
        data_dict["seed_xyz"] = xyz
        data_dict["seed_features"] = features
        
        xyz, features = self.vgen(xyz, features)
        features_norm = torch.norm(features, p=2, dim=1)
        features = features.div(features_norm.unsqueeze(1))
        data_dict["vote_xyz"] = xyz
        data_dict["vote_features"] = features
        """
        # --------- PROPOSAL GENERATION ---------
        #data_dict = self.proposal(xyz, features, data_dict)
        data_dict = self.proposal(query_xyz, box_features, data_dict)
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

    #3DETR forward pass functions
    def run_encoder(self, point_clouds):
        xyz, features = self._break_up_pc(point_clouds)
        pre_enc_xyz, pre_enc_features, pre_enc_inds = self.pre_encoder(xyz, features)
        # xyz: batch x npoints x 3
        # features: batch x channel x npoints
        # inds: batch x npoints

        # nn.MultiHeadAttention in encoder expects npoints x batch x channel features
        pre_enc_features = pre_enc_features.permute(2, 0, 1)

        # xyz points are in batch x npointx channel order
        enc_xyz, enc_features, enc_inds = self.encoder(
            pre_enc_features, xyz=pre_enc_xyz
        )
        if enc_inds is None:
            # encoder does not perform any downsampling
            enc_inds = pre_enc_inds
        else:
            # use gather here to ensure that it works for both FPS and random sampling
            enc_inds = torch.gather(pre_enc_inds, 1, enc_inds)
        return enc_xyz, enc_features, enc_inds

    def get_query_embeddings(self, encoder_xyz, point_cloud_dims):
        query_inds = furthest_point_sample(encoder_xyz, self.num_queries)
        query_inds = query_inds.long()
        query_xyz = [torch.gather(encoder_xyz[..., x], 1, query_inds) for x in range(3)]
        query_xyz = torch.stack(query_xyz)
        query_xyz = query_xyz.permute(1, 2, 0)

        # Gater op above can be replaced by the three lines below from the pointnet2 codebase
        # xyz_flipped = encoder_xyz.transpose(1, 2).contiguous()
        # query_xyz = gather_operation(xyz_flipped, query_inds.int())
        # query_xyz = query_xyz.transpose(1, 2)
        pos_embed = self.pos_embedding(query_xyz, input_range=point_cloud_dims)
        query_embed = self.query_projection(pos_embed)
        return query_xyz, query_embed
