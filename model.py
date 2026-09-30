from layers import *
th.set_printoptions(profile="full")

class Net(nn.Module):
    def __init__(self, args):
        super(Net, self).__init__()
        self.layers = args.layers
        self._act = get_activation(args.model_activation)
        self.TGCN = nn.ModuleList()
        self.TGCN.append(GCMCLayer(args.rating_vals,  # [0, 1]
                                   args.src_in_units,   
                                   args.dst_in_units,  
                                   args.gcn_agg_units,  
                                   args.gcn_out_units,  
                                   args.dropout,        
                                   args.gcn_agg_accum,  
                                   agg_act=self._act,  
                                   share_user_item_param=args.share_param,
                                   device=args.device))
        self.gcn_agg_accum = args.gcn_agg_accum
        self.rating_vals = args.rating_vals
        self.device = args.device
        self.gcn_agg_units = args.gcn_agg_units
        self.src_in_units = args.src_in_units
        for i in range(1, args.layers):
            if args.gcn_agg_accum == 'stack':
                gcn_out_units = args.gcn_out_units * len(args.rating_vals)
            else:
                gcn_out_units = args.gcn_out_units
            self.TGCN.append(GCMCLayer(args.rating_vals,
                                       args.gcn_out_units,
                                       args.gcn_out_units,
                                       gcn_out_units,
                                       args.gcn_out_units,
                                       args.dropout,
                                       args.gcn_agg_accum,
                                       agg_act=self._act,
                                       share_user_item_param=args.share_param,
                                       ini=False,
                                       device=args.device))
        
        self.FGCN = FGCN(args.fdim_drug,
                         args.fdim_disease,
                         args.nhid1,
                         args.nhid2,
                         args.dropout)
        
        # Add a new argument to control whether to use gated attention fusion.
        if False:
            self.gatedfusion = GatedMultimodalLayer(args.gcn_out_units,
                                                    args.gcn_out_units,
                                                    args.gcn_out_units)
        else:
            # Modify Attention class input dimension (topology features + similarity graph features + feature similarity graph features)
            self.attention = Attention(args.gcn_out_units, dropout_rate=args.attention_dropout)
        
        self.decoder = MLPDecoder(in_units=args.gcn_out_units, dropout_rate=args.dropout)
        self.rating_vals = args.rating_vals

        # Early fusion layer for combining drug embeddings and SMILES features.
        self.drug_feat_fusion = nn.Sequential(
                nn.Linear(args.src_in_units *2, args.src_in_units),
                nn.ReLU(),
                nn.Dropout(args.dropout)
            )

    def forward(self, enc_graph, dec_graph,
                drug_graph, drug_sim_feat, drug_feat,
                dis_graph, disease_sim_feat, dis_feat,
                drug_feature_graph=None, disease_feature_graph=None,
                h_smiles=None, Two_Stage=False):

        # Fuse SMILES features with drug features before graph convolution.
        if h_smiles is not None:
            if h_smiles.shape[1] != drug_feat.shape[1]:
                proj = nn.Linear(h_smiles.shape[1], drug_feat.shape[1]).to(drug_feat.device)
                h_smiles = proj(h_smiles)
            if h_smiles.shape[0] != drug_feat.shape[0]:
                if h_smiles.shape[0] < drug_feat.shape[0]:
                    padding = th.zeros(drug_feat.shape[0] - h_smiles.shape[0],
                                       h_smiles.shape[1], device=h_smiles.device)
                    h_smiles = th.cat([h_smiles, padding], dim=0)
                else:
                    h_smiles = h_smiles[:drug_feat.shape[0]]

            # Concatenate and project back to the original drug feature dimension.
            drug_feat = self.drug_feat_fusion(
                th.cat([drug_feat, h_smiles], dim=-1)
            )

        # Topology encoder loop with layer-wise residual accumulation.
        for i in range(0, self.layers):
            drug_o, dis_o = self.TGCN[i](enc_graph, drug_feat, dis_feat, Two_Stage)
            if i == 0:
                drug_out = drug_o
                dis_out = dis_o
            else:
                drug_out = drug_out + drug_o / float(i + 1)
                dis_out = dis_out + dis_o / float(i + 1)
            drug_feat = drug_o
            dis_feat = dis_o


        # Feature convolution
        drug_sim_out, dis_sim_out, drug_sim_only, drug_feat_only, dis_sim_only, dis_feat_only = self.FGCN(
                drug_graph, drug_sim_feat,
                dis_graph, disease_sim_feat,
                drug_feature_graph, disease_feature_graph,
                h_smiles=h_smiles
            )

        # Fusion
        drug_feats = th.stack([drug_out, drug_sim_out], dim=1)
        drug_feats, att_drug = self.attention(drug_feats)

        dis_feats = th.stack([dis_out, dis_sim_out], dim=1)
        dis_feats, att_dis = self.attention(dis_feats)

        # Decode
        pred_ratings = self.decoder(dec_graph, drug_feats, dis_feats)

        return pred_ratings, drug_out, drug_sim_out, dis_out, dis_sim_out