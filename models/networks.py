import torch
import numpy as np
import torch.nn.functional as F
from torch import optim
from torch import nn
import dgl
from dgl import function as fn
from models.flow import get_point_cnf
from models.flow import get_latent_cnf
from utils import truncated_normal, reduce_tensor, standard_normal_logprob


class Encoder(nn.Module):
    def __init__(self, zdim, input_dim=3, use_deterministic_encoder=False):
        super(Encoder, self).__init__()
        self.use_deterministic_encoder = use_deterministic_encoder
        self.zdim = zdim
        self.conv1 = nn.Conv1d(input_dim, 128, 1)
        self.conv2 = nn.Conv1d(128, 128, 1)
        self.conv3 = nn.Conv1d(128, 256, 1)
        self.conv4 = nn.Conv1d(256, 512, 1)
        self.bn1 = nn.BatchNorm1d(128)
        self.bn2 = nn.BatchNorm1d(128)
        self.bn3 = nn.BatchNorm1d(256)
        self.bn4 = nn.BatchNorm1d(512)

        if self.use_deterministic_encoder:
            self.fc1 = nn.Linear(512, 256)
            self.fc2 = nn.Linear(256, 128)
            self.fc_bn1 = nn.BatchNorm1d(256)
            self.fc_bn2 = nn.BatchNorm1d(128)
            self.fc3 = nn.Linear(128, zdim)
        else:
            # Mapping to [c], cmean
            self.fc1_m = nn.Linear(512, 256)
            self.fc2_m = nn.Linear(256, 128)
            self.fc3_m = nn.Linear(128, zdim)
            self.fc_bn1_m = nn.BatchNorm1d(256)
            self.fc_bn2_m = nn.BatchNorm1d(128)

            # Mapping to [c], cmean
            self.fc1_v = nn.Linear(512, 256)
            self.fc2_v = nn.Linear(256, 128)
            self.fc3_v = nn.Linear(128, zdim)
            self.fc_bn1_v = nn.BatchNorm1d(256)
            self.fc_bn2_v = nn.BatchNorm1d(128)

    def forward(self, x):
        print(x.shape)
        x = x.transpose(1, 2)
        print(x.shape)
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        x = self.bn4(self.conv4(x))
        x = torch.max(x, 2, keepdim=True)[0]
        x = x.view(-1, 512)

        if self.use_deterministic_encoder:
            ms = F.relu(self.fc_bn1(self.fc1(x)))
            ms = F.relu(self.fc_bn2(self.fc2(ms)))
            ms = self.fc3(ms)
            m, v = ms, 0
        else:
            m = F.relu(self.fc_bn1_m(self.fc1_m(x)))
            m = F.relu(self.fc_bn2_m(self.fc2_m(m)))
            m = self.fc3_m(m)
            v = F.relu(self.fc_bn1_v(self.fc1_v(x)))
            v = F.relu(self.fc_bn2_v(self.fc2_v(v)))
            v = self.fc3_v(v)

        return m, v

class AtomEncoder(torch.nn.Module):
    def __init__(self, emb_dim, feature_dims, use_scalar_feat=True, n_feats_to_use=None):
        # first element of feature_dims tuple is a list with the length of each categorical feature and the second is the number of scalar features
        super(AtomEncoder, self).__init__()
        self.use_scalar_feat = use_scalar_feat
        self.n_feats_to_use = n_feats_to_use
        self.atom_embedding_list = torch.nn.ModuleList()
        self.num_categorical_features = len(feature_dims[0])
        self.num_scalar_features = feature_dims[1]
        for i, dim in enumerate(feature_dims[0]):
            emb = torch.nn.Embedding(dim, emb_dim)
            torch.nn.init.xavier_uniform_(emb.weight.data)
            self.atom_embedding_list.append(emb)
            if i + 1 == self.n_feats_to_use:
                break

        if self.num_scalar_features > 0:
            self.linear = torch.nn.Linear(self.num_scalar_features, emb_dim)

    def forward(self, x):
        x_embedding = 0
        assert x.shape[1] == self.num_categorical_features + self.num_scalar_features
        for i in range(self.num_categorical_features):
            x_embedding += self.atom_embedding_list[i](x[:, i].long())
            if i + 1 == self.n_feats_to_use:
                break

        if self.num_scalar_features > 0 and self.use_scalar_feat:
            x_embedding += self.linear(x[:, self.num_categorical_features:])
        #if torch.isnan(x_embedding).any():
        #    log('nan')
        return x_embedding

class GraphEncoder(nn.Module):
    def __init__(self, zdim, input_node_features_dim, input_edge_features_dim, hidden_dim, input_coords_dim=3, embedding_dim = 64, negative_slope = 1e-2, use_dist_in_layers = True, use_deterministic_encoder=False):
        super(GraphEncoder, self).__init__()
        self.use_deterministic_encoder = use_deterministic_encoder
        self.use_dist_in_layers = use_dist_in_layers
        self.zdim = zdim
        self.num_layers = 1
        
        self.all_sigmas_dist = [1.5 ** x for x in range(10)]
        # Manually set the number of attention heads to 1
        self.num_att_heads = 1
        
        feature_dims = ([119, 4, 12, 12, 8, 10, 6, 6, 2, 8, 2, 2, 2, 2, 2, 2], 1)
        self.lig_atom_embedding = AtomEncoder(emb_dim=embedding_dim, feature_dims = feature_dims, use_scalar_feat=False, n_feats_to_use=None)
        
        if self.use_dist_in_layers:
            input_edge_features_dim4edge_mlp = input_edge_features_dim + 2 * embedding_dim + 2 * 5 + len(self.all_sigmas_dist)
        else:
            input_edge_features_dim4edge_mlp = input_edge_features_dim + 2 * embedding_dim + 2 * 5
            
        self.lig_edge_mlp = nn.Sequential(
                nn.Linear(input_edge_features_dim4edge_mlp, hidden_dim),
                nn.BatchNorm1d(hidden_dim),
                nn.LeakyReLU(negative_slope=negative_slope),
                nn.Dropout(0.1),
                nn.Linear(hidden_dim, hidden_dim),
                nn.BatchNorm1d(hidden_dim),
            )
    
        self.node_mlp_lig = nn.Sequential(
            nn.Linear(embedding_dim + hidden_dim + 5, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.LeakyReLU(negative_slope=negative_slope),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
        )
        
        self.mu_layer = nn.Sequential(
            nn.Linear(hidden_dim * input_coords_dim, zdim),
            nn.BatchNorm1d(zdim),
        )
        
        self.sigma_layer = nn.Sequential(
            nn.Linear(hidden_dim * input_coords_dim, zdim),
            nn.BatchNorm1d(zdim),
        )
            
    def apply_edges_lig(self, edges):
        if self.use_dist_in_layers:
            x_rel_mag = edges.data['x_rel'] ** 2
            x_rel_mag = torch.sum(x_rel_mag, dim=1, keepdim=True)
            x_rel_mag = torch.cat([torch.exp(-x_rel_mag / sigma) for sigma in self.all_sigmas_dist], dim=-1)
            return {'msg': self.lig_edge_mlp(torch.cat([edges.src['feat'], edges.dst['feat'], edges.data['feat'], x_rel_mag], dim=1))}
            

    def forward(self, lig_graph: dgl.graph):
        coords_lig = lig_graph.ndata['x']
        h_feats_lig = self.lig_atom_embedding(lig_graph.ndata['feat'])
        
        h_feats_lig = torch.cat([h_feats_lig, torch.log(lig_graph.ndata['mu_r_norm'])], dim=1)
        
        original_ligand_node_feats = h_feats_lig
        for ii in range(self.num_layers):
            with lig_graph.local_scope():
                lig_graph.ndata['feat'] = h_feats_lig
                if self.use_dist_in_layers:
                    lig_graph.apply_edges(fn.u_sub_v('x', 'x', 'x_rel'))
                lig_graph.apply_edges(self.apply_edges_lig)
                lig_graph.update_all(fn.copy_edge('msg','m'), fn.mean('m', 'aggr_msg'))
                
                input_node_update_ligand = torch.cat([lig_graph.ndata['feat'], lig_graph.ndata['aggr_msg']], dim=-1)
                
                node_update_ligand = self.node_mlp_lig(input_node_update_ligand)
                h_feats_lig = node_update_ligand
        
        ligs_node_idx = torch.cumsum(lig_graph.batch_num_nodes(), dim=0).tolist()
        ligs_node_idx.insert(0, 0)
        
        mu_batch = torch.zeros((len(ligs_node_idx) - 1, self.zdim))
        sigma_batch = torch.zeros((len(ligs_node_idx) - 1, self.zdim))
        
        lig_info_batch = []
        for idx in range(len(ligs_node_idx) - 1):
            lig_start = ligs_node_idx[idx]
            lig_end = ligs_node_idx[idx + 1]
            
            lig_feats = h_feats_lig[lig_start:lig_end]
            lig_coords = coords_lig[lig_start:lig_end]
            
            # d is the dime of feats
            #d = lig_feats.shape[1]
            #n = lig_feats.shape[0]
            
            #atention = (self.query_lig(lig_feats).view(-1, self.num_att_heads, -1).transpose(0, 1).transpose(1, 2) @ self.key_lig(lig_coords).view(-1, self.num_att_heads, -1).transpose(0, 1)) / np.sqrt(n)
            lig_info = lig_feats.transpose(0, 1) @ lig_coords
            lig_info = torch.flatten(lig_info)
            lig_info_batch.append(lig_info)
        
        lig_info_batch = torch.stack(lig_info_batch)
            
        mu_batch = F.relu(self.mu_layer(lig_info_batch))
        sigma_batch = F.relu(self.sigma_layer(lig_info_batch))

        return mu_batch, sigma_batch
    
# Model
class PointFlow(nn.Module):
    def __init__(self, args):
        super(PointFlow, self).__init__()
        self.input_dim = args.input_dim
        self.zdim = args.zdim
        self.use_latent_flow = args.use_latent_flow
        self.use_deterministic_encoder = args.use_deterministic_encoder
        self.prior_weight = args.prior_weight
        self.recon_weight = args.recon_weight
        self.entropy_weight = args.entropy_weight
        self.distributed = args.distributed
        self.truncate_std = None
        #self.encoder = Encoder(
        #        zdim=args.zdim, input_dim=args.input_dim,
        #        use_deterministic_encoder=args.use_deterministic_encoder)
        self.encoder = GraphEncoder(zdim=args.zdim, 
                                    input_node_features_dim=args.input_node_features_dim, 
                                    input_edge_features_dim=args.input_edge_features_dim, 
                                    hidden_dim=args.hidden_dim,
                                    input_coords_dim=args.input_coords_dim,
                                    use_dist_in_layers=args.use_dist_in_layers)
        
        self.point_cnf = get_point_cnf(args)
        self.latent_cnf = get_latent_cnf(args) if args.use_latent_flow else nn.Sequential()

    @staticmethod
    def sample_gaussian(size, truncate_std=None, gpu=None):
        y = torch.randn(*size).float()
        y = y if gpu is None else y.cuda(gpu)
        if truncate_std is not None:
            truncated_normal(y, mean=0, std=1, trunc_std=truncate_std)
        return y

    @staticmethod
    def reparameterize_gaussian(mean, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn(std.size()).to(mean)
        return mean + std * eps

    @staticmethod
    def gaussian_entropy(logvar):
        const = 0.5 * float(logvar.size(1)) * (1. + np.log(np.pi * 2))
        ent = 0.5 * logvar.sum(dim=1, keepdim=False) + const
        return ent

    def multi_gpu_wrapper(self, f):
        self.encoder = f(self.encoder)
        self.point_cnf = f(self.point_cnf)
        self.latent_cnf = f(self.latent_cnf)

    def make_optimizer(self, args):
        def _get_opt_(params):
            if args.optimizer == 'adam':
                optimizer = optim.Adam(params, lr=args.lr, betas=(args.beta1, args.beta2),
                                       weight_decay=args.weight_decay)
            elif args.optimizer == 'sgd':
                optimizer = torch.optim.SGD(params, lr=args.lr, momentum=args.momentum)
            else:
                assert 0, "args.optimizer should be either 'adam' or 'sgd'"
            return optimizer
        opt = _get_opt_(list(self.encoder.parameters()) + list(self.point_cnf.parameters())
                        + list(list(self.latent_cnf.parameters())))
        return opt

    def forward(self, x, lig_graphs, opt, step, writer=None):
        opt.zero_grad()
        batch_size = x.size(0)
        num_points = x.size(1)
        z_mu, z_sigma = self.encoder(lig_graphs)
        if self.use_deterministic_encoder:
            z = z_mu + 0 * z_sigma
        else:
            z = self.reparameterize_gaussian(z_mu, z_sigma)

        # Compute H[Q(z|X)]
        if self.use_deterministic_encoder:
            entropy = torch.zeros(batch_size).to(z)
        else:
            entropy = self.gaussian_entropy(z_sigma)

        # Compute the prior probability P(z)
        if self.use_latent_flow:
            w, delta_log_pw = self.latent_cnf(z, None, torch.zeros(batch_size, 1).to(z))
            log_pw = standard_normal_logprob(w).view(batch_size, -1).sum(1, keepdim=True)
            delta_log_pw = delta_log_pw.view(batch_size, 1)
            log_pz = log_pw - delta_log_pw
        else:
            log_pz = torch.zeros(batch_size, 1).to(z)

        # Compute the reconstruction likelihood P(X|z)
        z_new = z.view(*z.size())
        z_new = z_new + (log_pz * 0.).mean()
        y, delta_log_py = self.point_cnf(x, z_new, torch.zeros(batch_size, num_points, 1).to(x))
        log_py = standard_normal_logprob(y).view(batch_size, -1).sum(1, keepdim=True)
        delta_log_py = delta_log_py.view(batch_size, num_points, 1).sum(1)
        log_px = log_py - delta_log_py

        # Loss
        entropy_loss = -entropy.mean() * self.entropy_weight
        recon_loss = -log_px.mean() * self.recon_weight
        prior_loss = -log_pz.mean() * self.prior_weight
        loss = entropy_loss + prior_loss + recon_loss
        loss.backward()
        opt.step()

        # LOGGING (after the training)
        if self.distributed:
            entropy_log = reduce_tensor(entropy.mean())
            recon = reduce_tensor(-log_px.mean())
            prior = reduce_tensor(-log_pz.mean())
        else:
            entropy_log = entropy.mean()
            recon = -log_px.mean()
            prior = -log_pz.mean()

        recon_nats = recon / float(x.size(1) * x.size(2))
        prior_nats = prior / float(self.zdim)

        if writer is not None:
            writer.add_scalar('train/entropy', entropy_log, step)
            writer.add_scalar('train/prior', prior, step)
            writer.add_scalar('train/prior(nats)', prior_nats, step)
            writer.add_scalar('train/recon', recon, step)
            writer.add_scalar('train/recon(nats)', recon_nats, step)

        return {
            'entropy': entropy_log.cpu().detach().item()
            if not isinstance(entropy_log, float) else entropy_log,
            'prior_nats': prior_nats,
            'recon_nats': recon_nats,
        }

    def encode(self, lig_graphs):
        z_mu, z_sigma = self.encoder(lig_graphs)
        if self.use_deterministic_encoder:
            return z_mu
        else:
            return self.reparameterize_gaussian(z_mu, z_sigma)

    def decode(self, z, num_points, truncate_std=None):
        # transform points from the prior to a point cloud, conditioned on a shape code
        y = self.sample_gaussian((z.size(0), num_points, self.input_dim), truncate_std)
        x = self.point_cnf(y, z, reverse=True).view(*y.size())
        return y, x

    def sample(self, batch_size, num_points, truncate_std=None, truncate_std_latent=None, gpu=None):
        assert self.use_latent_flow, "Sampling requires `self.use_latent_flow` to be True."
        # Generate the shape code from the prior
        w = self.sample_gaussian((batch_size, self.zdim), truncate_std_latent, gpu=gpu)
        z = self.latent_cnf(w, None, reverse=True).view(*w.size())
        # Sample points conditioned on the shape code
        y = self.sample_gaussian((batch_size, num_points, self.input_dim), truncate_std, gpu=gpu)
        x = self.point_cnf(y, z, reverse=True).view(*y.size())
        return z, x

    def reconstruct(self, x, lig_graphs, num_points=None, truncate_std=None):
        num_points = x.size(1) if num_points is None else num_points
        z = self.encode(lig_graphs)
        _, x = self.decode(z, num_points, truncate_std)
        return x
