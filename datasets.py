import os
import torch
import numpy as np
from torch.utils.data import Dataset
from torch.utils import data
import random
from copy import deepcopy, copy
import dgl
from utils import apply_random_rotation

# taken from https://github.com/optas/latent_3d_points/blob/8e8f29f8124ed5fc59439e8551ba7ef7567c9a37/src/in_out.py
synsetid_to_cate = {
    '02691156': 'airplane', '02773838': 'bag', '02801938': 'basket',
    '02808440': 'bathtub', '02818832': 'bed', '02828884': 'bench',
    '02876657': 'bottle', '02880940': 'bowl', '02924116': 'bus',
    '02933112': 'cabinet', '02747177': 'can', '02942699': 'camera',
    '02954340': 'cap', '02958343': 'car', '03001627': 'chair',
    '03046257': 'clock', '03207941': 'dishwasher', '03211117': 'monitor',
    '04379243': 'table', '04401088': 'telephone', '02946921': 'tin_can',
    '04460130': 'tower', '04468005': 'train', '03085013': 'keyboard',
    '03261776': 'earphone', '03325088': 'faucet', '03337140': 'file',
    '03467517': 'guitar', '03513137': 'helmet', '03593526': 'jar',
    '03624134': 'knife', '03636649': 'lamp', '03642806': 'laptop',
    '03691459': 'speaker', '03710193': 'mailbox', '03759954': 'microphone',
    '03761084': 'microwave', '03790512': 'motorcycle', '03797390': 'mug',
    '03928116': 'piano', '03938244': 'pillow', '03948459': 'pistol',
    '03991062': 'pot', '04004475': 'printer', '04074963': 'remote_control',
    '04090263': 'rifle', '04099429': 'rocket', '04225987': 'skateboard',
    '04256520': 'sofa', '04330267': 'stove', '04530566': 'vessel',
    '04554684': 'washer', '02992529': 'cellphone',
    '02843684': 'birdhouse', '02871439': 'bookshelf',
    # '02858304': 'boat', no boat in our dataset, merged into vessels
    # '02834778': 'bicycle', not in our taxonomy
}
cate_to_synsetid = {v: k for k, v in synsetid_to_cate.items()}


class Uniform15KPC(Dataset):
    def __init__(self, root_dir, subdirs, tr_sample_size=10000,
                 te_sample_size=10000, split='train', scale=1.,
                 normalize_per_shape=False, random_subsample=False,
                 normalize_std_per_axis=False,
                 all_points_mean=None, all_points_std=None,
                 input_dim=3):
        self.root_dir = root_dir
        self.split = split
        self.in_tr_sample_size = tr_sample_size
        self.in_te_sample_size = te_sample_size
        self.subdirs = subdirs
        self.scale = scale
        self.random_subsample = random_subsample
        self.input_dim = input_dim

        self.all_cate_mids = []
        self.cate_idx_lst = []
        self.all_points = []
        for cate_idx, subd in enumerate(self.subdirs):
            # NOTE: [subd] here is synset id
            sub_path = os.path.join(root_dir, subd, self.split)
            if not os.path.isdir(sub_path):
                print("Directory missing : %s" % sub_path)
                continue

            all_mids = []
            for x in os.listdir(sub_path):
                if not x.endswith('.npy'):
                    continue
                all_mids.append(os.path.join(self.split, x[:-len('.npy')]))

            # NOTE: [mid] contains the split: i.e. "train/<mid>" or "val/<mid>" or "test/<mid>"
            for mid in all_mids:
                # obj_fname = os.path.join(sub_path, x)
                obj_fname = os.path.join(root_dir, subd, mid + ".npy")
                try:
                    point_cloud = np.load(obj_fname)  # (15k, 3)
                except:
                    continue

                assert point_cloud.shape[0] == 15000
                self.all_points.append(point_cloud[np.newaxis, ...])
                self.cate_idx_lst.append(cate_idx)
                self.all_cate_mids.append((subd, mid))

        # Shuffle the index deterministically (based on the number of examples)
        self.shuffle_idx = list(range(len(self.all_points)))
        random.Random(38383).shuffle(self.shuffle_idx)
        self.cate_idx_lst = [self.cate_idx_lst[i] for i in self.shuffle_idx]
        self.all_points = [self.all_points[i] for i in self.shuffle_idx]
        self.all_cate_mids = [self.all_cate_mids[i] for i in self.shuffle_idx]

        # Normalization
        self.all_points = np.concatenate(self.all_points)  # (N, 15000, 3)
        self.normalize_per_shape = normalize_per_shape
        self.normalize_std_per_axis = normalize_std_per_axis
        if all_points_mean is not None and all_points_std is not None:  # using loaded dataset stats
            self.all_points_mean = all_points_mean
            self.all_points_std = all_points_std
        elif self.normalize_per_shape:  # per shape normalization
            B, N = self.all_points.shape[:2]
            self.all_points_mean = self.all_points.mean(axis=1).reshape(B, 1, input_dim)
            if normalize_std_per_axis:
                self.all_points_std = self.all_points.reshape(B, N, -1).std(axis=1).reshape(B, 1, input_dim)
            else:
                self.all_points_std = self.all_points.reshape(B, -1).std(axis=1).reshape(B, 1, 1)
        else:  # normalize across the dataset
            self.all_points_mean = self.all_points.reshape(-1, input_dim).mean(axis=0).reshape(1, 1, input_dim)
            if normalize_std_per_axis:
                self.all_points_std = self.all_points.reshape(-1, input_dim).std(axis=0).reshape(1, 1, input_dim)
            else:
                self.all_points_std = self.all_points.reshape(-1).std(axis=0).reshape(1, 1, 1)

        self.all_points = (self.all_points - self.all_points_mean) / self.all_points_std
        self.train_points = self.all_points[:, :10000]
        self.test_points = self.all_points[:, 10000:]

        self.tr_sample_size = min(10000, tr_sample_size)
        self.te_sample_size = min(5000, te_sample_size)
        print("Total number of data:%d" % len(self.train_points))
        print("Min number of points: (train)%d (test)%d"
              % (self.tr_sample_size, self.te_sample_size))
        assert self.scale == 1, "Scale (!= 1) is deprecated"

    def get_pc_stats(self, idx):
        if self.normalize_per_shape:
            m = self.all_points_mean[idx].reshape(1, self.input_dim)
            s = self.all_points_std[idx].reshape(1, -1)
            return m, s

        return self.all_points_mean.reshape(1, -1), self.all_points_std.reshape(1, -1)

    def renormalize(self, mean, std):
        self.all_points = self.all_points * self.all_points_std + self.all_points_mean
        self.all_points_mean = mean
        self.all_points_std = std
        self.all_points = (self.all_points - self.all_points_mean) / self.all_points_std
        self.train_points = self.all_points[:, :10000]
        self.test_points = self.all_points[:, 10000:]

    def __len__(self):
        return len(self.train_points)

    def __getitem__(self, idx):
        tr_out = self.train_points[idx]
        if self.random_subsample:
            tr_idxs = np.random.choice(tr_out.shape[0], self.tr_sample_size)
        else:
            tr_idxs = np.arange(self.tr_sample_size)
        tr_out = torch.from_numpy(tr_out[tr_idxs, :]).float()

        te_out = self.test_points[idx]
        if self.random_subsample:
            te_idxs = np.random.choice(te_out.shape[0], self.te_sample_size)
        else:
            te_idxs = np.arange(self.te_sample_size)
        te_out = torch.from_numpy(te_out[te_idxs, :]).float()

        m, s = self.get_pc_stats(idx)
        cate_idx = self.cate_idx_lst[idx]
        sid, mid = self.all_cate_mids[idx]

        return {
            'idx': idx,
            'train_points': tr_out,
            'test_points': te_out,
            'mean': m, 'std': s, 'cate_idx': cate_idx,
            'sid': sid, 'mid': mid
        }


class ModelNet40PointClouds(Uniform15KPC):
    def __init__(self, root_dir="data/ModelNet40.PC15k",
                 tr_sample_size=10000, te_sample_size=2048,
                 split='train', scale=1., normalize_per_shape=False,
                 normalize_std_per_axis=False,
                 random_subsample=False,
                 all_points_mean=None, all_points_std=None):
        self.root_dir = root_dir
        self.split = split
        assert self.split in ['train', 'test']
        self.sample_size = tr_sample_size
        self.cates = []
        for cate in os.listdir(root_dir):
            if os.path.isdir(os.path.join(root_dir, cate)) \
                    and os.path.isdir(os.path.join(root_dir, cate, 'train')) \
                    and os.path.isdir(os.path.join(root_dir, cate, 'test')):
                self.cates.append(cate)
        assert len(self.cates) == 40, "%s %s" % (len(self.cates), self.cates)

        # For non-aligned MN
        # self.gravity_axis = 0
        # self.display_axis_order = [0,1,2]

        # Aligned MN has same axis-order as SN
        self.gravity_axis = 1
        self.display_axis_order = [0, 2, 1]

        super(ModelNet40PointClouds, self).__init__(
            root_dir, self.cates, tr_sample_size=tr_sample_size,
            te_sample_size=te_sample_size, split=split, scale=scale,
            normalize_per_shape=normalize_per_shape,
            normalize_std_per_axis=normalize_std_per_axis,
            random_subsample=random_subsample,
            all_points_mean=all_points_mean, all_points_std=all_points_std,
            input_dim=3)


class ModelNet10PointClouds(Uniform15KPC):
    def __init__(self, root_dir="data/ModelNet10.PC15k",
                 tr_sample_size=10000, te_sample_size=2048,
                 split='train', scale=1., normalize_per_shape=False,
                 normalize_std_per_axis=False,
                 random_subsample=False,
                 all_points_mean=None, all_points_std=None):
        self.root_dir = root_dir
        self.split = split
        assert self.split in ['train', 'test']
        self.cates = []
        for cate in os.listdir(root_dir):
            if os.path.isdir(os.path.join(root_dir, cate)) \
                    and os.path.isdir(os.path.join(root_dir, cate, 'train')) \
                    and os.path.isdir(os.path.join(root_dir, cate, 'test')):
                self.cates.append(cate)
        assert len(self.cates) == 10

        # That's prealigned MN
        # self.gravity_axis = 0
        # self.display_axis_order = [0,1,2]

        # Aligned MN has same axis-order as SN
        self.gravity_axis = 1
        self.display_axis_order = [0, 2, 1]

        super(ModelNet10PointClouds, self).__init__(
            root_dir, self.cates, tr_sample_size=tr_sample_size,
            te_sample_size=te_sample_size, split=split, scale=scale,
            normalize_per_shape=normalize_per_shape,
            normalize_std_per_axis=normalize_std_per_axis,
            random_subsample=random_subsample,
            all_points_mean=all_points_mean, all_points_std=all_points_std,
            input_dim=3)


class ShapeNet15kPointClouds(Uniform15KPC):
    def __init__(self, root_dir="data/ShapeNetCore.v2.PC15k",
                 categories=['airplane'], tr_sample_size=10000, te_sample_size=2048,
                 split='train', scale=1., normalize_per_shape=False,
                 normalize_std_per_axis=False,
                 random_subsample=False,
                 all_points_mean=None, all_points_std=None):
        self.root_dir = root_dir
        self.split = split
        assert self.split in ['train', 'test', 'val']
        self.tr_sample_size = tr_sample_size
        self.te_sample_size = te_sample_size
        self.cates = categories
        if 'all' in categories:
            self.synset_ids = list(cate_to_synsetid.values())
        else:
            self.synset_ids = [cate_to_synsetid[c] for c in self.cates]

        # assert 'v2' in root_dir, "Only supporting v2 right now."
        self.gravity_axis = 1
        self.display_axis_order = [0, 2, 1]

        super(ShapeNet15kPointClouds, self).__init__(
            root_dir, self.synset_ids,
            tr_sample_size=tr_sample_size,
            te_sample_size=te_sample_size,
            split=split, scale=scale,
            normalize_per_shape=normalize_per_shape,
            normalize_std_per_axis=normalize_std_per_axis,
            random_subsample=random_subsample,
            all_points_mean=all_points_mean, all_points_std=all_points_std,
            input_dim=3)


def init_np_seed(worker_id):
    seed = torch.initial_seed()
    np.random.seed(seed % 4294967296)


def _get_MN40_datasets_(args, data_dir=None):
    tr_dataset = ModelNet40PointClouds(
        split='train',
        tr_sample_size=args.tr_max_sample_points,
        te_sample_size=args.te_max_sample_points,
        root_dir=(args.data_dir if data_dir is None else data_dir),
        normalize_per_shape=args.normalize_per_shape,
        normalize_std_per_axis=args.normalize_std_per_axis,
        random_subsample=True)
    te_dataset = ModelNet40PointClouds(
        split='test',
        tr_sample_size=args.tr_max_sample_points,
        te_sample_size=args.te_max_sample_points,
        root_dir=(args.data_dir if data_dir is None else data_dir),
        normalize_per_shape=args.normalize_per_shape,
        normalize_std_per_axis=args.normalize_std_per_axis,
        all_points_mean=tr_dataset.all_points_mean,
        all_points_std=tr_dataset.all_points_std,
    )

    return tr_dataset, te_dataset


def _get_MN10_datasets_(args, data_dir=None):
    tr_dataset = ModelNet10PointClouds(
        split='train',
        tr_sample_size=args.tr_max_sample_points,
        te_sample_size=args.te_max_sample_points,
        root_dir=(args.data_dir if data_dir is None else data_dir),
        normalize_per_shape=args.normalize_per_shape,
        normalize_std_per_axis=args.normalize_std_per_axis,
        random_subsample=True)
    te_dataset = ModelNet10PointClouds(
        split='test',
        tr_sample_size=args.tr_max_sample_points,
        te_sample_size=args.te_max_sample_points,
        root_dir=(args.data_dir if data_dir is None else data_dir),
        normalize_per_shape=args.normalize_per_shape,
        normalize_std_per_axis=args.normalize_std_per_axis,
        all_points_mean=tr_dataset.all_points_mean,
        all_points_std=tr_dataset.all_points_std,
    )
    return tr_dataset, te_dataset


def get_datasets(args):
    if args.dataset_type == 'shapenet15k':
        tr_dataset = ShapeNet15kPointClouds(
            categories=args.cates, split='train',
            tr_sample_size=args.tr_max_sample_points,
            te_sample_size=args.te_max_sample_points,
            scale=args.dataset_scale, root_dir=args.data_dir,
            normalize_per_shape=args.normalize_per_shape,
            normalize_std_per_axis=args.normalize_std_per_axis,
            random_subsample=True)
        te_dataset = ShapeNet15kPointClouds(
            categories=args.cates, split='val',
            tr_sample_size=args.tr_max_sample_points,
            te_sample_size=args.te_max_sample_points,
            scale=args.dataset_scale, root_dir=args.data_dir,
            normalize_per_shape=args.normalize_per_shape,
            normalize_std_per_axis=args.normalize_std_per_axis,
            all_points_mean=tr_dataset.all_points_mean,
            all_points_std=tr_dataset.all_points_std,
        )
    elif args.dataset_type == 'modelnet40_15k':
        tr_dataset, te_dataset = _get_MN40_datasets_(args)
    elif args.dataset_type == 'modelnet10_15k':
        tr_dataset, te_dataset = _get_MN10_datasets_(args)
    elif args.dataset_type == "BioLipPocketPointCloud":
        tr_dataset = PocketPointCloudReconstruction(device=args.gpu, 
                                                    processed_data_dir="data/processed_biolip_data/train",
                                                    complex_names_file=None,
                                                    mol_dir=None,
                                                    pocket_dir=None)
        te_dataset = PocketPointCloudReconstruction(device=args.gpu, 
                                                    processed_data_dir="data/processed_biolip_data/val",
                                                    complex_names_file=None,
                                                    mol_dir=None,
                                                    pocket_dir=None)
    else:
        raise Exception("Invalid dataset type:%s" % args.dataset_type)

    return tr_dataset, te_dataset


def get_clf_datasets(args):
    return {
        'MN40': _get_MN40_datasets_(args, data_dir=args.mn40_data_dir),
        'MN10': _get_MN10_datasets_(args, data_dir=args.mn10_data_dir),
    }

def graph_collate_revised4inverse_pocket(batch):
    random_rotation = True
    point_clouds, pockets_coords, lig_graphs, ligs_coords, complex_names = map(list, zip(*batch))
    point_clouds = torch.stack(point_clouds, dim=0)
    
    if random_rotation:
        point_clouds[:, :, :3], rot, theta = apply_random_rotation(point_clouds[:, :, :3], rot_axis=1)
        num_ligs = len(lig_graphs)
        for ii in range(num_ligs):
            lig_graphs[ii].ndata['x'] = torch.mm(lig_graphs[ii].ndata['x'], rot[ii])
    
    return point_clouds, pockets_coords, dgl.batch(lig_graphs), ligs_coords, complex_names

def get_data_loaders(args):
    tr_dataset, te_dataset = get_datasets(args)
    train_loader = data.DataLoader(
        dataset=tr_dataset, batch_size=args.batch_size,
        shuffle=True, num_workers=args.num_workers, drop_last=True,
        worker_init_fn=init_np_seed)
    train_unshuffle_loader = data.DataLoader(
        dataset=tr_dataset, batch_size=args.batch_size,
        shuffle=False, num_workers=args.num_workers, drop_last=True,
        worker_init_fn=init_np_seed)
    test_loader = data.DataLoader(
        dataset=te_dataset, batch_size=args.batch_size,
        shuffle=False, num_workers=args.num_workers, drop_last=False,
        worker_init_fn=init_np_seed)

    loaders = {
        "test_loader": test_loader,
        'train_loader': train_loader,
        'train_unshuffle_loader': train_unshuffle_loader,
    }
    return loaders


def write_strings_to_txt(strings: list, path):
    # every string of the list will be saved in one line
    textfile = open(path, "w")
    for element in strings:
        textfile.write(element + "\n")
    textfile.close()

def read_strings_from_txt(path):
    # every line will be one element of the returned list
    with open(path) as file:
        lines = file.readlines()
        return [line.rstrip() for line in lines]

class PocketPointCloudReconstruction(Dataset):
    def __init__(self, device, 
                 processed_data_dir, 
                 complex_names_file, 
                 mol_dir, 
                 pocket_dir, 
                 prefix = None,
                 ignore_hydrogen = True, 
                 num_points = 7680, # 64 * 120
                 scale_vdw_radius = 1.5, 
                 element_dict = {'C': 0, 'N': 1, 'O': 2, 'S': 3, "P": 4},
                 convert_point_cloud2graph = False, 
                 random_rotation = False,
                 point_cloud_resampling = False,):
        super(PocketPointCloudReconstruction, self).__init__()
        """Constructor for PocketPointCloudReconstruction dataset.
        """
        self.device = device # cpu or gpu
        self.processed_dataset_dir = processed_data_dir # Used for output dir
        if prefix is not None:
            self.processed_dataset_dir = os.path.join(self.processed_dataset_dir, prefix)
        #self.complex_names_file = complex_names_file # List of complex names, only the names in this list will be processed
        #self.mol_dir = mol_dir # Path to the directory containing the mol2 files
        #self.pocket_dir = pocket_dir # Path to the directory containing the pocket files
        self.ignore_hydrogen = ignore_hydrogen # Whether to ignore hydrogen atoms
        self.num_points = num_points # Num of points in sampling for a pocket totally.
        self.scale_vdw_radius = scale_vdw_radius # Define the radius of the sphere for spherical sampling. The sphere radius is defined as scale_vdw_radius * vdw_radius]
        self.element_dict = element_dict # A dictionary containing the element name and its corresponding index. If None, the element_dict in generate_point_cloud will be used. {"C":0, "N":1, "O":2, "S":3, "P":4, "F":5, "Cl":6, "Br":7, "I":8}
        self.dim_features = len(element_dict.keys()) # Features dimension. If None, the features dimension will be the number of element.
        self.convert_point_cloud2graph = convert_point_cloud2graph # Whether to convert the point cloud to graph.
        
        # Define the saved file names
        self.path2lig_graphs_pt = os.path.join(self.processed_dataset_dir, 'lig_graphs.pt')
        self.path2pocket_point_clouds_pt = os.path.join(self.processed_dataset_dir, 'pocket_point_clouds.pt')
        self.path2pocket_coords_pt = os.path.join(self.processed_dataset_dir, 'pocket_coords.pt')
        self.path2valid_complex_names = os.path.join(self.processed_dataset_dir, 'valid_complex_names.txt')
        
        # Load complex names from the file        
        self.complex_names = None

        self.lig_graphs = None
        self.pocket_point_clouds = None
        self.pocket_coords = None
        self.valid_complex_names = []
        
        self.gravity_axis = 1
        self.display_axis_order = [0, 2, 1]
        
        if not os.path.exists(self.processed_dataset_dir):
            os.makedirs(self.processed_dataset_dir)
            self.process()
        else:
            if not os.path.exists(self.path2lig_graphs_pt):
                self.process()
            if not os.path.exists(self.path2pocket_point_clouds_pt):
                self.process()
            if not os.path.exists(self.path2pocket_coords_pt):
                self.process()
            if not os.path.exists(self.path2valid_complex_names):
                self.process()
                
        # TODO Implement the support for point cloud resampling.
                
        temp = torch.load(self.path2pocket_coords_pt)
        self.pockets_coords = temp['pockets_coords']
        
        temp = torch.load(self.path2pocket_point_clouds_pt)
        self.pocket_point_clouds_coords = temp['pocket_point_clouds_coords']
        self.pocket_point_clouds_feats = temp['pocket_point_clouds_feats']
        
        temp, _  = dgl.load_graphs(self.path2lig_graphs_pt)
        self.lig_graphs = temp
        
        self.valid_complex_names = read_strings_from_txt(self.path2valid_complex_names)
        
    
        # Stack all of the data into a single tensor.
        self.pocket_point_clouds_coords = torch.stack(self.pocket_point_clouds_coords, dim=0)
    
    def __len__(self):
        return len(self.lig_graphs)
    
    def __getitem__(self, index):
        pocket_coords = self.pockets_coords[index]
        lig_graph = deepcopy(self.lig_graphs[index])
        lig_coords = lig_graph.ndata['x']
        point_clouds = self.pocket_point_clouds_coords[index]
        
        complex_name = self.valid_complex_names[index]
        return point_clouds.to(self.device), pocket_coords, lig_graph.to(self.device), lig_coords.to(self.device), complex_name
    


if __name__ == "__main__":
    shape_ds = ShapeNet15kPointClouds(categories=['airplane'], split='val')
    x_tr, x_te = next(iter(shape_ds))
    print(x_tr.shape)
    print(x_te.shape)

