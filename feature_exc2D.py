from rdkit import Chem
from rdkit.Chem import AllChem
import numpy as np
import os
import pandas as pd

class AtomFeatureExtractor:
    def __init__(self, num_atoms=0,
                 use_degree=True, use_hybridization=True, use_implicit_valence=True, use_partial_charge=False,
                 use_formal_charge=True, use_ring_size=True, use_hydrogen_bonding=True, use_acid_base=True,
                 use_aromaticity=True, use_chirality=True, use_num_hydrogen=True, use_atom_symbol=True):
        self.num_atoms = num_atoms
        self.use_atom_symbol = use_atom_symbol
        self.use_degree = use_degree
        self.use_hybridization = use_hybridization
        self.use_implicit_valence = use_implicit_valence
        self.use_partial_charge = use_partial_charge
        self.use_formal_charge = use_formal_charge
        self.use_ring_size = use_ring_size
        self.use_hydrogen_bonding = use_hydrogen_bonding
        self.use_acid_base = use_acid_base
        self.use_aromaticity = use_aromaticity
        self.use_chirality = use_chirality
        self.use_num_hydrogen = use_num_hydrogen

        # Define SMARTS patterns
        self.hydrogen_donor = Chem.MolFromSmarts("[$([N;!H0;v3,v4&+1]),$([O,S;H1;+0]),n&H1&+0]")
        self.hydrogen_acceptor = Chem.MolFromSmarts(
            "[$([O,S;H1;v2;!$(*-*=[O,N,P,S])]),$([O,S;H0;v2]),$([O,S;-]),$([N;v3;!$(N-*=[O,N,P,S])]),n&H0&+0,$([o,s;+0;!$([o,s]:n);!$([o,s]:c:n)])]")
        self.acidic = Chem.MolFromSmarts("[$([C,S](=[O,S,P])-[O;H1,-1])]")
        self.basic = Chem.MolFromSmarts(
            "[#7;+,$([N;H2&+0][$([C,a]);!$([C,a](=O))]),$([N;H1&+0]([$([C,a]);!$([C,a](=O))])[$([C,a]);!$([C,a](=O))]),$([N;H0&+0]([C;!$(C(=O))])([C;!$(C(=O))])[C;!$(C(=O))])]")

    def get_atom_features(self, mol):
        AllChem.ComputeGasteigerCharges(mol)
        Chem.AssignStereochemistry(mol)
        hydrogen_donor_match = sum(mol.GetSubstructMatches(self.hydrogen_donor), ())
        hydrogen_acceptor_match = sum(mol.GetSubstructMatches(self.hydrogen_acceptor), ())
        acidic_match = sum(mol.GetSubstructMatches(self.acidic), ())
        basic_match = sum(mol.GetSubstructMatches(self.basic), ())
        ring = mol.GetRingInfo()
        m = []
        for atom_idx in range(mol.GetNumAtoms()):
            atom = mol.GetAtomWithIdx(atom_idx)
            o = []
            o += one_hot(atom.GetSymbol(), ['C', 'O', 'N', 'S', 'Cl', 'F', 'Br', 'P',
                                            'I', 'Si', 'B', 'Na', 'Sn', 'Se', 'other']) if self.use_atom_symbol else []
            o += one_hot(atom.GetDegree(), [0, 1, 2, 3, 4, 5, 6]) if self.use_degree else []
            o += one_hot(atom.GetHybridization(), [Chem.rdchem.HybridizationType.SP,
                                                   Chem.rdchem.HybridizationType.SP2,
                                                   Chem.rdchem.HybridizationType.SP3,
                                                   Chem.rdchem.HybridizationType.SP3D,
                                                   Chem.rdchem.HybridizationType.SP3D2]) if self.use_hybridization else []
            o += one_hot(atom.GetImplicitValence(), [0, 1, 2, 3, 4, 5, 6]) if self.use_implicit_valence else []
            o += one_hot(atom.GetFormalCharge(), [-3, -2, -1, 0, 1, 2, 3]) if self.use_formal_charge else []
            o += [atom.GetIsAromatic()] if self.use_aromaticity else []
            o += [ring.IsAtomInRingOfSize(atom_idx, 3),
                  ring.IsAtomInRingOfSize(atom_idx, 4),
                  ring.IsAtomInRingOfSize(atom_idx, 5),
                  ring.IsAtomInRingOfSize(atom_idx, 6),
                  ring.IsAtomInRingOfSize(atom_idx, 7),
                  ring.IsAtomInRingOfSize(atom_idx, 8)] if self.use_ring_size else []
            o += one_hot(atom.GetTotalNumHs(), [0, 1, 2, 3, 4]) if self.use_num_hydrogen else []
            if self.use_chirality:
                try:
                    o += one_hot(atom.GetProp('_CIPCode'), ["R", "S"]) + [atom.HasProp("_ChiralityPossible")]
                except:
                    o += [False, False] + [atom.HasProp("_ChiralityPossible")]
            if self.use_hydrogen_bonding:
                o += [atom_idx in hydrogen_donor_match]
                o += [atom_idx in hydrogen_acceptor_match]
            if self.use_acid_base:
                o += [atom_idx in acidic_match]
                o += [atom_idx in basic_match]
            m.append(o)
        return np.array(m, dtype=float)

    def get_adjacency_matrix(self, mol):
        return Chem.GetAdjacencyMatrix(mol)

def one_hot(x, allowable_set):
    if x not in allowable_set:
        x = allowable_set[-1]
    return list(map(lambda s: x == s, allowable_set))

# 文件路径
input_excel = "data/dataset.xlsx"
output_dir_features = "data/feature/2D"
output_dir_adj = "data/feature/2D"

# 创建保存目录
os.makedirs(output_dir_features, exist_ok=True)
os.makedirs(output_dir_adj, exist_ok=True)

# 读取 SMILES 数据
df = pd.read_excel(input_excel, sheet_name="Sheet1")
smiles_list = df["SMILES"]

# 初始化特征提取器
extractor = AtomFeatureExtractor()

# 处理每个 SMILES 并保存特征和邻接矩阵
for i, smiles in enumerate(smiles_list):
    try:
        
        mol = Chem.MolFromSmiles(smiles)
        # 为分子添加氢原子
        mol = Chem.AddHs(mol)
        if mol is None:
            print(f"Invalid SMILES at index {i}: {smiles}")
            continue

        # 提取原子特征矩阵
        atom_features = extractor.get_atom_features(mol)
        output_path_features = os.path.join(output_dir_features, f"atom_features_molecule_{i}.npy")
        np.save(output_path_features, atom_features)

        # 提取邻接矩阵
        adjacency_matrix = extractor.get_adjacency_matrix(mol)
        output_path_adj = os.path.join(output_dir_adj, f"adjacency_matrix_molecule_{i}.npy")
        np.save(output_path_adj, adjacency_matrix)
        print(atom_features.shape)
        print(adjacency_matrix.shape)
        print(f"Saved features and adjacency matrix for molecule {i}.")

    except Exception as e:
        print(f"Error processing molecule {i} ({smiles}): {e}")
