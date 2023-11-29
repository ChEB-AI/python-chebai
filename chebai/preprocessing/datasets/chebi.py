__all__ = [
    "JCIData",
    "JCIExtendedTokenData",
    "JCIExtendedBPEData",
    "JCIExtSelfies",
    "JCITokenData",
    "ChEBIOver100",
    "JCI_500_COLUMNS",
    "JCI_500_COLUMNS_INT",
]

from abc import ABC
from collections import OrderedDict
import os
import pickle
import random

from iterstrat.ml_stratifiers import MultilabelStratifiedShuffleSplit

import fastobo
import networkx as nx
import pandas as pd
import requests
import torch

from chebai.preprocessing import reader as dr
from chebai.preprocessing.datasets.base import XYBaseDataModule


class JCIBase(XYBaseDataModule):
    LABEL_INDEX = 2
    SMILES_INDEX = 1

    @property
    def _name(self):
        return "JCI"

    @property
    def processed_file_names(self):
        return ["test.pt", "train.pt", "validation.pt"]

    def download(self):
        pass

    @property
    def raw_file_names(self):
        return ["test.pkl", "train.pkl", "validation.pkl"]

    def prepare_data(self, *args, **kwargs):
        print("Check for raw data in", self.raw_dir)
        if any(
                not os.path.isfile(os.path.join(self.raw_dir, f))
                for f in self.raw_file_names
        ):
            raise ValueError("Raw data is missing")

    @staticmethod
    def _load_tuples(input_file_path):
        with open(input_file_path, "rb") as input_file:
            for row in pickle.load(input_file).values:
                yield row[1], row[2:].astype(bool), row[0]

    @staticmethod
    def _get_data_size(input_file_path):
        with open(input_file_path, "rb") as f:
            return len(pickle.load(f))

    def setup_processed(self):
        print("Transform splits")
        os.makedirs(self.processed_dir, exist_ok=True)
        for k in ["test", "train", "validation"]:
            print("transform", k)
            torch.save(
                self._load_data_from_file(os.path.join(self.raw_dir, f"{k}.pkl")),
                os.path.join(self.processed_dir, f"{k}.pt"),
            )

    @property
    def label_number(self):
        return 500


class JCIData(JCIBase):
    READER = dr.OrdReader


class JCISelfies(JCIBase):
    READER = dr.SelfiesReader


class JCITokenData(JCIBase):
    READER = dr.ChemDataReader


def extract_class_hierarchy(chebi_path):
    with open(chebi_path, encoding='utf-8') as chebi:  # encoding for windows users
        chebi = "\n".join(l for l in chebi if not l.startswith("xref:"))
    elements = [
        term_callback(clause)
        for clause in fastobo.loads(chebi)
        if clause and ":" in str(clause.id)
    ]
    g = nx.DiGraph()
    for n in elements:
        g.add_node(n["id"], **n)
    g.add_edges_from([(p, q["id"]) for q in elements for p in q["parents"]])
    print("Compute transitive closure")
    return nx.transitive_closure_dag(g)


class _ChEBIDataExtractor(XYBaseDataModule, ABC):

    def __init__(self, chebi_version_train: int = None, **kwargs):
        super(_ChEBIDataExtractor, self).__init__(**kwargs)
        # use different version of chebi for training and validation (if not None) - still use self.chebi_version for test set
        self.chebi_version_train = chebi_version_train

    def select_classes(self, g, split_name, *args, **kwargs):
        raise NotImplementedError

    def graph_to_raw_dataset(self, g, split_name=None):
        """Preparation step before creating splits, uses graph created by extract_class_hierarchy()
        split_name is only relevant, if a separate train_version is set"""
        smiles = nx.get_node_attributes(g, "smiles")
        names = nx.get_node_attributes(g, "name")

        print("build labels")
        print(f"Process graph")

        molecules, smiles_list = zip(
            *(
                (n, smiles)
                for n, smiles in ((n, smiles.get(n)) for n in smiles.keys())
                if smiles
            )
        )
        data = OrderedDict(id=molecules)
        data["name"] = [names.get(node) for node in molecules]
        data["SMILES"] = smiles_list
        for n in self.select_classes(g, split_name):
            data[n] = [
                ((n in g.predecessors(node)) or (n == node)) for node in molecules
            ]

        data = pd.DataFrame(data)
        data = data[~data["SMILES"].isnull()]
        data = data[data.iloc[:, 3:].any(axis=1)]
        return data

    def save(self, data: pd.DataFrame, split_name: str):

        pickle.dump(data, open(os.path.join(self.raw_dir, split_name), "wb"))

    @staticmethod
    def _load_dict(input_file_path):
        with open(input_file_path, "rb") as input_file:
            for row in pickle.load(input_file).values:
                yield dict(features=row[2], labels=row[3:].astype(bool), ident=row[0])

    @staticmethod
    def _get_data_size(input_file_path):
        with open(input_file_path, "rb") as f:
            return len(pickle.load(f))

    def _setup_pruned_test_set(self):
        """Create test set with same leaf nodes, but use classes that appear in train set"""
        # TODO: find a more efficient way to do this
        filename_old = 'classes.txt'
        filename_new = f'classes_v{self.chebi_version_train}.txt'
        dataset = torch.load(os.path.join(self.processed_dir, 'test.pt'))
        with open(os.path.join(self.raw_dir, filename_old), "r") as file:
            orig_classes = file.readlines()
        with open(os.path.join(self.raw_dir, filename_new), "r") as file:
            new_classes = file.readlines()
        mapping = [None if or_class not in new_classes else new_classes.index(or_class) for or_class in
                   orig_classes]
        for row in dataset:
            new_labels = [False for _ in new_classes]
            for ind, label in enumerate(row['labels']):
                if mapping[ind] is not None and label:
                    new_labels[mapping[ind]] = label
            row['labels'] = new_labels
        torch.save(dataset, os.path.join(self.processed_dir, self.processed_file_names_dict['test']))

    def setup_processed(self):
        print("Transform splits")
        os.makedirs(self.processed_dir, exist_ok=True)
        for k in self.processed_file_names_dict.keys():
            processed_name = 'test.pt' if k == 'test' else self.processed_file_names_dict[k]
            if not os.path.isfile(os.path.join(self.processed_dir, processed_name)):
                print("transform", k)
                # create two test sets: one with the classes from the original test set, one with only classes used in train
                torch.save(
                    self._load_data_from_file(os.path.join(self.raw_dir, self.raw_file_names_dict[k])),
                    os.path.join(self.processed_dir, processed_name),
                )
        if self.chebi_version_train is not None:
            print("transform test (select classes)")
            self._setup_pruned_test_set()
        self.reader.save_token_cache()

    def get_splits(self, df: pd.DataFrame):
        print("Split dataset")

        df_list = df.values.tolist()
        df_list = [row[3:] for row in df_list]

        msss = MultilabelStratifiedShuffleSplit(n_splits=1, test_size=1-self.train_split, random_state=0)

        train_split = []
        test_split = []
        for (train_split, test_split) in msss.split(
            df_list, df_list,
        ):
            train_split = train_split
            test_split = test_split
            break
        df_train = df.iloc[train_split]
        df_test = df.iloc[test_split]
        if self.use_inner_cross_validation:
            return df_train, df_test

        df_test_list = df_test.values.tolist()
        df_test_list = [row[3:] for row in df_test_list]
        validation_split = []
        test_split = []
        msss = MultilabelStratifiedShuffleSplit(n_splits=1, test_size=1-self.train_split, random_state=0)
        for (test_split, validation_split) in msss.split(
                df_test_list, df_test_list
        ):
            test_split = test_split
            validation_split = validation_split
            break

        df_validation = df_test.iloc[validation_split]
        df_test = df_test.iloc[test_split]
        return df_train, df_test, df_validation

    def get_splits_given_test(self, df: pd.DataFrame, test_df: pd.DataFrame):
        """ Use test set from another chebi version the model does not train on, avoid overlap"""
        print(f"Split dataset for chebi_v{self.chebi_version_train}")
        df_trainval = df
        test_smiles = test_df['SMILES'].tolist()
        mask = []
        for row in df_trainval:
            if row['SMILES'] in test_smiles:
                mask.append(False)
            else:
                mask.append(True)
        df_trainval = df_trainval[mask]

        if self.use_inner_cross_validation:
            return df_trainval

        # assume that size of validation split should relate to train split as in get_splits()
        msss = MultilabelStratifiedShuffleSplit(n_splits=1, test_size=self.train_split ** 2, random_state=0)

        df_trainval_list = df_trainval.tolist()
        df_trainval_list = [row[3:] for row in df_trainval_list]
        train_split = []
        validation_split = []
        for (train_split, validation_split) in msss.split(
            df_trainval_list, df_trainval_list
        ):
            train_split = train_split
            validation_split = validation_split

        df_validation = df_trainval.iloc[validation_split]
        df_train = df_trainval.iloc[train_split]
        return df_train, df_validation

    @property
    def processed_dir(self):
        return os.path.join("data", self._name, f'chebi_v{self.chebi_version}', "processed", *self.identifier)

    @property
    def raw_dir(self):
        return os.path.join("data", self._name, f'chebi_v{self.chebi_version}', "raw")

    @property
    def processed_file_names_dict(self) -> dict:
        train_v_str = f'_v{self.chebi_version_train}' if self.chebi_version_train else ''
        res = {'test': f"test{train_v_str}.pt"}
        if self.use_inner_cross_validation:
            res['train_val'] = f'trainval{train_v_str}.pt'  # for cv, split train/val on runtime
        else:
            res['train'] = f"train{train_v_str}.pt"
            res['validation'] = f"validation{train_v_str}.pt"
        return res

    @property
    def raw_file_names_dict(self) -> dict:
        train_v_str = f'_v{self.chebi_version_train}' if self.chebi_version_train else ''
        res = {'test': f"test.pkl"}  # no extra raw test version for chebi_version_train - use default test set and only
        # adapt processed file
        if self.use_inner_cross_validation:
            res['train_val'] = f'trainval{train_v_str}.pkl'  # for cv, split train/val on runtime
        else:
            res['train'] = f"train{train_v_str}.pkl"
            res['validation'] = f"validation{train_v_str}.pkl"

        return res

    @property
    def processed_file_names(self):
        return list(self.processed_file_names_dict.values())

    @property
    def raw_file_names(self):
        return list(self.raw_file_names_dict.values())

    def prepare_data(self, *args, **kwargs):
        print("Check for raw data in", self.raw_dir)
        if any(
                not os.path.isfile(os.path.join(self.raw_dir, f))
                for f in self.raw_file_names
        ):
            os.makedirs(self.raw_dir, exist_ok=True)
            print("Missing raw data. Go fetch...")
            if self.chebi_version_train is None:
                # load chebi_v{chebi_version}, create splits
                chebi_path = os.path.join(self.raw_dir, f"chebi.obo")
                if not os.path.isfile(chebi_path):
                    print("Load ChEBI ontology")
                    url = f"http://purl.obolibrary.org/obo/chebi/{self.chebi_version}/chebi.obo"
                    r = requests.get(url, allow_redirects=True)
                    open(chebi_path, "wb").write(r.content)
                g = extract_class_hierarchy(chebi_path)
                splits = {}
                full_data = self.graph_to_raw_dataset(g)
                if self.use_inner_cross_validation:
                    splits['train_val'], splits['test'] = self.get_splits(full_data)
                else:
                    splits['train'], splits['test'], splits['validation'] = self.get_splits(full_data)
                for label, split in splits.items():
                    self.save(split, self.raw_file_names_dict[label])
            else:
                # missing test set -> create
                if not os.path.isfile(os.path.join(self.raw_dir, self.raw_file_names_dict['test'])):
                    chebi_path = os.path.join(self.raw_dir, f"chebi.obo")
                    if not os.path.isfile(chebi_path):
                        print("Load ChEBI ontology")
                        url = f"http://purl.obolibrary.org/obo/chebi/{self.chebi_version}/chebi.obo"
                        r = requests.get(url, allow_redirects=True)
                        open(chebi_path, "wb").write(r.content)
                    g = extract_class_hierarchy(chebi_path)
                    df = self.graph_to_raw_dataset(g, self.raw_file_names_dict['test'])
                    _, test_split, _ = self.get_splits(df)
                    self.save(df, self.raw_file_names_dict['test'])
                else:
                    # load test_split from file
                    with open(os.path.join(self.raw_dir, self.raw_file_names_dict['test']), "rb") as input_file:
                        test_split = [row[0] for row in pickle.load(input_file).values]
                chebi_path = os.path.join(self.raw_dir, f"chebi_v{self.chebi_version_train}.obo")
                if not os.path.isfile(chebi_path):
                    print(f"Load ChEBI ontology (v_{self.chebi_version_train})")
                    url = f"http://purl.obolibrary.org/obo/chebi/{self.chebi_version_train}/chebi.obo"
                    r = requests.get(url, allow_redirects=True)
                    open(chebi_path, "wb").write(r.content)
                g = extract_class_hierarchy(chebi_path)
                if self.use_inner_cross_validation:
                    df = self.graph_to_raw_dataset(g, self.raw_file_names_dict['train_val'])
                    train_val_df = self.get_splits_given_test(df, test_split)
                    self.save(train_val_df, self.raw_file_names_dict['train_val'])
                else:
                    df = self.graph_to_raw_dataset(g, self.raw_file_names_dict['train'])
                    train_split, val_split = self.get_splits_given_test(df, test_split)
                    self.save(train_split, self.raw_file_names_dict['train'])
                    self.save(val_split, self.raw_file_names_dict['validation'])


class JCIExtendedBase(_ChEBIDataExtractor):
    LABEL_INDEX = 3
    SMILES_INDEX = 2

    @property
    def label_number(self):
        return 500

    @property
    def _name(self):
        return "JCI_extended"

    def select_classes(self, g, *args, **kwargs):
        return JCI_500_COLUMNS_INT


class ChEBIOverX(_ChEBIDataExtractor):
    LABEL_INDEX = 3
    SMILES_INDEX = 2
    READER = dr.ChemDataReader
    THRESHOLD = None

    @property
    def label_number(self):
        return 854

    @property
    def _name(self):
        return f"ChEBI{self.THRESHOLD}"

    def select_classes(self, g, split_name, *args, **kwargs):
        smiles = nx.get_node_attributes(g, "smiles")
        nodes = list(
            sorted(
                {
                    node
                    for node in g.nodes
                    if sum(
                    1 if smiles[s] is not None else 0 for s in g.successors(node)
                )
                       >= self.THRESHOLD
                }
            )
        )
        filename = 'classes.txt'
        if self.chebi_version_train is not None and self.raw_file_names_dict['test'] != split_name:
            filename = f'classes_v{self.chebi_version_train}.txt'
        with open(os.path.join(self.raw_dir, filename), "wt") as fout:
            fout.writelines(str(node) + "\n" for node in nodes)
        return nodes


class ChEBIOverXDeepSMILES(ChEBIOverX):
    READER = dr.DeepChemDataReader


class ChEBIOverXSELFIES(ChEBIOverX):
    READER = dr.SelfiesReader


class ChEBIOver100(ChEBIOverX):
    THRESHOLD = 100

    def label_number(self):
        return 854


class ChEBIOver50(ChEBIOverX):
    THRESHOLD = 50

    def label_number(self):
        return 1332


class ChEBIOver100DeepSMILES(ChEBIOverXDeepSMILES, ChEBIOver100):
    pass


class ChEBIOver100SELFIES(ChEBIOverXSELFIES, ChEBIOver100):
    pass


class JCIExtendedBPEData(JCIExtendedBase):
    READER = dr.ChemBPEReader


class JCIExtendedTokenData(JCIExtendedBase):
    READER = dr.ChemDataReader


class JCIExtSelfies(JCIExtendedBase):
    READER = dr.SelfiesReader


def chebi_to_int(s):
    return int(s[s.index(":") + 1:])


def term_callback(doc):
    parts = set()
    parents = []
    name = None
    smiles = None
    for clause in doc:
        if isinstance(clause, fastobo.term.PropertyValueClause):
            t = clause.property_value
            if str(t.relation) == "http://purl.obolibrary.org/obo/chebi/smiles":
                assert smiles is None
                smiles = t.value
        # in older chebi versions, smiles strings are synonyms
        # e.g. synonym: "[F-].[Na+]" RELATED SMILES [ChEBI]
        elif isinstance(clause, fastobo.term.SynonymClause):
            if "SMILES" in clause.raw_value():
                assert smiles is None
                smiles = clause.raw_value().split('"')[1]
        elif isinstance(clause, fastobo.term.RelationshipClause):
            if str(clause.typedef) == "has_part":
                parts.add(chebi_to_int(str(clause.term)))
        elif isinstance(clause, fastobo.term.IsAClause):
            parents.append(chebi_to_int(str(clause.term)))
        elif isinstance(clause, fastobo.term.NameClause):
            name = str(clause.name)
    return {
        "id": chebi_to_int(str(doc.id)),
        "parents": parents,
        "has_part": parts,
        "name": name,
        "smiles": smiles,
    }


atom_index = (
    "\*",
    "H",
    "He",
    "Li",
    "Be",
    "B",
    "C",
    "N",
    "O",
    "F",
    "Ne",
    "Na",
    "Mg",
    "Al",
    "Si",
    "P",
    "S",
    "Cl",
    "Ar",
    "K",
    "Ca",
    "Sc",
    "Ti",
    "V",
    "Cr",
    "Mn",
    "Fe",
    "Co",
    "Ni",
    "Cu",
    "Zn",
    "Ga",
    "Ge",
    "As",
    "Se",
    "Br",
    "Kr",
    "Rb",
    "Sr",
    "Y",
    "Zr",
    "Nb",
    "Mo",
    "Tc",
    "Ru",
    "Rh",
    "Pd",
    "Ag",
    "Cd",
    "In",
    "Sn",
    "Sb",
    "Te",
    "I",
    "Xe",
    "Cs",
    "Ba",
    "La",
    "Hf",
    "Ta",
    "W",
    "Re",
    "Os",
    "Ir",
    "Pt",
    "Au",
    "Hg",
    "Tl",
    "Pb",
    "Bi",
    "Po",
    "At",
    "Rn",
    "Fr",
    "Ra",
    "Ac",
    "Rf",
    "Db",
    "Sg",
    "Bh",
    "Hs",
    "Mt",
    "Ds",
    "Rg",
    "Cn",
    "Nh",
    "Fl",
    "Mc",
    "Lv",
    "Ts",
    "Og",
    "Ce",
    "Pr",
    "Nd",
    "Pm",
    "Sm",
    "Eu",
    "Gd",
    "Tb",
    "Dy",
    "Ho",
    "Er",
    "Tm",
    "Yb",
    "Lu",
    "Th",
    "Pa",
    "U",
    "Np",
    "Pu",
    "Am",
    "Cm",
    "Bk",
    "Cf",
    "Es",
    "Fm",
    "Md",
    "No",
    "Lr",
    "c",
    "n",
    "s",
    "o",
    "se",
    "p",
)

JCI_500_COLUMNS = [
    "CHEBI:25716",
    "CHEBI:72010",
    "CHEBI:60926",
    "CHEBI:39206",
    "CHEBI:24315",
    "CHEBI:22693",
    "CHEBI:23981",
    "CHEBI:23066",
    "CHEBI:35343",
    "CHEBI:18303",
    "CHEBI:60971",
    "CHEBI:35753",
    "CHEBI:24836",
    "CHEBI:59268",
    "CHEBI:35992",
    "CHEBI:51718",
    "CHEBI:27093",
    "CHEBI:38311",
    "CHEBI:46940",
    "CHEBI:26399",
    "CHEBI:27325",
    "CHEBI:33637",
    "CHEBI:37010",
    "CHEBI:36786",
    "CHEBI:59777",
    "CHEBI:36871",
    "CHEBI:26799",
    "CHEBI:50525",
    "CHEBI:26848",
    "CHEBI:52782",
    "CHEBI:75885",
    "CHEBI:37533",
    "CHEBI:47018",
    "CHEBI:27150",
    "CHEBI:26707",
    "CHEBI:131871",
    "CHEBI:134179",
    "CHEBI:24727",
    "CHEBI:59238",
    "CHEBI:26373",
    "CHEBI:46774",
    "CHEBI:33642",
    "CHEBI:38686",
    "CHEBI:74222",
    "CHEBI:23666",
    "CHEBI:46770",
    "CHEBI:16460",
    "CHEBI:37485",
    "CHEBI:21644",
    "CHEBI:52565",
    "CHEBI:33576",
    "CHEBI:76170",
    "CHEBI:46640",
    "CHEBI:61902",
    "CHEBI:22750",
    "CHEBI:35348",
    "CHEBI:48030",
    "CHEBI:2571",
    "CHEBI:38131",
    "CHEBI:83575",
    "CHEBI:136889",
    "CHEBI:26250",
    "CHEBI:36244",
    "CHEBI:23906",
    "CHEBI:38261",
    "CHEBI:22916",
    "CHEBI:35924",
    "CHEBI:24689",
    "CHEBI:32877",
    "CHEBI:50511",
    "CHEBI:26588",
    "CHEBI:24385",
    "CHEBI:5653",
    "CHEBI:48591",
    "CHEBI:38295",
    "CHEBI:58944",
    "CHEBI:134396",
    "CHEBI:49172",
    "CHEBI:26558",
    "CHEBI:64708",
    "CHEBI:35923",
    "CHEBI:25961",
    "CHEBI:47779",
    "CHEBI:46812",
    "CHEBI:37863",
    "CHEBI:22718",
    "CHEBI:36562",
    "CHEBI:38771",
    "CHEBI:36078",
    "CHEBI:26935",
    "CHEBI:33555",
    "CHEBI:23044",
    "CHEBI:15693",
    "CHEBI:33892",
    "CHEBI:33909",
    "CHEBI:35766",
    "CHEBI:51149",
    "CHEBI:35972",
    "CHEBI:38304",
    "CHEBI:46942",
    "CHEBI:24026",
    "CHEBI:33721",
    "CHEBI:38093",
    "CHEBI:38830",
    "CHEBI:26875",
    "CHEBI:37963",
    "CHEBI:61910",
    "CHEBI:47891",
    "CHEBI:74818",
    "CHEBI:50401",
    "CHEBI:24834",
    "CHEBI:33299",
    "CHEBI:63424",
    "CHEBI:63427",
    "CHEBI:15841",
    "CHEBI:33666",
    "CHEBI:26214",
    "CHEBI:22484",
    "CHEBI:27024",
    "CHEBI:46845",
    "CHEBI:64365",
    "CHEBI:63566",
    "CHEBI:38757",
    "CHEBI:83264",
    "CHEBI:24867",
    "CHEBI:37841",
    "CHEBI:33720",
    "CHEBI:36885",
    "CHEBI:59412",
    "CHEBI:64612",
    "CHEBI:36500",
    "CHEBI:37015",
    "CHEBI:84135",
    "CHEBI:51751",
    "CHEBI:18133",
    "CHEBI:57613",
    "CHEBI:38976",
    "CHEBI:25810",
    "CHEBI:24873",
    "CHEBI:35571",
    "CHEBI:83812",
    "CHEBI:37909",
    "CHEBI:51750",
    "CHEBI:15889",
    "CHEBI:48470",
    "CHEBI:24676",
    "CHEBI:22480",
    "CHEBI:139051",
    "CHEBI:23252",
    "CHEBI:51454",
    "CHEBI:88061",
    "CHEBI:46874",
    "CHEBI:38338",
    "CHEBI:62618",
    "CHEBI:59266",
    "CHEBI:84403",
    "CHEBI:27116",
    "CHEBI:77632",
    "CHEBI:38418",
    "CHEBI:35213",
    "CHEBI:35496",
    "CHEBI:78799",
    "CHEBI:38314",
    "CHEBI:35568",
    "CHEBI:35573",
    "CHEBI:33847",
    "CHEBI:16038",
    "CHEBI:33741",
    "CHEBI:33654",
    "CHEBI:17387",
    "CHEBI:33572",
    "CHEBI:36233",
    "CHEBI:22297",
    "CHEBI:23990",
    "CHEBI:38102",
    "CHEBI:24436",
    "CHEBI:35189",
    "CHEBI:79202",
    "CHEBI:68489",
    "CHEBI:18254",
    "CHEBI:78189",
    "CHEBI:47019",
    "CHEBI:61655",
    "CHEBI:24373",
    "CHEBI:26347",
    "CHEBI:36709",
    "CHEBI:73539",
    "CHEBI:35507",
    "CHEBI:35293",
    "CHEBI:140326",
    "CHEBI:46668",
    "CHEBI:17188",
    "CHEBI:61109",
    "CHEBI:35819",
    "CHEBI:33744",
    "CHEBI:73474",
    "CHEBI:134361",
    "CHEBI:33238",
    "CHEBI:26766",
    "CHEBI:17517",
    "CHEBI:25508",
    "CHEBI:22580",
    "CHEBI:26394",
    "CHEBI:35356",
    "CHEBI:50918",
    "CHEBI:24860",
    "CHEBI:2468",
    "CHEBI:33581",
    "CHEBI:26519",
    "CHEBI:37948",
    "CHEBI:33823",
    "CHEBI:59554",
    "CHEBI:46848",
    "CHEBI:24897",
    "CHEBI:26893",
    "CHEBI:63394",
    "CHEBI:29348",
    "CHEBI:35790",
    "CHEBI:25241",
    "CHEBI:58958",
    "CHEBI:24397",
    "CHEBI:25413",
    "CHEBI:24302",
    "CHEBI:46850",
    "CHEBI:51867",
    "CHEBI:35314",
    "CHEBI:50893",
    "CHEBI:36130",
    "CHEBI:33558",
    "CHEBI:24782",
    "CHEBI:36087",
    "CHEBI:26649",
    "CHEBI:47923",
    "CHEBI:33184",
    "CHEBI:23643",
    "CHEBI:25985",
    "CHEBI:33257",
    "CHEBI:61355",
    "CHEBI:24697",
    "CHEBI:36838",
    "CHEBI:23451",
    "CHEBI:33242",
    "CHEBI:26872",
    "CHEBI:50523",
    "CHEBI:16701",
    "CHEBI:36699",
    "CHEBI:35505",
    "CHEBI:24360",
    "CHEBI:59737",
    "CHEBI:26455",
    "CHEBI:51285",
    "CHEBI:35504",
    "CHEBI:36309",
    "CHEBI:33554",
    "CHEBI:47909",
    "CHEBI:50858",
    "CHEBI:53339",
    "CHEBI:25609",
    "CHEBI:23665",
    "CHEBI:35902",
    "CHEBI:35552",
    "CHEBI:139592",
    "CHEBI:35724",
    "CHEBI:38337",
    "CHEBI:35241",
    "CHEBI:29075",
    "CHEBI:62941",
    "CHEBI:140345",
    "CHEBI:59769",
    "CHEBI:28863",
    "CHEBI:47882",
    "CHEBI:35903",
    "CHEBI:33641",
    "CHEBI:47784",
    "CHEBI:23079",
    "CHEBI:25036",
    "CHEBI:50018",
    "CHEBI:28874",
    "CHEBI:35276",
    "CHEBI:26764",
    "CHEBI:65323",
    "CHEBI:51276",
    "CHEBI:37022",
    "CHEBI:22478",
    "CHEBI:23449",
    "CHEBI:72823",
    "CHEBI:63567",
    "CHEBI:50753",
    "CHEBI:38785",
    "CHEBI:46952",
    "CHEBI:36914",
    "CHEBI:33653",
    "CHEBI:62937",
    "CHEBI:36315",
    "CHEBI:37667",
    "CHEBI:38835",
    "CHEBI:35315",
    "CHEBI:33551",
    "CHEBI:18154",
    "CHEBI:79346",
    "CHEBI:26932",
    "CHEBI:39203",
    "CHEBI:25235",
    "CHEBI:23003",
    "CHEBI:64583",
    "CHEBI:46955",
    "CHEBI:33658",
    "CHEBI:59202",
    "CHEBI:28892",
    "CHEBI:33599",
    "CHEBI:33259",
    "CHEBI:64611",
    "CHEBI:37947",
    "CHEBI:65321",
    "CHEBI:63571",
    "CHEBI:25830",
    "CHEBI:50492",
    "CHEBI:26961",
    "CHEBI:33482",
    "CHEBI:63436",
    "CHEBI:47017",
    "CHEBI:51681",
    "CHEBI:48901",
    "CHEBI:52575",
    "CHEBI:35683",
    "CHEBI:24353",
    "CHEBI:61778",
    "CHEBI:13248",
    "CHEBI:35990",
    "CHEBI:33485",
    "CHEBI:35871",
    "CHEBI:27933",
    "CHEBI:27136",
    "CHEBI:26407",
    "CHEBI:33566",
    "CHEBI:47880",
    "CHEBI:24921",
    "CHEBI:38077",
    "CHEBI:48975",
    "CHEBI:59835",
    "CHEBI:83273",
    "CHEBI:22562",
    "CHEBI:33838",
    "CHEBI:35627",
    "CHEBI:51614",
    "CHEBI:36836",
    "CHEBI:63423",
    "CHEBI:22331",
    "CHEBI:25529",
    "CHEBI:36314",
    "CHEBI:83822",
    "CHEBI:38164",
    "CHEBI:51006",
    "CHEBI:28965",
    "CHEBI:38716",
    "CHEBI:76567",
    "CHEBI:35381",
    "CHEBI:51269",
    "CHEBI:37141",
    "CHEBI:25872",
    "CHEBI:36526",
    "CHEBI:51702",
    "CHEBI:25106",
    "CHEBI:51737",
    "CHEBI:38672",
    "CHEBI:36132",
    "CHEBI:38700",
    "CHEBI:25558",
    "CHEBI:17855",
    "CHEBI:18946",
    "CHEBI:83565",
    "CHEBI:15705",
    "CHEBI:35186",
    "CHEBI:33694",
    "CHEBI:36711",
    "CHEBI:23403",
    "CHEBI:35238",
    "CHEBI:36807",
    "CHEBI:47788",
    "CHEBI:24531",
    "CHEBI:33663",
    "CHEBI:22715",
    "CHEBI:57560",
    "CHEBI:38163",
    "CHEBI:23899",
    "CHEBI:50994",
    "CHEBI:26776",
    "CHEBI:51569",
    "CHEBI:35259",
    "CHEBI:77636",
    "CHEBI:35727",
    "CHEBI:35786",
    "CHEBI:24780",
    "CHEBI:26714",
    "CHEBI:26712",
    "CHEBI:26819",
    "CHEBI:63944",
    "CHEBI:36520",
    "CHEBI:25409",
    "CHEBI:22928",
    "CHEBI:23824",
    "CHEBI:79020",
    "CHEBI:26605",
    "CHEBI:139588",
    "CHEBI:52396",
    "CHEBI:37668",
    "CHEBI:50995",
    "CHEBI:52395",
    "CHEBI:61777",
    "CHEBI:38445",
    "CHEBI:24698",
    "CHEBI:63551",
    "CHEBI:35693",
    "CHEBI:83403",
    "CHEBI:36094",
    "CHEBI:35479",
    "CHEBI:25704",
    "CHEBI:25754",
    "CHEBI:38958",
    "CHEBI:21731",
    "CHEBI:23697",
    "CHEBI:38260",
    "CHEBI:33861",
    "CHEBI:22485",
    "CHEBI:2580",
    "CHEBI:18379",
    "CHEBI:23424",
    "CHEBI:33296",
    "CHEBI:37554",
    "CHEBI:33839",
    "CHEBI:36054",
    "CHEBI:23232",
    "CHEBI:18035",
    "CHEBI:63353",
    "CHEBI:23114",
    "CHEBI:76578",
    "CHEBI:26208",
    "CHEBI:32955",
    "CHEBI:24922",
    "CHEBI:36141",
    "CHEBI:24043",
    "CHEBI:35692",
    "CHEBI:46867",
    "CHEBI:38530",
    "CHEBI:24654",
    "CHEBI:38032",
    "CHEBI:26820",
    "CHEBI:35789",
    "CHEBI:62732",
    "CHEBI:26912",
    "CHEBI:22160",
    "CHEBI:26410",
    "CHEBI:36059",
    "CHEBI:51069",
    "CHEBI:33570",
    "CHEBI:24129",
    "CHEBI:37826",
    "CHEBI:16385",
    "CHEBI:26822",
    "CHEBI:46761",
    "CHEBI:83925",
    "CHEBI:25248",
    "CHEBI:37581",
    "CHEBI:35748",
    "CHEBI:26195",
    "CHEBI:33958",
    "CHEBI:58342",
    "CHEBI:17478",
    "CHEBI:36834",
    "CHEBI:25513",
    "CHEBI:57643",
    "CHEBI:38298",
    "CHEBI:64482",
    "CHEBI:33240",
    "CHEBI:47622",
    "CHEBI:33704",
    "CHEBI:83820",
    "CHEBI:33676",
    "CHEBI:32952",
    "CHEBI:131927",
    "CHEBI:26188",
    "CHEBI:35716",
    "CHEBI:28963",
    "CHEBI:22798",
    "CHEBI:60980",
    "CHEBI:17984",
    "CHEBI:37240",
    "CHEBI:28868",
    "CHEBI:27208",
    "CHEBI:15904",
    "CHEBI:35715",
    "CHEBI:22251",
    "CHEBI:61078",
    "CHEBI:61079",
    "CHEBI:58946",
    "CHEBI:37123",
    "CHEBI:33497",
    "CHEBI:50699",
    "CHEBI:22475",
    "CHEBI:35436",
]

JCI_500_COLUMNS_INT = [int(n.split(":")[-1]) for n in JCI_500_COLUMNS]
