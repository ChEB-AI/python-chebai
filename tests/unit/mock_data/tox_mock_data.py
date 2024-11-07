from typing import Dict, List


class Tox21MolNetMockData:
    """
    A utility class providing mock data for testing the Tox21MolNet dataset.

    This class includes static methods that return mock data in various formats, simulating
    the raw and processed data of the Tox21MolNet dataset. The mock data is used for unit tests
    to verify the functionality of methods within the Tox21MolNet class without relying on actual
    data files.
    """

    @staticmethod
    def get_raw_data() -> str:
        """
        Returns a raw CSV string that simulates the raw data of the Tox21MolNet dataset.
        """
        return (
            "NR-AR,NR-AR-LBD,NR-AhR,NR-Aromatase,NR-ER,NR-ER-LBD,NR-PPAR-gamma,SR-ARE,SR-ATAD5,SR-HSE,SR-MMP,SR-p53,"
            "mol_id,smiles\n"
            "0,0,1,0,1,1,0,1,0,,1,0,TOX958,Nc1ccc([N+](=O)[O-])cc1N\n"
            ",,,,,,,,,1,,,TOX31681,Nc1cc(C(F)(F)F)ccc1S\n"
            "0,0,0,0,0,0,0,,0,0,0,0,TOX5110,CC(C)(C)OOC(C)(C)CCC(C)(C)OOC(C)(C)C\n"
            "0,0,0,0,0,0,0,0,0,0,0,0,TOX6619,O=S(=O)(Cl)c1ccccc1\n"
            "0,0,0,,0,0,,,0,,1,,TOX27679,CCCCCc1ccco1\n"
            "0,,1,,,,0,,1,1,1,1,TOX2801,Oc1c(Cl)cc(Cl)c2cccnc12\n"
            "0,0,0,0,,0,,,0,0,,1,TOX2808,CN(C)CCCN1c2ccccc2Sc2ccc(Cl)cc21\n"
            "0,,0,1,,,,1,0,,1,,TOX29085,CCCCCCCCCCCCCCn1cc[n+](C)c1\n"
        )

    @staticmethod
    def get_processed_data() -> List[Dict]:
        """
        Returns a list of dictionaries simulating the processed data for the Tox21MolNet dataset.
        Each dictionary contains 'ident', 'features', and 'labels'.
        """
        data_list = [
            {
                "ident": "TOX958",
                "features": "Nc1ccc([N+](=O)[O-])cc1N",
                "labels": [
                    False,
                    False,
                    True,
                    False,
                    True,
                    True,
                    False,
                    True,
                    False,
                    None,
                    True,
                    False,
                ],
            },
            {
                "ident": "TOX31681",
                "features": "Nc1cc(C(F)(F)F)ccc1S",
                "labels": [
                    None,
                    None,
                    None,
                    None,
                    None,
                    None,
                    None,
                    None,
                    None,
                    True,
                    None,
                    None,
                ],
            },
            {
                "ident": "TOX5110",
                "features": "CC(C)(C)OOC(C)(C)CCC(C)(C)OOC(C)(C)C",
                "labels": [
                    False,
                    False,
                    False,
                    False,
                    False,
                    False,
                    False,
                    None,
                    False,
                    False,
                    False,
                    False,
                ],
            },
            {
                "ident": "TOX6619",
                "features": "O=S(=O)(Cl)c1ccccc1",
                "labels": [
                    False,
                    False,
                    False,
                    False,
                    False,
                    False,
                    False,
                    False,
                    False,
                    False,
                    False,
                    False,
                ],
            },
            {
                "ident": "TOX27679",
                "features": "CCCCCc1ccco1",
                "labels": [
                    False,
                    False,
                    False,
                    None,
                    False,
                    False,
                    None,
                    None,
                    False,
                    None,
                    True,
                    None,
                ],
            },
            {
                "ident": "TOX2801",
                "features": "Oc1c(Cl)cc(Cl)c2cccnc12",
                "labels": [
                    False,
                    None,
                    True,
                    None,
                    None,
                    None,
                    False,
                    None,
                    True,
                    True,
                    True,
                    True,
                ],
            },
            {
                "ident": "TOX2808",
                "features": "CN(C)CCCN1c2ccccc2Sc2ccc(Cl)cc21",
                "labels": [
                    False,
                    False,
                    False,
                    False,
                    None,
                    False,
                    None,
                    None,
                    False,
                    False,
                    None,
                    True,
                ],
            },
            {
                "ident": "TOX29085",
                "features": "CCCCCCCCCCCCCCn1cc[n+](C)c1",
                "labels": [
                    False,
                    None,
                    False,
                    True,
                    None,
                    None,
                    None,
                    True,
                    False,
                    None,
                    True,
                    None,
                ],
            },
        ]

        data_with_group = [{**data, "group": None} for data in data_list]
        return data_with_group

    @staticmethod
    def get_processed_grouped_data() -> List[Dict]:
        """
        Returns a list of dictionaries simulating the processed data for the Tox21MolNet dataset.
        Each dictionary contains 'ident', 'features', and 'labels'.
        """
        processed_data = Tox21MolNetMockData.get_processed_data()
        groups = ["A", "A", "B", "B", "C", "C", "C", "C"]

        assert len(processed_data) == len(
            groups
        ), "The number of processed data entries does not match the number of groups."

        # Combine processed data with their corresponding groups
        grouped_data = [
            {**data, "group": group, "original": True}
            for data, group in zip(processed_data, groups)
        ]

        return grouped_data


class Tox21ChallengeMockData:

    MOL_BINARY_STR = (
        b"cyclobutane\n"
        b"     RDKit          2D\n\n"
        b"  4  4  0  0  0  0  0  0  0  0999 V2000\n"
        b"    1.0607   -0.0000    0.0000 C   0  0  0  0  0  0  0  0  0  0  0  0\n"
        b"   -0.0000   -1.0607    0.0000 C   0  0  0  0  0  0  0  0  0  0  0  0\n"
        b"   -1.0607    0.0000    0.0000 C   0  0  0  0  0  0  0  0  0  0  0  0\n"
        b"    0.0000    1.0607    0.0000 C   0  0  0  0  0  0  0  0  0  0  0  0\n"
        b"  1  2  1  0\n"
        b"  2  3  1  0\n"
        b"  3  4  1  0\n"
        b"  4  1  1  0\n"
        b"M  END\n\n"
    )

    SMILES_OF_MOL = "C1CCC1"
    # Feature encoding of SMILES as per chebai/preprocessing/bin/smiles_token/tokens.txt
    FEATURE_OF_SMILES = [19, 42, 19, 19, 19, 42]

    @staticmethod
    def get_raw_train_data() -> bytes:
        raw_str = (
            Tox21ChallengeMockData.MOL_BINARY_STR + b">  <DSSTox_CID>\n"
            b"25848\n\n"
            b">  <SR-HSE>\n"
            b"0\n\n"
            b"$$$$\n" + Tox21ChallengeMockData.MOL_BINARY_STR + b">  <DSSTox_CID>\n"
            b"2384\n\n"
            b">  <NR-Aromatase>\n"
            b"1\n\n"
            b">  <NR-AR>\n"
            b"0\n\n"
            b"$$$$\n" + Tox21ChallengeMockData.MOL_BINARY_STR + b">  <DSSTox_CID>\n"
            b"27102\n\n"
            b">  <NR-AR>\n"
            b"0\n\n"
            b">  <NR-AhR>\n"
            b"0\n\n"
            b"$$$$\n" + Tox21ChallengeMockData.MOL_BINARY_STR + b">  <DSSTox_CID>\n"
            b"26792\n\n"
            b">  <NR-AR>\n"
            b"1\n\n"
            b">  <NR-AR-LBD>\n"
            b"1\n\n"
            b">  <NR-AhR>\n"
            b"1\n\n"
            b">  <NR-Aromatase>\n"
            b"1\n\n"
            b">  <NR-ER>\n"
            b"1\n\n"
            b">  <NR-ER-LBD>\n"
            b"1\n\n"
            b">  <NR-PPAR-gamma>\n"
            b"1\n\n"
            b">  <SR-ARE>\n"
            b"1\n\n"
            b">  <SR-ATAD5>\n"
            b"1\n\n"
            b">  <SR-HSE>\n"
            b"1\n\n"
            b">  <SR-MMP>\n"
            b"1\n\n"
            b">  <SR-p53>\n"
            b"1\n\n"
            b"$$$$\n" + Tox21ChallengeMockData.MOL_BINARY_STR + b">  <DSSTox_CID>\n"
            b"26401\n\n"
            b">  <SR-ARE>\n"
            b"1\n\n"
            b">  <SR-HSE>\n"
            b"1\n\n"
            b"$$$$\n" + Tox21ChallengeMockData.MOL_BINARY_STR + b">  <DSSTox_CID>\n"
            b"25973\n\n"
            b"$$$$\n"
        )
        return raw_str

    @staticmethod
    def data_in_dict_format() -> List[Dict]:
        data_list = [
            {
                "labels": [
                    None,
                    None,
                    None,
                    None,
                    None,
                    None,
                    None,
                    None,
                    None,
                    0,
                    None,
                    None,
                ],
                "ident": "25848",
            },
            {
                "labels": [
                    0,
                    None,
                    None,
                    1,
                    None,
                    None,
                    None,
                    None,
                    None,
                    None,
                    None,
                    None,
                ],
                "ident": "2384",
            },
            {
                "labels": [
                    0,
                    None,
                    0,
                    None,
                    None,
                    None,
                    None,
                    None,
                    None,
                    None,
                    None,
                    None,
                ],
                "ident": "27102",
            },
            {
                "labels": [
                    1,
                    1,
                    1,
                    1,
                    1,
                    1,
                    1,
                    1,
                    1,
                    1,
                    1,
                    1,
                ],
                "ident": "26792",
            },
            {
                "labels": [
                    None,
                    None,
                    None,
                    None,
                    None,
                    None,
                    None,
                    1,
                    None,
                    1,
                    None,
                    None,
                ],
                "ident": "26401",
            },
            {
                "labels": [
                    None,
                    None,
                    None,
                    None,
                    None,
                    None,
                    None,
                    None,
                    None,
                    None,
                    None,
                    None,
                ],
                "ident": "25973",
            },
        ]

        for dict_ in data_list:
            dict_["features"] = Tox21ChallengeMockData.FEATURE_OF_SMILES
            dict_["group"] = None

        return data_list

    @staticmethod
    def get_raw_smiles_data() -> str:
        """
        Returns mock SMILES data in a tab-delimited format (mocks test.smiles file).

        The data represents molecules and their associated sample IDs.

        Returns:
            str: A string containing SMILES representations and corresponding sample IDs.
        """
        return (
            "#SMILES\tSample ID\n"
            f"{Tox21ChallengeMockData.SMILES_OF_MOL}\tNCGC00260869-01\n"
            f"{Tox21ChallengeMockData.SMILES_OF_MOL}\tNCGC00261776-01\n"
            f"{Tox21ChallengeMockData.SMILES_OF_MOL}\tNCGC00261380-01\n"
            f"{Tox21ChallengeMockData.SMILES_OF_MOL}\tNCGC00261842-01\n"
            f"{Tox21ChallengeMockData.SMILES_OF_MOL}\tNCGC00261662-01\n"
            f"{Tox21ChallengeMockData.SMILES_OF_MOL}\tNCGC00261190-01\n"
        )

    @staticmethod
    def get_raw_score_txt_data() -> str:
        """
        Returns mock score data in a tab-delimited format (mocks test_results.txt file).

        The data represents toxicity test results for different molecular samples, including several toxicity endpoints.

        Returns:
            str: A string containing toxicity scores for each molecular sample and corresponding toxicity endpoints.
        """
        return (
            "Sample ID\tNR-AhR\tNR-AR\tNR-AR-LBD\tNR-Aromatase\tNR-ER\tNR-ER-LBD\tNR-PPAR-gamma\t"
            "SR-ARE\tSR-ATAD5\tSR-HSE\tSR-MMP\tSR-p53\n"
            "NCGC00260869-01\t0\t0\t0\t0\t0\t0\t0\t0\t0\t0\t0\t0\n"
            "NCGC00261776-01\t1\t1\t1\t1\t1\t1\t1\t1\t1\t1\t1\t1\n"
            "NCGC00261380-01\tx\tx\tx\tx\tx\tx\tx\tx\tx\tx\tx\tx\n"
            "NCGC00261842-01\t0\t0\t0\tx\t0\t0\t0\t0\t0\t0\tx\t1\n"
            "NCGC00261662-01\t1\t0\t0\tx\t1\t1\t1\tx\t1\t1\tx\t1\n"
            "NCGC00261190-01\tx\t0\t0\tx\t1\t0\t0\t1\t0\t0\t1\t1\n"
        )

    @staticmethod
    def get_setup_processed_output_data() -> List[Dict]:
        """
        Returns mock processed data used for testing the `setup_processed` method.

        The data contains molecule identifiers and their corresponding toxicity labels for multiple endpoints.
        Each dictionary in the list represents a molecule with its associated labels, features, and group information.

        Returns:
            List[Dict]: A list of dictionaries where each dictionary contains:
                        - "features": The SMILES features of the molecule.
                        - "labels": A list of toxicity endpoint labels (0, 1, or None).
                        - "ident": The sample identifier.
                        - "group": None (default value for the group key).
        """

        # "NR-AR", "NR-AR-LBD", "NR-AhR", "NR-Aromatase", "NR-ER", "NR-ER-LBD", "NR-PPAR-gamma", "SR-ARE", "SR-ATAD5",
        # "SR-HSE", "SR-MMP", "SR-p53",
        data_list = [
            {
                "labels": [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                "ident": "NCGC00260869-01",
            },
            {
                "labels": [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
                "ident": "NCGC00261776-01",
            },
            {
                "labels": [
                    None,
                    None,
                    None,
                    None,
                    None,
                    None,
                    None,
                    None,
                    None,
                    None,
                    None,
                    None,
                ],
                "ident": "NCGC00261380-01",
            },
            {
                "labels": [0, 0, 0, None, 0, 0, 0, 0, 0, 0, None, 1],
                "ident": "NCGC00261842-01",
            },
            {
                "labels": [0, 0, 1, None, 1, 1, 1, None, 1, 1, None, 1],
                "ident": "NCGC00261662-01",
            },
            {
                "labels": [0, 0, None, None, 1, 0, 0, 1, 0, 0, 1, 1],
                "ident": "NCGC00261190-01",
            },
        ]

        complete_list = []
        for dict_ in data_list:
            complete_list.append(
                {
                    "features": Tox21ChallengeMockData.FEATURE_OF_SMILES,
                    **dict_,
                    "group": None,
                }
            )

        return complete_list
