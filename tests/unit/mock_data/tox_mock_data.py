class Tox21MockData:
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
            + "mol_id,smiles\n"
            + "0,0,1,0,1,1,0,1,0,,1,0,TOX958,Nc1ccc([N+](=O)[O-])cc1N\n"
            + ",,,,,,,,,1,,,TOX31681,Nc1cc(C(F)(F)F)ccc1S\n"
            + "0,0,0,0,0,0,0,,0,0,0,0,TOX5110,CC(C)(C)OOC(C)(C)CCC(C)(C)OOC(C)(C)C\n"
            + "0,0,0,0,0,0,0,0,0,0,0,0,TOX6619,O=S(=O)(Cl)c1ccccc1\n"
            + "0,0,0,,0,0,,,0,,1,,TOX27679,CCCCCc1ccco1\n"
            + "0,,1,,,,0,,1,1,1,1,TOX2801,Oc1c(Cl)cc(Cl)c2cccnc12\n"
            + "0,0,0,0,,0,,,0,0,,1,TOX2808,CN(C)CCCN1c2ccccc2Sc2ccc(Cl)cc21\n"
            + "0,,0,1,,,,1,0,,1,,TOX29085,CCCCCCCCCCCCCCn1cc[n+](C)c1\n"
        )

    @staticmethod
    def get_processed_data() -> list:
        """
        Returns a list of dictionaries simulating the processed data for the Tox21MolNet dataset.
        Each dictionary contains 'ident', 'features', and 'labels'.
        """
        return [
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

    @staticmethod
    def get_processed_grouped_data():
        """
        Returns a list of dictionaries simulating the processed data for the Tox21MolNet dataset.
        Each dictionary contains 'ident', 'features', and 'labels'.
        """
        processed_data = Tox21MockData.get_processed_data()
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
