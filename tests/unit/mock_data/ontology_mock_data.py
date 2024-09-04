class ChebiMockOntology:
    """
    Nodes:
    - CHEBI:12345 (Compound A)
    - CHEBI:54321 (Compound B)
    - CHEBI:67890 (Compound C)
    - CHEBI:11111 (Compound D)
    - CHEBI:22222 (Compound E)
    - CHEBI:99999 (Compound F)
    - CHEBI:77533 (Compound F, Obsolete node)
    - CHEBI:77564 (Compound H, Obsolete node)
    - CHEBI:88888 (Compound I)

    Valid Edges:
    - CHEBI:54321 -> CHEBI:12345
    - CHEBI:67890 -> CHEBI:12345
    - CHEBI:67890 -> CHEBI:88888
    - CHEBI:11111 -> CHEBI:54321
    - CHEBI:77564 -> CHEBI:54321 (Ignored due to obsolete status)
    - CHEBI:22222 -> CHEBI:67890
    - CHEBI:12345 -> CHEBI:99999
    - CHEBI:77533 -> CHEBI:99999 (Ignored due to obsolete status)
    """

    @staticmethod
    def get_nodes():
        return {12345, 54321, 67890, 11111, 22222, 99999, 88888}

    @staticmethod
    def get_number_of_nodes():
        return len(ChebiMockOntology.get_nodes())

    @staticmethod
    def get_edges_of_transitive_closure_graph():
        return {
            (54321, 12345),
            (54321, 99999),
            (67890, 12345),
            (67890, 99999),
            (67890, 88888),
            (11111, 54321),
            (11111, 12345),
            (11111, 99999),
            (22222, 67890),
            (22222, 12345),
            (22222, 99999),
            (22222, 88888),
            (12345, 99999),
        }

    @staticmethod
    def get_number_of_transitive_edges():
        return len(ChebiMockOntology.get_edges_of_transitive_closure_graph())

    @staticmethod
    def get_edges():
        return {
            (54321, 12345),
            (67890, 12345),
            (67890, 88888),
            (11111, 54321),
            (22222, 67890),
            (12345, 99999),
        }

    @staticmethod
    def get_number_of_edges():
        return len(ChebiMockOntology.get_edges())

    @staticmethod
    def get_obsolete_nodes_ids():
        return {77533, 77564}

    @staticmethod
    def get_raw_data():
        # Create mock terms with a complex hierarchy, names, and SMILES strings
        return """
        [Term]
        id: CHEBI:12345
        name: Compound A
        subset: 2_STAR
        property_value: http://purl.obolibrary.org/obo/chebi/formula "C26H35ClN4O6S" xsd:string
        property_value: http://purl.obolibrary.org/obo/chebi/charge "0" xsd:string
        property_value: http://purl.obolibrary.org/obo/chebi/monoisotopicmass "566.19658" xsd:string
        property_value: http://purl.obolibrary.org/obo/chebi/mass "567.099" xsd:string
        property_value: http://purl.obolibrary.org/obo/chebi/inchikey "ROXPMFGZZQEKHB-IUKKYPGJSA-N" xsd:string
        property_value: http://purl.obolibrary.org/obo/chebi/smiles "C1=CC=CC=C1" xsd:string
        property_value: http://purl.obolibrary.org/obo/chebi/inchi "InChI=1S/C26H35ClN4O6S/c1-16(2)28-26(34)30(5)14-23-17(3)13-31(18(4)15-32)25(33)21-7-6-8-22(24(21)37-23)29-38(35,36)20-11-9-19(27)10-12-20/h6-12,16-18,23,29,32H,13-15H2,1-5H3,(H,28,34)/t17-,18-,23+/m0/s1" xsd:string
        xref: LINCS:LSM-20139
        is_a: CHEBI:54321
        is_a: CHEBI:67890

        [Term]
        id: CHEBI:54321
        name: Compound B
        property_value: http://purl.obolibrary.org/obo/chebi/smiles "C1=CC=CC=C1O" xsd:string
        is_a: CHEBI:11111
        is_a: CHEBI:77564

        [Term]
        id: CHEBI:67890
        name: Compound C
        property_value: http://purl.obolibrary.org/obo/chebi/smiles "C1=CC=CC=C1N" xsd:string
        is_a: CHEBI:22222

        [Term]
        id: CHEBI:11111
        name: Compound D
        property_value: http://purl.obolibrary.org/obo/chebi/smiles "C1=CC=CC=C1F" xsd:string

        [Term]
        id: CHEBI:22222
        name: Compound E
        property_value: http://purl.obolibrary.org/obo/chebi/smiles "C1=CC=CC=C1Cl" xsd:string

        [Term]
        id: CHEBI:99999
        name: Compound F
        property_value: http://purl.obolibrary.org/obo/chebi/smiles "C1=CC=CC=C1Br" xsd:string
        is_a: CHEBI:12345

        [Term]
        id: CHEBI:77533
        name: Compound G
        is_a: CHEBI:99999
        property_value: http://purl.obolibrary.org/obo/chebi/smiles "C1=C1Br" xsd:string
        is_obsolete: true

        [Term]
        id: CHEBI:77564
        name: Compound H
        property_value: http://purl.obolibrary.org/obo/chebi/smiles "CC=C1Br" xsd:string
        is_obsolete: true

        [Typedef]
        id: has_major_microspecies_at_pH_7_3
        name: has major microspecies at pH 7.3
        is_cyclic: true
        is_transitive: false

        [Term]
        id: CHEBI:88888
        name: Compound I
        property_value: http://purl.obolibrary.org/obo/chebi/smiles "C1=CC=CC=C1[Mg+]" xsd:string
        is_a: CHEBI:67890
        """
