from __future__ import annotations
import re
from typing import List

from rdkit import Chem


def _build_bracket_atoms() -> List[str]:
    """Enumerate chemically meaningful bracketed atoms."""
    pt = Chem.GetPeriodicTable()
    elements = [pt.GetElementSymbol(i) for i in range(1, 119)]
    # aromatic forms used in SMILES
    elements += ["c", "n", "o", "s", "p", "b", "te", "se", "si"]
    charges = [i for i in range(-5, 9) if i != 0]
    hydrogens = range(1, 9)
    stereo = ["@", "@@"]
    isotopes = range(1, 300)  # [295Og] is the heaviest isotope in PubChem

    tokens = []
    for el in elements:
        tokens.append(f"element_{el}")
    for ch in charges:
        tokens.append(f"charge_{ch}")
    for h in hydrogens:
        tokens.append(f"hydrogens_{h}")
    for st in stereo:
        tokens.append(f"stereo_{st}")
    for iso in isotopes:
        tokens.append(f"isotope_{iso}")

    return tokens


NON_BRACKET_TOKENS = [
    # bonds / structure
    "(",
    ")",
    "=",
    "#",
    "->",
    "<-",
    ">>",
    "-",
    "+",
    "/",
    "\\",
    ":",
    ".",
    "~",
    "*",
    "$",
    "?",
    "@",
    "@@",
    # ring closures: single-digit
    *[str(d) for d in range(10)],
    # ring closures: %10..%99
    *[f"%{n:02d}" for n in range(10, 100)],
]


def _build_default_vocab() -> List[str]:
    """non-bracket symbols + bracketed atoms."""

    brackets = _build_bracket_atoms()

    # de-duplicate while preserving order (specials first)
    seen, vocab = set(), []
    for tok in NON_BRACKET_TOKENS + brackets:
        if tok not in seen:
            seen.add(tok)
            vocab.append(tok)
    return vocab


# change to original regex: added -> and <- for handling dative bonds (we use RDKit-normalised SMILES, this is not a standard SMILES feature)
SMI_REGEX_PATTERN = r"""(\[[^\]]+]|Br?|Cl?|N|O|S|P|F|I|b|c|n|o|s|p|\(|\)|\.|=|#|->|<-|>>?|-|\+|\\|\/|:|~|@@|@|\?|\*|\$|\%[0-9]{2}|[0-9])"""
EMBEDDING_OFFSET = 10
UNKNOWN_TOKEN_IDX = 3
ORGANIC_SUBSET = frozenset(
    {"B", "C", "N", "O", "P", "S", "F", "Cl", "Br", "I", "b", "c", "n", "o", "s", "p"}
)


class BasicSmilesTokenizer(object):
    """
    References
    ----------
    .. [1] Philippe Schwaller, Teodoro Laino, Théophile Gaudin, Peter Bolgar, Christopher A. Hunter, Costas Bekas, and Alpha A. Lee
        ACS Central Science 2019 5 (9): Molecular Transformer: A Model for Uncertainty-Calibrated Chemical Reaction Prediction
        1572-1583 DOI: 10.1021/acscentsci.9b00576
    """

    def __init__(self, regex_pattern: str = SMI_REGEX_PATTERN):
        """Constructs a BasicSMILESTokenizer.

        Parameters
        ----------
        regex: string
            SMILES token regex
        """
        self.regex_pattern = regex_pattern
        self.regex = re.compile(self.regex_pattern)

        self.vocab = _build_default_vocab()
        self.vocab_dict = {
            tok: idx + EMBEDDING_OFFSET for idx, tok in enumerate(self.vocab)
        }
        self.idx_to_token = {idx: tok for tok, idx in self.vocab_dict.items()}

    def _parse_bracket_atom(self, bracket_token: str) -> List[str]:
        """
        Parse a bracketed atom token into its components.

        When all attributes are at their defaults (charge=0, hydrogens=0, isotope=None, stereo=None),
        only the element token is emitted. Otherwise all 5 tokens are emitted.

        E.g. "[N]"   -> ["element_N"]
             "[85Kr]" -> ["isotope_85", "element_Kr", "charge_0", "hydrogens_0", "stereo_None"]
        """
        atom_str = bracket_token[1:-1]  # Remove brackets

        # special case: any atom containing a * is treated as a wildcard (e.g. [3*:0],[1*])
        if "*" in atom_str:
            return ["*"]

        isotope = None
        element = None
        charge = 0
        hydrogens = 0
        stereo = None

        pos = 0

        # Parse isotope (leading digits)
        iso_str = ""
        while pos < len(atom_str) and atom_str[pos].isdigit():
            iso_str += atom_str[pos]
            pos += 1
        if iso_str:
            isotope = iso_str

        # Parse element (1-2 letters)
        if pos < len(atom_str) and (atom_str[pos].isupper() or atom_str[pos].islower()):
            element = atom_str[pos]
            pos += 1
            if pos < len(atom_str) and atom_str[pos].islower():
                element += atom_str[pos]
                pos += 1

        # Parse stereo (@ or @@)
        if pos < len(atom_str) and atom_str[pos] == "@":
            if pos + 1 < len(atom_str) and atom_str[pos + 1] == "@":
                stereo = "@@"
                pos += 2
            else:
                stereo = "@"
                pos += 1

        # Parse hydrogens (H, H2, H3, H4)
        if pos < len(atom_str) and atom_str[pos] == "H":
            hydrogens = 1
            pos += 1
            if pos < len(atom_str) and atom_str[pos].isdigit():
                hydrogens = int(atom_str[pos])
                pos += 1

        # Parse charge (+, -, +2, -2, +3, -3)
        if pos < len(atom_str):
            if atom_str[pos] == "+":
                pos += 1
                if pos < len(atom_str) and atom_str[pos].isdigit():
                    charge = int(atom_str[pos])
                    pos += 1
                else:
                    charge = 1
            elif atom_str[pos] == "-":
                pos += 1
                if pos < len(atom_str) and atom_str[pos].isdigit():
                    charge = -int(atom_str[pos])
                    pos += 1
                else:
                    charge = -1

        # return element token and optionally isotope, charge, hydrogens, stereo tokens
        res = [f"element_{element}"]
        if isotope is not None:
            res.append(f"isotope_{isotope}")
        if charge != 0:
            res.append(f"charge_{charge}")
        if hydrogens != 0:
            res.append(f"hydrogens_{hydrogens}")
        if stereo is not None:
            res.append(f"stereo_{stereo}")
        return res

    def tokenize(self, text):
        """Tokenize a SMILES string, breaking bracketed atoms into 5 components.

        Non-bracketed tokens are returned as-is.
        Bracketed atoms are decomposed into: isotope, element, charge, hydrogens, stereo.
        """
        raw_tokens = [token for token in self.regex.findall(text)]
        tokens = []

        for token in raw_tokens:
            if token in NON_BRACKET_TOKENS:
                tokens.append(token)
                continue
            if not (token.startswith("[") and token.endswith("]")):
                token = (
                    f"[{token}]"  # Wrap non-bracket tokens in brackets for uniformity
                )
            # Parse bracketed atom into 5 components
            components = self._parse_bracket_atom(token)
            tokens.extend(components)

        return tokens

    def encode(self, text):
        tokens = self.tokenize(text)
        return [self.vocab_dict.get(token, UNKNOWN_TOKEN_IDX) for token in tokens]

    def _reassemble_bracket_atom(
        self,
        element: str,
        isotope: str = "None",
        charge: int = 0,
        hydrogens: int = 0,
        stereo: str = "None",
    ) -> str:
        if (
            isotope == "None"
            and charge == 0
            and stereo == "None"
            and hydrogens == 0
            and element in ORGANIC_SUBSET
        ):
            return element
        inner = ""
        if isotope != "None":
            inner += isotope
        inner += element
        if stereo != "None":
            inner += stereo
        if hydrogens > 0:
            inner += "H"
            if hydrogens > 1:
                inner += str(hydrogens)
        if charge > 0:
            inner += "+"
            if charge > 1:
                inner += str(charge)
        elif charge < 0:
            inner += "-"
            if charge < -1:
                inner += str(-charge)
        return f"[{inner}]"

    def decode(self, token_ids, skip_special_tokens=False):
        tokens = [self.idx_to_token.get(idx, "[UNK]") for idx in token_ids]
        if skip_special_tokens:
            tokens = [tok for tok in tokens if tok not in self.vocab_dict]

        result = []
        i = 0
        while i < len(tokens):
            tok = tokens[i]
            if tok.startswith("element_"):
                i += 1
                add = ["None", 0, 0, "None"]
                for idx, additional_token in enumerate(
                    ["isotope_", "charge_", "hydrogens_", "stereo_"]
                ):
                    if i >= len(tokens):
                        break
                    if tokens[i].startswith(additional_token):
                        add[idx] = tokens[i][len(additional_token) :]
                        if additional_token in ["charge_", "hydrogens_"]:
                            add[idx] = int(add[idx])
                        i += 1

                result.append(
                    self._reassemble_bracket_atom(
                        tok[len("element_") :],
                        *add,
                    )
                )
            else:
                result.append(tok)
                i += 1

        return "".join(result)


# ---- quick self-test ------------------------------------------------------
if __name__ == "__main__":
    # tok = SmilesTokenizer.build_default()
    # print(f"Vocab size: {tok.vocab_size}")
    tok = BasicSmilesTokenizer()
    print(f"Vocab size: {len(tok.vocab)}")
    examples = [
        "CC(=O)Oc1ccccc1C(=O)O",  # aspirin
        "C[C@H](N)C(=O)O",  # L-alanine
        "[13CH3]CO",  # isotope
        "C1CC2(CCCCC2)CC1",  # spiro
        "c1ccc2c(c1)[nH]cn2",  # benzimidazole with [nH]
        # "CC(=O)N[C@@H]1[C@H](O[C@H]2[C@H](O)[C@@H](NC(C)=O)[C@H](O)O[C@@H]2CO[C@@H]2O[C@@H](C)[C@@H](O)[C@@H](O)[C@@H]2O)O[C@H](CO)[C@@H](O[C@@H]2O[C@H](CO[C@H]3O[C@H](CO[C@@H]4O[C@H](CO)[C@@H](O[C@@H]5O[C@H](CO)[C@H](O)[C@H](O[C@@H]6O[C@H](CO)[C@@H](O[C@@H]7O[C@H](CO)[C@H](O)[C@H](O[C@]8(C(=O)O)C[C@H](O)[C@@H](NC(C)=O)[C@H]([C@H](O)[C@H](O)CO)O8)[C@H]7O)[C@H](O)[C@H]6NC(C)=O)[C@H]5O)[C@H](O)[C@H]4NC(C)=O)[C@@H](O)[C@H](O)[C@@H]3O[C@@H]3O[C@H](CO)[C@@H](O[C@@H]4O[C@H](CO)[C@H](O)[C@H](O[C@@H]5O[C@H](CO)[C@@H](O[C@@H]6O[C@H](CO)[C@H](O)[C@H](O[C@]7(C(=O)O)C[C@H](O)[C@@H](NC(C)=O)[C@H]([C@H](O)[C@H](O)CO)O7)[C@H]6O)[C@H](O)[C@H]5NC(C)=O)[C@H]4O)[C@H](O)[C@H]3NC(C)=O)[C@@H](O)[C@H](O[C@H]3O[C@H](CO)[C@@H](O[C@@H]4O[C@H](CO)[C@@H](O[C@@H]5O[C@H](CO)[C@H](O)[C@H](O[C@]6(C(=O)O)C[C@H](O)[C@@H](NC(C)=O)[C@H]([C@H](O)[C@H](O)CO)O6)[C@H]5O)[C@H](O)[C@H]4NC(C)=O)[C@H](O)[C@@H]3O[C@@H]3O[C@H](CO)[C@@H](O[C@@H]4O[C@H](CO)[C@H](O)[C@H](O)[C@H]4O)[C@H](O)[C@@H]3NC(C)=O)[C@@H]2O)[C@@H]1O",
    ]

    for s in examples:
        tokens = tok.tokenize(s)
        ids = tok.encode(s)
        print(f"\n{s}")
        print(f"  tokens: {tokens}")
        print(f"  ids:    {ids}")
        print(f"  decode: {tok.decode(ids)}")
