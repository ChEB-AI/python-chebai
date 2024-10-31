import json

from chebai.preprocessing.reader import (
    CLS_TOKEN,
    EMBEDDING_OFFSET,
    MASK_TOKEN_INDEX,
    PADDING_TOKEN_INDEX,
)

# Define the constants you want to export
# Any changes in the key names here should also follow the same change in verify_constants.yml code
constants = {
    "EMBEDDING_OFFSET": EMBEDDING_OFFSET,
    "CLS_TOKEN": CLS_TOKEN,
    "PADDING_TOKEN_INDEX": PADDING_TOKEN_INDEX,
    "MASK_TOKEN_INDEX": MASK_TOKEN_INDEX,
}

if __name__ == "__main__":
    # Write constants to a JSON file
    with open("constants.json", "w") as f:
        json.dump(constants, f)
