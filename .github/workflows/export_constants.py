import json

import chebai.preprocessing.reader as dr

# Define the constants you want to export
constants = {
    "EMBEDDING_OFFSET": dr.EMBEDDING_OFFSET,
    "CLS_TOKEN": dr.CLS_TOKEN,
    "PADDING_TOKEN_INDEX": dr.PADDING_TOKEN_INDEX,
    "MASK_TOKEN_INDEX": dr.MASK_TOKEN_INDEX,
}

# Write constants to a JSON file
with open("constants.json", "w") as f:
    json.dump(constants, f)
