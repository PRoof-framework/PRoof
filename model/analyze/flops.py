
DEFAULT_FLOPS_TABLE = {
    '_UNKNOWN': 1,

    # simple
    'CMP': 1,
    'ADD': 1,
    'MUL': 1,
    'MAC': 2,

    # complex
    'CAST': 2,
    'DIV': 2,
    'LOG': 6,
    'POW': 4,
    'EXP': 4,
    'SQRT': 8,
    'ERF': 8,
}
