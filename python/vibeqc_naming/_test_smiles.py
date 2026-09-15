"""SMILES generation tests — assert correct chemistry."""

import sys; sys.path.insert(0,'python')
from vibeqc_naming.smiles import smiles_from_name

TESTS = [
    ("water",           "O"),
    ("methane",         "C"),
    ("ammonia",         "N"),
    ("ethane",          "CC"),
    ("ethanol",         "CCO"),
    ("ethylene",        "C=C"),
    ("acetylene",       "C#C"),
    ("formaldehyde",    "C=O"),
    ("formic acid",     "O=CO"),
    ("benzene",         "c1ccccc1"),
    ("pyridine",        "n1ccccc1"),
    ("acetic acid",     "CC=O(O)"),
]

ok = fail = 0
for name, expected in TESTS:
    smi = smiles_from_name(name)
    if smi == expected:
        ok += 1
        print(f"  PASS  {name:<16s} -> {smi}")
    else:
        fail += 1
        print(f"  FAIL  {name:<16s} -> {smi!r}  (expected {expected!r})")

print(f"\n{ok}/{len(TESTS)} pass")
if fail:
    import sys; sys.exit(1)
