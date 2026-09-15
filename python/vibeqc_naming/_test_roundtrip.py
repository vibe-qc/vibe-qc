"""Round-trip test: structure_from_name -> name_from_atoms for all structures.

Tests all 83 entries in the structure database. Formula-shorthand keys
(c2h4, etc.) are mapped to their expected full names. Every other name
must round-trip: the naming engine must recognise the coordinates it
produced from that name.
"""

import sys; sys.path.insert(0,'python')
from vibeqc_naming import name_from_atoms
from vibeqc_naming.structure_db import known_names, structure_from_name

# Formula-shorthand keys that expand to the canonical name.
_SHORTHAND = {
    'c2h2':'acetylene','c2h4':'ethylene','c2h6':'ethane',
    'c2h4o':'acetaldehyde','c2h6o':'ethanol','c3h6':'cyclopropane',
    'c3h8':'propane','c4h10':'butane','c6h12':'cyclohexane',
    'c6h6':'benzene','c7h6o':'benzaldehyde',
    'ch2o':'formaldehyde','ch4':'methane','ch4o':'methanol',
    'co':'carbon monoxide','co2':'carbon dioxide',
    'h2':'dihydrogen','h2o':'water','h2o2':'hydrogen peroxide',
    'hcl':'hydrogen chloride','hf':'hydrogen fluoride',
    'n2':'dinitrogen','n2o':'dinitrogen monoxide',
    'nacl':'sodium chloride','nh3':'ammonia','no2':'nitrogen dioxide',
    'o2':'dioxygen','so3':'sulfur trioxide',
    'ethene':'ethylene','ethyne':'acetylene',
    'b2h6':'diborane','hcn':'hydrogen cyanide','dmso':'dimethyl sulfoxide','thf':'tetrahydrofuran',
}

ok = fail = 0
for name in sorted(known_names()):
    atoms = structure_from_name(name)
    named = name_from_atoms(atoms)
    expected = _SHORTHAND.get(name, name)
    if expected.lower() in named.lower() or named.lower() in expected.lower():
        ok += 1
    else:
        fail += 1
        print(f'  FAIL  {name:<20s} -> {named!r}  (expected {expected!r})')

print(f'\nRound-trip: {ok}/{len(known_names())} pass, {fail} fail')
if fail:
    print(f'\n{fail} structure(s) do not round-trip.')
    print('These are real naming gaps — either the geometry is wrong or')
    print('the naming engine does not recognise the molecule it built.')
    import sys; sys.exit(1)
