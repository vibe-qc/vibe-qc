import sys; sys.path.insert(0,'python')
from vibeqc_naming import name_from_atoms
from vibeqc_naming.structure_db import known_names, structure_from_name

all_names = known_names()
print(f'Built-in structures: {len(all_names)}')

ok = fail = 0
for name in sorted(all_names):
    atoms = structure_from_name(name)
    if atoms and len(atoms) >= 2:
        z_vals = {z for z,*_ in atoms}
        ok += 1
    else:
        print(f'  FAIL: {name} -> {len(atoms) if atoms else 0} atoms')
        fail += 1

print(f'\n{ok}/{len(all_names)} structures load with >=2 atoms ({fail} failed)')
