"""IUPAC solid compositional naming rules.

Red Book 2005, Tables IV, VI and IX; IR-2.13 and IR-5.3.3.2:
https://iupac.org/wp-content/uploads/2016/07/Red_Book_2005.pdf
Numerical terms through 9999, Recommendations 1986, NT-1:
https://iupac.qmul.ac.uk/misc/numb.html

This is a conventional sequence, not measured electronegativity. Elements
112-118 are absent from Table VI; their binaries receive formula labels.
"""

# Reverse traversal of Table VI: electropositive constituent first.
ELEMENT_SEQUENCE = (
    "Rn Xe Kr Ar Ne He Fr Cs Rb K Na Li Ra Ba Sr Ca Mg Be "
    "Lr No Md Fm Es Cf Bk Cm Am Pu Np U Pa Th Ac "
    "Lu Yb Tm Er Ho Dy Tb Gd Eu Sm Pm Nd Pr Ce La Y Sc "
    "Rf Hf Zr Ti Db Ta Nb V Sg W Mo Cr Bh Re Tc Mn Hs Os Ru Fe "
    "Mt Ir Rh Co Ds Pt Pd Ni Rg Au Ag Cu Hg Cd Zn Tl In Ga Al B "
    "Pb Sn Ge Si C Bi Sb As P N H Po Te Se S O At I Br Cl F"
).split()
ELEMENT_RANK = {symbol: index for index, symbol in enumerate(ELEMENT_SEQUENCE)}


def multiplicative_prefix(count: int) -> str | None:
    """Simple multiplier; None requests a formula label beyond supported range."""
    if not 1 <= count <= 9999:
        return None
    units = ("", "hen", "do", "tri", "tetra", "penta", "hexa", "hepta", "octa", "nona")
    tens = ("", "deca", "icosa", "triaconta", "tetraconta", "pentaconta",
            "hexaconta", "heptaconta", "octaconta", "nonaconta")
    hundreds = ("", "hecta", "dicta", "tricta", "tetracta", "pentacta",
                "hexacta", "heptacta", "octacta", "nonacta")
    thousands = ("", "kilia", "dilia", "trilia", "tetralia", "pentalia",
                 "hexalia", "heptalia", "octalia", "nonalia")
    if count < 3:
        return ("mono", "di")[count - 1]
    low = count % 100
    if low == 11:
        prefix = "undeca"
    else:
        prefix = units[low % 10]
        decade = tens[low // 10]
        if decade == "icosa" and prefix and prefix[-1] in "aeiou":
            decade = "cosa"
        prefix += decade
    return prefix + hundreds[count // 100 % 10] + thousands[count // 1000]


IDE_NAMES = {
    "Ac": "actinide",
    "Ag": "argentide",
    "Al": "aluminide",
    "Am": "americide",
    "Ar": "argonide",
    "As": "arsenide",
    "At": "astatide",
    "Au": "auride",
    "B": "boride",
    "Ba": "baride",
    "Be": "beryllide",
    "Bh": "bohride",
    "Bi": "bismuthide",
    "Bk": "berkelide",
    "Br": "bromide",
    "C": "carbide",
    "Ca": "calcide",
    "Cd": "cadmide",
    "Ce": "ceride",
    "Cf": "californide",
    "Cl": "chloride",
    "Cm": "curide",
    "Co": "cobaltide",
    "Cr": "chromide",
    "Cs": "caeside",
    "Cu": "cupride",
    "Db": "dubnide",
    "Ds": "darmstadtide",
    "Dy": "dysproside",
    "Er": "erbide",
    "Es": "einsteinide",
    "Eu": "europide",
    "F": "fluoride",
    "Fe": "ferride",
    "Fm": "fermide",
    "Fr": "francide",
    "Ga": "gallide",
    "Gd": "gadolinide",
    "Ge": "germide",
    "H": "hydride",
    "He": "helide",
    "Hf": "hafnide",
    "Hg": "mercuride",
    "Ho": "holmide",
    "Hs": "hasside",
    "I": "iodide",
    "In": "indide",
    "Ir": "iridide",
    "K": "potasside",
    "Kr": "kryptonide",
    "La": "lanthanide",
    "Li": "lithide",
    "Lr": "lawrencide",
    "Lu": "lutetide",
    "Md": "mendelevide",
    "Mg": "magneside",
    "Mn": "manganide",
    "Mo": "molybdenide",
    "Mt": "meitneride",
    "N": "nitride",
    "Na": "sodide",
    "Nb": "niobide",
    "Nd": "neodymide",
    "Ne": "neonide",
    "Ni": "nickelide",
    "No": "nobelide",
    "Np": "neptunide",
    "O": "oxide",
    "Os": "osmide",
    "P": "phosphide",
    "Pa": "protactinide",
    "Pb": "plumbide",
    "Pd": "palladide",
    "Pm": "promethide",
    "Po": "polonide",
    "Pr": "praseodymide",
    "Pt": "platinide",
    "Pu": "plutonide",
    "Ra": "radide",
    "Rb": "rubidide",
    "Re": "rhenide",
    "Rf": "rutherfordide",
    "Rg": "roentgenide",
    "Rh": "rhodide",
    "Rn": "radonide",
    "Ru": "ruthenide",
    "S": "sulfide",
    "Sb": "antimonide",
    "Sc": "scandide",
    "Se": "selenide",
    "Sg": "seaborgide",
    "Si": "silicide",
    "Sm": "samaride",
    "Sn": "stannide",
    "Sr": "strontide",
    "Ta": "tantalide",
    "Tb": "terbide",
    "Tc": "technetide",
    "Te": "telluride",
    "Th": "thoride",
    "Ti": "titanide",
    "Tl": "thallide",
    "Tm": "thulide",
    "U": "uranide",
    "V": "vanadide",
    "W": "tungstide",
    "Xe": "xenonide",
    "Y": "yttride",
    "Yb": "ytterbide",
    "Zn": "zincide",
    "Zr": "zirconide",
}
