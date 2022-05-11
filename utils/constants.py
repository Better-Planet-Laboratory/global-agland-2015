# Units defined in utils
Arc = 'Arc'
Ha = 'Ha'
Kha = 'Kha'
Donum = 'Donum'
Km2 = 'Km2'
M2 = 'M2'
Mha = 'Mha'

# Units Convention
KHA_TO_ARC = 2471.0538
KHA_TO_HA = 1000
KHA_TO_KHA = 1
KHA_TO_DONUM = 10000
KHA_TO_KM2 = 10
KHA_TO_M2 = 10000000
KHA_TO_MHA = 0.001

# Units Lookup Table
UNIT_LOOKUP = {Arc: KHA_TO_ARC,
               Ha: KHA_TO_HA,
               Kha: KHA_TO_KHA,
               Donum: KHA_TO_DONUM,
               Km2: KHA_TO_KM2,
               M2: KHA_TO_M2,
               Mha: KHA_TO_MHA}
