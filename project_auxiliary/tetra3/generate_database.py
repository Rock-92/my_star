from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from tetra3 import tetra3

t3 = tetra3.Tetra3(load_database=None)
t3.generate_database(
    max_fov=40,
    min_fov=5,
    star_max_magnitude=7,
    save_as='gaia5-40',
    star_catalog='gaia',
)
