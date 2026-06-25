import os

from astroquery.gaia import Gaia


OUTPUT_FILENAME = 'gaia.csv'
OUTPUT_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), OUTPUT_FILENAME)
LIMITING_MAG = 9.0


print('Submitting Gaia DR3 all-sky catalogue query...')
print(f'Selecting stars with Gaia G magnitude <= {LIMITING_MAG} and proper motion data.')

adql_query = f"""
SELECT source_id, ra, dec, phot_g_mean_mag, pmra, pmdec, ref_epoch
FROM gaiadr3.gaia_source
WHERE phot_g_mean_mag <= {LIMITING_MAG}
  AND pmra IS NOT NULL
  AND pmdec IS NOT NULL
  AND ref_epoch IS NOT NULL
"""

job = Gaia.launch_job_async(adql_query)
results = job.get_results()

df = results.to_pandas()
print(f'Download complete: {len(df)} stars.')

df.to_csv(OUTPUT_PATH, index=False)
print(f'Saved catalogue to {OUTPUT_PATH}')
