#!/usr/bin/env python3
"""
Tutorial 04: Multi-Survey Cross-Matching with HEALPix

This tutorial demonstrates:
1. Understanding HEALPix spatial indexing
2. Downloading survey data for specific HEALPix regions
3. Reading HDF5 files directly
4. Cross-matching multiple surveys (Gaia, SDSS, Chandra)
5. Creating multi-wavelength catalogs

Learning Objectives:
- Understand how astronomical surveys organize data spatially
- Learn to download and access HEALPix-partitioned data
- Perform spatial cross-matching with astropy
- Combine data from optical, spectroscopic, and X-ray surveys

Expected runtime: ~2-3 minutes
"""

import subprocess
from pathlib import Path
import h5py
import numpy as np
from astropy.coordinates import SkyCoord
import astropy.units as u
import pandas as pd
import matplotlib.pyplot as plt

print("="*70)
print("Tutorial 04: Multi-Survey Cross-Matching")
print("="*70)

# ============================================================================
# Part 1: Understanding HEALPix
# ============================================================================
print("\n" + "="*70)
print("Part 1: Understanding HEALPix Spatial Indexing")
print("="*70)

print("""
What is HEALPix?
----------------
HEALPix (Hierarchical Equal Area isoLatitude Pixelization) divides the
sky into equal-area pixels, making it ideal for large astronomical surveys.

Key Properties:
- Equal-area pixels (important for statistics)
- Hierarchical structure (nside = 2^k)
- Each pixel has unique integer ID
- Used by: Planck, SDSS, Gaia, MultimodalUniverse

For MultimodalUniverse:
- Data organized by HEALPix pixel at nside=32
- This creates 12,288 pixels covering the full sky
- Each pixel ≈ 13.4 square degrees
- Sources in same region share HEALPix ID

Why HEALPix for cross-matching?
- Only need to download overlapping pixels
- Fast spatial queries within pixels
- Efficient for distributed data storage
""")

# ============================================================================
# Part 2: Selecting a HEALPix Region
# ============================================================================
print("\n" + "="*70)
print("Part 2: Selecting HEALPix Region for Download")
print("="*70)

print("""
For this tutorial, we'll use HEALPix pixel 583 because:
- Contains data from multiple surveys
- Small enough for quick download
- Rich in different source types

Region coverage:
- RA: ~180° (near celestial equator)
- Dec: ~0° (galactic plane region)
- Area: ~13.4 square degrees
""")

healpix = 583
print(f"\nSelected HEALPix pixel: {healpix}")

# Create data directory
print("\nCreating data directory structure...")
Path("data/MultimodalUniverse/v1/gaia").mkdir(parents=True, exist_ok=True)
Path("data/MultimodalUniverse/v1/sdss").mkdir(parents=True, exist_ok=True)
Path("data/MultimodalUniverse/v1/chandra/spectra").mkdir(parents=True, exist_ok=True)
print("✅ Directories created")

# ============================================================================
# Part 3: Downloading Survey Data
# ============================================================================
print("\n" + "="*70)
print("Part 3: Downloading Survey Data for HEALPix 583")
print("="*70)

print("""
We'll download data from three surveys:
1. Gaia - Optical astrometry and photometry (baseline catalog)
2. SDSS - Optical spectroscopy with redshifts
3. Chandra - X-ray source catalog

Data format:
- HDF5 files (.hdf5) - Efficient binary format
- Direct column access without loading full file
- Standard in astronomy for large datasets
""")

# Download Gaia data
print("\n[1/3] Downloading Gaia data...")
subprocess.run([
    "wget", "-r", "-np", "-nH", "--cut-dirs=1",
    "-R", "index.html*", "-q",
    f"https://users.flatironinstitute.org/~polymathic/data/MultimodalUniverse/v1/gaia/gaia/healpix={healpix}/"
], check=True)
print("  ✅ Gaia data downloaded")

# Download SDSS data (all sub-surveys)
print("\n[2/3] Downloading SDSS spectroscopic data (all sub-surveys)...")
print("     SDSS includes: sdss, boss, eboss, segue1, segue2")
sdss_subsurveys = ['sdss', 'boss', 'eboss', 'segue1', 'segue2']
for subsurvey in sdss_subsurveys:
    print(f"     Downloading {subsurvey}...", end=" ")
    result = subprocess.run([
        "wget", "-r", "-np", "-nH", "--cut-dirs=1",
        "-R", "index.html*", "-q",
        f"https://users.flatironinstitute.org/~polymathic/data/MultimodalUniverse/v1/sdss/{subsurvey}/healpix={healpix}/"
    ])
    if result.returncode == 0:
        print("✓")
    else:
        print("(no data for this HEALPix)")
print("  ✅ SDSS data downloaded (all available sub-surveys)")

# Download Chandra data
print("\n[3/3] Downloading Chandra X-ray data...")
subprocess.run([
    "wget", "-r", "-np", "-nH", "--cut-dirs=1",
    "-R", "index.html*", "-q",
    f"https://users.flatironinstitute.org/~polymathic/data/MultimodalUniverse/v1/chandra/spectra/healpix={healpix}/"
], check=True)
print("  ✅ Chandra data downloaded")

print("\n✅ All survey data downloaded successfully")
has_chandra = True

# ============================================================================
# Part 4: Reading HDF5 Files
# ============================================================================
print("\n" + "="*70)
print("Part 4: Understanding HDF5 File Structure")
print("="*70)

print("""
HDF5 (Hierarchical Data Format) files store:
- Multiple datasets (columns) in one file
- Metadata about each dataset
- Efficient random access to columns
- Compressed data

Advantages:
- Read only needed columns
- Fast I/O for large datasets
- Self-describing format
""")

# Define file paths
gaia_file = f"data/MultimodalUniverse/v1/gaia/gaia/healpix={healpix}/001-of-001.hdf5"
sdss_file = f"data/MultimodalUniverse/v1/sdss/sdss/healpix={healpix}/001-of-001.hdf5"
chandra_file = f"data/MultimodalUniverse/v1/chandra/spectra/healpix={healpix}/001-of-001.hdf5"

# Explore Gaia HDF5 structure
print(f"\nExploring Gaia HDF5 file structure:")
print(f"File: {gaia_file}")
with h5py.File(gaia_file, 'r') as f:
    print(f"  Number of datasets (columns): {len(f.keys())}")
    print(f"  Number of sources: {len(f['ra'][:])}")
    print(f"\n  Key columns:")
    key_cols = ['ra', 'dec', 'parallax', 'phot_g_mean_mag', 'source_id']
    for col in key_cols:
        if col in f.keys():
            print(f"    - {col:20s}: {f[col].shape}")

# ============================================================================
# Part 5: Loading Data for Cross-Matching
# ============================================================================
print("\n" + "="*70)
print("Part 5: Loading Catalog Data")
print("="*70)

print("\nLoading Gaia catalog (our baseline)...")
with h5py.File(gaia_file, 'r') as f:
    gaia_ra = f['ra'][:]
    gaia_dec = f['dec'][:]
    gaia_parallax = f['parallax'][:]
    gaia_g_mag = f['phot_g_mean_mag'][:]
    print(f"  ✅ Loaded {len(gaia_ra):,} Gaia sources")

print("\nLoading SDSS catalog (all sub-surveys)...")
# Load and combine all available sub-surveys
sdss_ra_list = []
sdss_dec_list = []
sdss_z_list = []

for subsurvey in sdss_subsurveys:
    sdss_file = f"data/MultimodalUniverse/v1/sdss/{subsurvey}/healpix={healpix}/001-of-001.hdf5"
    if Path(sdss_file).exists():
        with h5py.File(sdss_file, 'r') as f:
            n_sources = len(f['ra'][:])
            print(f"    - {subsurvey}: {n_sources:,} sources")
            sdss_ra_list.append(f['ra'][:])
            sdss_dec_list.append(f['dec'][:])
            sdss_z_list.append(f['Z'][:])
    else:
        print(f"    - {subsurvey}: no data for this HEALPix")

# Concatenate all sub-surveys
if sdss_ra_list:
    sdss_ra = np.concatenate(sdss_ra_list)
    sdss_dec = np.concatenate(sdss_dec_list)
    sdss_z = np.concatenate(sdss_z_list)
    print(f"  ✅ Loaded {len(sdss_ra):,} total SDSS sources (all sub-surveys)")
else:
    print("  ⚠️  No SDSS data available for this HEALPix")
    sdss_ra = np.array([])
    sdss_dec = np.array([])
    sdss_z = np.array([])

print("\nLoading Chandra catalog...")
with h5py.File(chandra_file, 'r') as f:
    chandra_ra = f['ra'][:]
    chandra_dec = f['dec'][:]
    # Chandra has flux in different energy bands
    if 'flux_aper_b' in f.keys():
        chandra_flux = f['flux_aper_b'][:]
    else:
        chandra_flux = np.ones(len(chandra_ra))  # placeholder
    print(f"  ✅ Loaded {len(chandra_ra):,} Chandra sources")

# ============================================================================
# Part 6: Spatial Cross-Matching
# ============================================================================
print("\n" + "="*70)
print("Part 6: Performing Spatial Cross-Matching")
print("="*70)

print("""
Cross-matching strategy:
1. Use Gaia as baseline (most sources, best astrometry)
2. Match SDSS to Gaia (1 arcsec radius)
3. Match Chandra to Gaia (2 arcsec radius - X-ray positions less precise)

Matching radius considerations:
- Gaia: ~0.1 mas positional accuracy
- SDSS: ~0.1 arcsec positional accuracy
- Chandra: ~0.5-1 arcsec positional accuracy
""")

# Create SkyCoord objects
print("\nCreating coordinate objects...")
gaia_coords = SkyCoord(ra=gaia_ra*u.deg, dec=gaia_dec*u.deg)
if len(sdss_ra) > 0:
    sdss_coords = SkyCoord(ra=sdss_ra*u.deg, dec=sdss_dec*u.deg)
chandra_coords = SkyCoord(ra=chandra_ra*u.deg, dec=chandra_dec*u.deg)
print("  ✅ Coordinate objects created")

# Match SDSS to Gaia
if len(sdss_ra) > 0:
    print("\n[1/2] Matching SDSS to Gaia (1 arcsec radius)...")
    idx_sdss_gaia, sep2d_sdss, _ = sdss_coords.match_to_catalog_sky(gaia_coords)
    match_mask_sdss = sep2d_sdss < 1.0 * u.arcsec
    n_gaia_sdss = np.sum(match_mask_sdss)
    print(f"  ✅ Found {n_gaia_sdss} Gaia-SDSS matches")
    if n_gaia_sdss > 0:
        print(f"     Median separation: {np.median(sep2d_sdss[match_mask_sdss].arcsec):.3f} arcsec")
else:
    print("\n[1/2] Skipping SDSS matching (no SDSS data available)")
    idx_sdss_gaia = np.array([])
    sep2d_sdss = np.array([]) * u.arcsec
    match_mask_sdss = np.array([], dtype=bool)
    n_gaia_sdss = 0

# Match Chandra to Gaia
print("\n[2/2] Matching Chandra to Gaia (2 arcsec radius)...")
idx_chandra_gaia, sep2d_chandra, _ = chandra_coords.match_to_catalog_sky(gaia_coords)
match_mask_chandra = sep2d_chandra < 2.0 * u.arcsec
n_gaia_chandra = np.sum(match_mask_chandra)
print(f"  ✅ Found {n_gaia_chandra} Gaia-Chandra matches")
print(f"     Median separation: {np.median(sep2d_chandra[match_mask_chandra].arcsec):.3f} arcsec")

# ============================================================================
# Part 7: Creating Multi-Wavelength Catalog
# ============================================================================
print("\n" + "="*70)
print("Part 7: Creating Multi-Wavelength Catalog")
print("="*70)

print("""
We'll create three catalogs:
1. Gaia-SDSS: Optical + spectroscopy
2. Gaia-Chandra: Optical + X-ray
3. Gaia-SDSS-Chandra: Complete multi-wavelength (rarest)
""")

# Gaia-SDSS matches
gaia_sdss_matches = []
for i, has_match in enumerate(match_mask_sdss):
    if has_match:
        gaia_idx = idx_sdss_gaia[i]
        gaia_sdss_matches.append({
            'ra': gaia_ra[gaia_idx],
            'dec': gaia_dec[gaia_idx],
            'gaia_g_mag': gaia_g_mag[gaia_idx],
            'gaia_parallax': gaia_parallax[gaia_idx],
            'sdss_z': sdss_z[i],
            'separation_arcsec': sep2d_sdss[i].arcsec
        })

df_gaia_sdss = pd.DataFrame(gaia_sdss_matches)
print(f"\n✅ Gaia-SDSS catalog: {len(df_gaia_sdss)} sources")

# Gaia-Chandra matches
gaia_chandra_matches = []
for i, has_match in enumerate(match_mask_chandra):
    if has_match:
        gaia_idx = idx_chandra_gaia[i]
        gaia_chandra_matches.append({
            'ra': gaia_ra[gaia_idx],
            'dec': gaia_dec[gaia_idx],
            'gaia_g_mag': gaia_g_mag[gaia_idx],
            'gaia_parallax': gaia_parallax[gaia_idx],
            'chandra_flux': chandra_flux[i],
            'separation_arcsec': sep2d_chandra[i].arcsec
        })

df_gaia_chandra = pd.DataFrame(gaia_chandra_matches)
print(f"✅ Gaia-Chandra catalog: {len(df_gaia_chandra)} sources")

# Find triple matches (Gaia-SDSS-Chandra)
print("\nFinding triple matches (Gaia + SDSS + Chandra)...")
# This requires finding Gaia sources that appear in both match lists
gaia_sdss_indices = set(idx_sdss_gaia[match_mask_sdss])
gaia_chandra_indices = set(idx_chandra_gaia[match_mask_chandra])
gaia_triple_indices = gaia_sdss_indices.intersection(gaia_chandra_indices)

print(f"✅ Triple matches: {len(gaia_triple_indices)} sources")
print(f"   (Optical + Spectroscopy + X-ray)")

# ============================================================================
# Part 8: Analyzing Cross-Match Results
# ============================================================================
print("\n" + "="*70)
print("Part 8: Analyzing Cross-Match Results")
print("="*70)

# Summary statistics
sdss_pct = 100*len(df_gaia_sdss)/len(sdss_ra) if len(sdss_ra) > 0 else 0
chandra_pct = 100*len(df_gaia_chandra)/len(chandra_ra) if len(chandra_ra) > 0 else 0

print(f"""
Cross-Match Summary:
-------------------
Base catalog (Gaia):           {len(gaia_ra):,} sources
SDSS spectroscopy:             {len(sdss_ra):,} sources
Chandra X-ray:                 {len(chandra_ra):,} sources

Matches:
  Gaia-SDSS:                   {len(df_gaia_sdss):,} ({sdss_pct:.1f}% of SDSS)
  Gaia-Chandra:                {len(df_gaia_chandra):,} ({chandra_pct:.1f}% of Chandra)
  Gaia-SDSS-Chandra (triple):  {len(gaia_triple_indices):,}

Match fractions:
  SDSS with Gaia match:        {sdss_pct:.1f}%
  Chandra with Gaia match:     {chandra_pct:.1f}%
""")

# ============================================================================
# Part 9: Visualizing Cross-Match Results
# ============================================================================
print("\n" + "="*70)
print("Part 9: Visualizing Multi-Wavelength Data")
print("="*70)

fig, axes = plt.subplots(2, 2, figsize=(14, 12))

# Plot 1: Sky positions
ax = axes[0, 0]
ax.scatter(gaia_ra, gaia_dec, s=0.5, alpha=0.3, label=f'Gaia ({len(gaia_ra):,})', c='gray')
if len(sdss_ra) > 0:
    ax.scatter(sdss_ra, sdss_dec, s=2, alpha=0.5, label=f'SDSS ({len(sdss_ra):,})', c='blue')
ax.scatter(chandra_ra, chandra_dec, s=10, alpha=0.7, label=f'Chandra ({len(chandra_ra):,})', c='red', marker='x')
ax.set_xlabel('RA (deg)')
ax.set_ylabel('Dec (deg)')
ax.set_title(f'Sky Coverage (HEALPix {healpix})')
ax.legend()
ax.grid(True, alpha=0.3)

# Plot 2: Redshift distribution (Gaia-SDSS matches)
ax = axes[0, 1]
if len(df_gaia_sdss) > 0:
    ax.hist(df_gaia_sdss['sdss_z'], bins=30, edgecolor='black', alpha=0.7)
    ax.set_xlabel('Redshift')
    ax.set_ylabel('Number of Galaxies')
    ax.set_title('Redshift Distribution (Gaia-SDSS matches)')
else:
    ax.text(0.5, 0.5, 'No Gaia-SDSS matches', ha='center', va='center', transform=ax.transAxes)
    ax.set_title('Redshift Distribution (Gaia-SDSS matches)')
ax.grid(True, alpha=0.3)

# Plot 3: Color-magnitude for matches
ax = axes[1, 0]
if len(df_gaia_sdss) > 0:
    ax.scatter(df_gaia_sdss['sdss_z'], df_gaia_sdss['gaia_g_mag'],
              s=5, alpha=0.5, label='Gaia-SDSS')
    ax.set_xlabel('Redshift')
    ax.set_ylabel('Gaia G magnitude')
    ax.set_title('Magnitude vs Redshift')
    ax.invert_yaxis()
    ax.legend()
else:
    ax.text(0.5, 0.5, 'No Gaia-SDSS matches', ha='center', va='center', transform=ax.transAxes)
    ax.set_title('Magnitude vs Redshift')
ax.grid(True, alpha=0.3)

# Plot 4: Match separation distributions
ax = axes[1, 1]
if len(df_gaia_sdss) > 0:
    ax.hist(df_gaia_sdss['separation_arcsec'], bins=30, alpha=0.6,
           label=f'Gaia-SDSS (median={np.median(df_gaia_sdss["separation_arcsec"]):.3f}")',
           edgecolor='black')
if len(df_gaia_chandra) > 0:
    ax.hist(df_gaia_chandra['separation_arcsec'], bins=30, alpha=0.6,
           label=f'Gaia-Chandra (median={np.median(df_gaia_chandra["separation_arcsec"]):.3f}")',
           edgecolor='black')
ax.set_xlabel('Separation (arcsec)')
ax.set_ylabel('Number of Matches')
ax.set_title('Cross-Match Separation Distribution')
ax.legend()
ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('tutorial04_crossmatch_analysis.png', dpi=150, bbox_inches='tight')
print("✅ Saved visualization to tutorial04_crossmatch_analysis.png")
plt.close()

# ============================================================================
# Part 10: Saving Catalogs
# ============================================================================
print("\n" + "="*70)
print("Part 10: Saving Cross-Matched Catalogs")
print("="*70)

# Save catalogs
df_gaia_sdss.to_csv('tutorial04_gaia_sdss_matches.csv', index=False)
print(f"✅ Saved Gaia-SDSS catalog: tutorial04_gaia_sdss_matches.csv")

df_gaia_chandra.to_csv('tutorial04_gaia_chandra_matches.csv', index=False)
print(f"✅ Saved Gaia-Chandra catalog: tutorial04_gaia_chandra_matches.csv")

# ============================================================================
# Summary and Next Steps
# ============================================================================
print("\n" + "="*70)
print("Tutorial Complete!")
print("="*70)

print(f"""
What we learned:
---------------
1. HEALPix spatial indexing for astronomical surveys
2. Downloading HEALPix-partitioned data efficiently
3. Reading HDF5 files with h5py
4. Spatial cross-matching with astropy SkyCoord
5. Creating multi-wavelength catalogs

Key Results:
-----------
- Downloaded data for HEALPix pixel {healpix}
- Loaded {len(gaia_ra):,} Gaia, {len(sdss_ra):,} SDSS, {len(chandra_ra):,} Chandra sources
- Created {len(df_gaia_sdss)} Gaia-SDSS matches
- Created {len(df_gaia_chandra)} Gaia-Chandra matches
- Identified {len(gaia_triple_indices)} triple matches

Generated Files:
--------------
- tutorial04_crossmatch_analysis.png: Visualizations
- tutorial04_gaia_sdss_matches.csv: Optical + spectroscopy catalog
- tutorial04_gaia_chandra_matches.csv: Optical + X-ray catalog

Next Steps:
----------
1. Analyze redshift distribution of matched galaxies
2. Study X-ray to optical flux ratios
3. Identify AGN candidates (X-ray bright sources)
4. Cross-match with additional surveys (WISE, 2MASS, etc.)
5. Use matched catalogs for science analysis

Scientific Applications:
----------------------
- AGN identification and classification
- Galaxy evolution across cosmic time
- Multi-wavelength SEDs
- Stellar population studies
- Variable and transient source identification
""")

print("\n✅ Tutorial 04 Complete!")
