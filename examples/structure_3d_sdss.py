#!/usr/bin/env python3
"""
Research Example: 3D Mapping of Large-Scale Structure with SDSS

This example demonstrates how to:
1. Load SDSS spectroscopic data with redshifts
2. Convert redshifts to comoving distances
3. Transform sky coordinates to 3D Cartesian coordinates
4. Visualize the cosmic web (filaments, voids, clusters)
5. Identify overdense and underdense regions

Research Applications:
- Studying large-scale structure formation
- Measuring clustering statistics (correlation functions)
- Identifying galaxy clusters and voids
- Testing cosmological models
- Understanding galaxy evolution in different environments

Expected runtime: ~5 minutes
"""

print("="*70)
print("Research Example: 3D Mapping of Large-Scale Structure")
print("="*70)

# ============================================================================
# Setup and Imports
# ============================================================================
print("\n" + "="*70)
print("1. Setup and Imports")
print("="*70)

import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend for script mode
from mpl_toolkits.mplot3d import Axes3D
import pandas as pd
from datasets import load_dataset
from astropy.cosmology import FlatLambdaCDM
import astropy.units as u
from astropy.coordinates import SkyCoord
from scipy.spatial import cKDTree
import warnings
warnings.filterwarnings('ignore')

print("✅ All packages imported successfully")

# ============================================================================
# Research Motivation
# ============================================================================
print("\n" + "="*70)
print("2. Research Motivation")
print("="*70)

print("""
Large-Scale Structure and the Cosmic Web:

The distribution of galaxies in the universe is not random - galaxies are
organized into a "cosmic web" of:
- Filaments: Dense structures connecting galaxy clusters
- Walls/Sheets: Planar concentrations of galaxies
- Clusters: Massive concentrations at filament intersections
- Voids: Underdense regions with few galaxies

Why study large-scale structure?
1. Tests our cosmological model (ΛCDM)
2. Reveals dark matter distribution
3. Shows how environment affects galaxy evolution
4. Measures the growth of structure over cosmic time
5. Probes dark energy through geometric measurements

SDSS Spectroscopic Survey:
- Redshifts for millions of galaxies
- 3D positions from redshift + angular coordinates
- Traces structure out to z ~ 0.5 (billions of light years)
- Reveals the cosmic web in unprecedented detail

Our Analysis:
- Load SDSS galaxy redshifts
- Convert to comoving distances
- Create 3D maps of galaxy distribution
- Identify clusters and voids
""")

# ============================================================================
# Setup Cosmology
# ============================================================================
print("\n" + "="*70)
print("3. Setting Up Cosmology")
print("="*70)

# Use Planck 2018 cosmology (standard ΛCDM model)
cosmo = FlatLambdaCDM(H0=67.4, Om0=0.315)

print(f"""
Cosmological Parameters (Planck 2018):
- H0 = {cosmo.H0.value} km/s/Mpc (Hubble constant)
- Ωm = {cosmo.Om0} (matter density)
- ΩΛ = {cosmo.Ode0} (dark energy density)

This cosmology is used to convert redshift to comoving distance.
""")

# ============================================================================
# Load SDSS Spectroscopic Data
# ============================================================================
print("\n" + "="*70)
print("4. Loading SDSS Spectroscopic Data")
print("="*70)

print("Loading SDSS dataset...")
# Load SDSS data in streaming mode
sdss_data = load_dataset(
    "MultimodalUniverse/sdss",
    split="train",
    streaming=True
)

print("✅ SDSS dataset loaded in streaming mode")

# ============================================================================
# Collect Galaxies with Good Redshifts
# ============================================================================
print("\n" + "="*70)
print("5. Collecting Galaxies with Reliable Redshifts")
print("="*70)

print("""
Quality cuts for reliable 3D positions:
- Redshift in range 0.01 < z < 0.3 (nearby but not too nearby)
- Valid spectrum with good quality flags
- Reasonable spectral classification (galaxy, not star or QSO)
""")

galaxies = []
sdss_iter = iter(sdss_data)
n_galaxies = 5000  # Number of galaxies to collect

print(f"\nCollecting {n_galaxies} galaxies with good redshifts...")

collected = 0
attempted = 0
max_attempts = 20000  # Don't loop forever

while collected < n_galaxies and attempted < max_attempts:
    try:
        obj = next(sdss_iter)
        attempted += 1

        # Extract redshift
        z = obj.get('Z')

        # Extract coordinates
        ra = obj.get('RA')
        dec = obj.get('DEC')

        # Quality cuts
        if z is None or ra is None or dec is None:
            continue

        # Convert to float
        z = float(z)
        ra = float(ra)
        dec = float(dec)

        # Redshift range: not too close, not too far
        if z < 0.01 or z > 0.3:
            continue

        # Compute comoving distance
        dist_mpc = cosmo.comoving_distance(z).to(u.Mpc).value

        # Convert RA, Dec, Distance to Cartesian coordinates
        # Use small angle approximation for local region
        coord = SkyCoord(ra=ra*u.deg, dec=dec*u.deg, distance=dist_mpc*u.Mpc)

        # Cartesian coordinates (in Mpc)
        x = coord.cartesian.x.to(u.Mpc).value
        y = coord.cartesian.y.to(u.Mpc).value
        z_coord = coord.cartesian.z.to(u.Mpc).value

        galaxies.append({
            'object_id': obj.get('object_id'),
            'ra': ra,
            'dec': dec,
            'redshift': z,
            'distance_mpc': dist_mpc,
            'x': x,
            'y': y,
            'z': z_coord
        })

        collected += 1

        if collected % 500 == 0:
            print(f"  Collected {collected}/{n_galaxies} galaxies (checked {attempted})...")

    except StopIteration:
        print(f"  Reached end of dataset after {attempted} objects")
        break
    except Exception as e:
        continue

galaxies_df = pd.DataFrame(galaxies)
print(f"\n✅ Collected {len(galaxies_df)} galaxies with reliable redshifts")

# ============================================================================
# Data Summary
# ============================================================================
print("\n" + "="*70)
print("6. Data Summary")
print("="*70)

print(f"""
Galaxy Sample Statistics:
- Number of galaxies: {len(galaxies_df)}
- Redshift range: {galaxies_df['redshift'].min():.4f} - {galaxies_df['redshift'].max():.4f}
- Distance range: {galaxies_df['distance_mpc'].min():.1f} - {galaxies_df['distance_mpc'].max():.1f} Mpc
- RA range: {galaxies_df['ra'].min():.2f}° - {galaxies_df['ra'].max():.2f}°
- Dec range: {galaxies_df['dec'].min():.2f}° - {galaxies_df['dec'].max():.2f}°

Cartesian Coordinates (Mpc):
- X range: {galaxies_df['x'].min():.1f} - {galaxies_df['x'].max():.1f}
- Y range: {galaxies_df['y'].min():.1f} - {galaxies_df['y'].max():.1f}
- Z range: {galaxies_df['z'].min():.1f} - {galaxies_df['z'].max():.1f}

Survey volume: ~{(galaxies_df['x'].max()-galaxies_df['x'].min()) * (galaxies_df['y'].max()-galaxies_df['y'].min()) * (galaxies_df['z'].max()-galaxies_df['z'].min()) / 1e6:.2f} Gpc³
""")

# ============================================================================
# Create 3D Visualization - Scatter Plot
# ============================================================================
print("\n" + "="*70)
print("7. Creating 3D Visualization")
print("="*70)

# Create 3D scatter plot
fig = plt.figure(figsize=(12, 10))
ax = fig.add_subplot(111, projection='3d')

scatter = ax.scatter(
    galaxies_df['x'],
    galaxies_df['y'],
    galaxies_df['z'],
    c=galaxies_df['redshift'],
    s=1,
    alpha=0.5,
    cmap='viridis'
)

ax.set_xlabel('X (Mpc)', fontsize=11)
ax.set_ylabel('Y (Mpc)', fontsize=11)
ax.set_zlabel('Z (Mpc)', fontsize=11)
ax.set_title('3D Distribution of SDSS Galaxies', fontsize=14)

# Add colorbar
cbar = plt.colorbar(scatter, ax=ax, pad=0.1)
cbar.set_label('Redshift', fontsize=11)

plt.tight_layout()
plt.savefig('sdss_3d_structure.png', dpi=150, bbox_inches='tight')
print("✅ Saved 3D structure map to sdss_3d_structure.png")
plt.close()

# ============================================================================
# Slice Views (Cone Diagrams)
# ============================================================================
print("\n" + "="*70)
print("8. Creating Slice Views (Cone Diagrams)")
print("="*70)

fig, axes = plt.subplots(2, 2, figsize=(14, 12))

# Slice 1: X-Y plane (looking down on galaxy distribution)
ax = axes[0, 0]
scatter = ax.scatter(
    galaxies_df['x'],
    galaxies_df['y'],
    c=galaxies_df['redshift'],
    s=1,
    alpha=0.5,
    cmap='viridis'
)
ax.set_xlabel('X (Mpc)', fontsize=11)
ax.set_ylabel('Y (Mpc)', fontsize=11)
ax.set_title('X-Y Projection (Top View)', fontsize=12)
ax.set_aspect('equal')
ax.grid(True, alpha=0.3)
plt.colorbar(scatter, ax=ax, label='Redshift')

# Slice 2: X-Z plane (side view)
ax = axes[0, 1]
scatter = ax.scatter(
    galaxies_df['x'],
    galaxies_df['z'],
    c=galaxies_df['redshift'],
    s=1,
    alpha=0.5,
    cmap='viridis'
)
ax.set_xlabel('X (Mpc)', fontsize=11)
ax.set_ylabel('Z (Mpc)', fontsize=11)
ax.set_title('X-Z Projection (Side View)', fontsize=12)
ax.set_aspect('equal')
ax.grid(True, alpha=0.3)
plt.colorbar(scatter, ax=ax, label='Redshift')

# Slice 3: Cone diagram (RA vs distance)
ax = axes[1, 0]
scatter = ax.scatter(
    galaxies_df['ra'],
    galaxies_df['distance_mpc'],
    c=galaxies_df['dec'],
    s=2,
    alpha=0.5,
    cmap='coolwarm'
)
ax.set_xlabel('Right Ascension (degrees)', fontsize=11)
ax.set_ylabel('Comoving Distance (Mpc)', fontsize=11)
ax.set_title('Cone Diagram: RA vs Distance', fontsize=12)
ax.grid(True, alpha=0.3)
plt.colorbar(scatter, ax=ax, label='Declination (deg)')

# Slice 4: Cone diagram (Dec vs distance)
ax = axes[1, 1]
scatter = ax.scatter(
    galaxies_df['dec'],
    galaxies_df['distance_mpc'],
    c=galaxies_df['ra'],
    s=2,
    alpha=0.5,
    cmap='coolwarm'
)
ax.set_xlabel('Declination (degrees)', fontsize=11)
ax.set_ylabel('Comoving Distance (Mpc)', fontsize=11)
ax.set_title('Cone Diagram: Dec vs Distance', fontsize=12)
ax.grid(True, alpha=0.3)
plt.colorbar(scatter, ax=ax, label='RA (deg)')

plt.suptitle('SDSS Large-Scale Structure: Multiple Views', fontsize=14)
plt.tight_layout()
plt.savefig('sdss_slice_views.png', dpi=150, bbox_inches='tight')
print("✅ Saved slice views to sdss_slice_views.png")
plt.close()

# ============================================================================
# Density Analysis - Find Overdense Regions
# ============================================================================
print("\n" + "="*70)
print("9. Identifying Overdense Regions (Clusters)")
print("="*70)

print("Computing local galaxy density using nearest neighbors...")

# Create coordinate array for KD-tree
coords = np.column_stack([galaxies_df['x'], galaxies_df['y'], galaxies_df['z']])

# Build KD-tree for efficient neighbor searching
tree = cKDTree(coords)

# Count neighbors within 10 Mpc radius for each galaxy
radius = 10.0  # Mpc
neighbor_counts = []

for i, coord in enumerate(coords):
    # Find neighbors within radius
    neighbors = tree.query_ball_point(coord, radius)
    neighbor_counts.append(len(neighbors) - 1)  # Exclude self

galaxies_df['density'] = neighbor_counts

print(f"Density statistics (neighbors within {radius} Mpc):")
print(f"  Mean: {np.mean(neighbor_counts):.1f}")
print(f"  Median: {np.median(neighbor_counts):.1f}")
print(f"  Min: {np.min(neighbor_counts)}")
print(f"  Max: {np.max(neighbor_counts)}")

# Identify overdense regions (clusters)
density_threshold = np.percentile(neighbor_counts, 90)  # Top 10%
cluster_members = galaxies_df['density'] > density_threshold

print(f"\nOverdense regions (density > {density_threshold:.0f}):")
print(f"  Number of galaxies in overdense regions: {cluster_members.sum()}")
print(f"  Percentage: {100*cluster_members.sum()/len(galaxies_df):.1f}%")

# Identify underdense regions (voids)
void_threshold = np.percentile(neighbor_counts, 10)  # Bottom 10%
void_members = galaxies_df['density'] < void_threshold

print(f"\nUnderdense regions (density < {void_threshold:.0f}):")
print(f"  Number of galaxies in voids: {void_members.sum()}")
print(f"  Percentage: {100*void_members.sum()/len(galaxies_df):.1f}%")

# ============================================================================
# Visualize Density Distribution
# ============================================================================
print("\n" + "="*70)
print("10. Visualizing Density Distribution")
print("="*70)

fig, axes = plt.subplots(2, 2, figsize=(14, 12))

# Plot 1: 3D with density coloring
ax = fig.add_subplot(221, projection='3d')
scatter = ax.scatter(
    galaxies_df['x'],
    galaxies_df['y'],
    galaxies_df['z'],
    c=galaxies_df['density'],
    s=2,
    alpha=0.6,
    cmap='hot',
    vmin=0,
    vmax=np.percentile(galaxies_df['density'], 95)
)
ax.set_xlabel('X (Mpc)', fontsize=10)
ax.set_ylabel('Y (Mpc)', fontsize=10)
ax.set_zlabel('Z (Mpc)', fontsize=10)
ax.set_title('3D Structure Colored by Density', fontsize=12)
cbar = plt.colorbar(scatter, ax=ax, pad=0.1, shrink=0.8)
cbar.set_label('N (neighbors within 10 Mpc)', fontsize=9)

# Plot 2: Density histogram
ax = axes[0, 1]
ax.hist(galaxies_df['density'], bins=50, edgecolor='black', alpha=0.7)
ax.axvline(density_threshold, color='red', linestyle='--',
           label=f'Cluster threshold ({density_threshold:.0f})')
ax.axvline(void_threshold, color='blue', linestyle='--',
           label=f'Void threshold ({void_threshold:.0f})')
ax.set_xlabel('Local Density (N within 10 Mpc)', fontsize=11)
ax.set_ylabel('Number of Galaxies', fontsize=11)
ax.set_title('Density Distribution', fontsize=12)
ax.legend()
ax.grid(True, alpha=0.3)

# Plot 3: X-Y slice with density
ax = axes[1, 0]
scatter = ax.scatter(
    galaxies_df['x'],
    galaxies_df['y'],
    c=galaxies_df['density'],
    s=3,
    alpha=0.6,
    cmap='hot',
    vmin=0,
    vmax=np.percentile(galaxies_df['density'], 95)
)
ax.set_xlabel('X (Mpc)', fontsize=11)
ax.set_ylabel('Y (Mpc)', fontsize=11)
ax.set_title('X-Y Projection Colored by Density', fontsize=12)
ax.set_aspect('equal')
ax.grid(True, alpha=0.3)
cbar = plt.colorbar(scatter, ax=ax)
cbar.set_label('Density', fontsize=10)

# Plot 4: Clusters vs Voids
ax = axes[1, 1]
ax.scatter(galaxies_df[~cluster_members & ~void_members]['x'],
          galaxies_df[~cluster_members & ~void_members]['y'],
          s=1, alpha=0.2, c='gray', label='Normal')
ax.scatter(galaxies_df[cluster_members]['x'],
          galaxies_df[cluster_members]['y'],
          s=5, alpha=0.7, c='red', label='Clusters')
ax.scatter(galaxies_df[void_members]['x'],
          galaxies_df[void_members]['y'],
          s=5, alpha=0.7, c='blue', label='Voids')
ax.set_xlabel('X (Mpc)', fontsize=11)
ax.set_ylabel('Y (Mpc)', fontsize=11)
ax.set_title('Clusters and Voids in X-Y Projection', fontsize=12)
ax.set_aspect('equal')
ax.legend()
ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('sdss_density_analysis.png', dpi=150, bbox_inches='tight')
print("✅ Saved density analysis to sdss_density_analysis.png")
plt.close()

# ============================================================================
# Redshift Distribution
# ============================================================================
print("\n" + "="*70)
print("11. Redshift and Distance Distributions")
print("="*70)

fig, axes = plt.subplots(1, 2, figsize=(12, 5))

# Redshift histogram
ax = axes[0]
ax.hist(galaxies_df['redshift'], bins=50, edgecolor='black', alpha=0.7)
ax.set_xlabel('Redshift', fontsize=11)
ax.set_ylabel('Number of Galaxies', fontsize=11)
ax.set_title('Redshift Distribution', fontsize=12)
ax.grid(True, alpha=0.3)

# Distance histogram
ax = axes[1]
ax.hist(galaxies_df['distance_mpc'], bins=50, edgecolor='black', alpha=0.7)
ax.set_xlabel('Comoving Distance (Mpc)', fontsize=11)
ax.set_ylabel('Number of Galaxies', fontsize=11)
ax.set_title('Distance Distribution', fontsize=12)
ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('sdss_redshift_distribution.png', dpi=150, bbox_inches='tight')
print("✅ Saved redshift distribution to sdss_redshift_distribution.png")
plt.close()

# ============================================================================
# Save Results
# ============================================================================
print("\n" + "="*70)
print("12. Saving Results")
print("="*70)

# Save catalog
galaxies_df.to_csv('sdss_3d_catalog.csv', index=False)
print("✅ Saved 3D galaxy catalog to sdss_3d_catalog.csv")

# ============================================================================
# Project Extensions
# ============================================================================
print("\n" + "="*70)
print("13. Project Ideas and Extensions")
print("="*70)

print("""
Ideas to extend this analysis for your research project:

1. Clustering Statistics:
   - Compute 2-point correlation function ξ(r)
   - Measure clustering length r₀
   - Compare with theoretical predictions
   - Study redshift-space distortions

2. Void Analysis:
   - Identify void regions systematically
   - Measure void size distribution
   - Study void shapes and properties
   - Compare void galaxy properties with cluster galaxies

3. Filament Detection:
   - Use algorithms like DisPerSE or NEXUS
   - Trace filamentary structure
   - Measure filament properties
   - Study galaxy properties along filaments

4. Redshift Evolution:
   - Split sample into redshift bins
   - Study how clustering evolves with time
   - Compare low-z and high-z structure
   - Test structure growth predictions

5. Environmental Studies:
   - Classify galaxies by local density
   - Study star formation vs environment
   - Analyze color-density relations
   - Identify preprocessed galaxies

6. Multi-wavelength Cross-matching:
   - Cross-match with X-ray cluster catalogs
   - Find radio galaxies in clusters
   - Study AGN fraction vs environment
   - Identify cluster member galaxies

7. Cosmological Tests:
   - Measure baryon acoustic oscillations (BAO)
   - Use as standard ruler for dark energy
   - Test modified gravity theories
   - Constrain cosmological parameters
""")

# ============================================================================
# Summary
# ============================================================================
print("\n" + "="*70)
print("Analysis Complete!")
print("="*70)

print(f"""
Summary of Results:
- Analyzed {len(galaxies_df)} SDSS galaxies with spectroscopic redshifts
- Redshift range: {galaxies_df['redshift'].min():.3f} - {galaxies_df['redshift'].max():.3f}
- Distance range: {galaxies_df['distance_mpc'].min():.1f} - {galaxies_df['distance_mpc'].max():.1f} Mpc
- Survey volume: ~{(galaxies_df['x'].max()-galaxies_df['x'].min()) * (galaxies_df['y'].max()-galaxies_df['y'].min()) * (galaxies_df['z'].max()-galaxies_df['z'].min()) / 1e6:.2f} Gpc³

Structure Identification:
- Overdense regions (clusters): {cluster_members.sum()} galaxies ({100*cluster_members.sum()/len(galaxies_df):.1f}%)
- Underdense regions (voids): {void_members.sum()} galaxies ({100*void_members.sum()/len(galaxies_df):.1f}%)
- Mean local density: {np.mean(neighbor_counts):.1f} neighbors within {radius} Mpc

Generated Files:
- sdss_3d_structure.png: 3D visualization of galaxy distribution
- sdss_slice_views.png: Multiple 2D projections (cone diagrams)
- sdss_density_analysis.png: Density map and cluster/void identification
- sdss_redshift_distribution.png: Redshift and distance distributions
- sdss_3d_catalog.csv: Complete catalog with 3D positions and densities

Next Steps:
1. Increase sample size for better statistics
2. Apply clustering algorithms to identify structures
3. Cross-match with cluster catalogs
4. Study environmental effects on galaxy properties
5. Compute correlation functions
""")

print("\n✅ 3D Structure Mapping Complete!")
