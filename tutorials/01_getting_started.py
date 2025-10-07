#!/usr/bin/env python3
"""
Tutorial 1: Getting Started with the Multimodal Universe

This script introduces the Multimodal Universe dataset and demonstrates:
- How to load data from HuggingFace
- Understanding data structure
- Creating visualizations
- Saving data for offline work

Expected runtime: ~5 minutes
"""

print("="*60)
print("Tutorial 1: Getting Started with the Multimodal Universe")
print("="*60)

# ============================================================================
# Section 1: Environment Check
# ============================================================================
print("\n" + "="*60)
print("Section 1: Environment Check")
print("="*60)

import sys
print(f"Python version: {sys.version}")

# Import required packages
try:
    import numpy as np
    print(f"NumPy version: {np.__version__}")

    import matplotlib.pyplot as plt
    import matplotlib
    # Use non-interactive backend for script mode
    matplotlib.use('Agg')
    print(f"Matplotlib version: {matplotlib.__version__}")

    import pandas as pd
    print(f"Pandas version: {pd.__version__}")

    from astropy.coordinates import SkyCoord
    import astropy.units as u
    import astropy
    print(f"Astropy version: {astropy.__version__}")

    from datasets import load_dataset
    import datasets
    print(f"Datasets version: {datasets.__version__}")

    import pickle

    print("\n✅ All packages imported successfully!")

except ImportError as e:
    print(f"❌ Import error: {e}")
    print("Please install required packages:")
    print("pip install datasets numpy matplotlib pandas astropy")
    sys.exit(1)

# ============================================================================
# Section 2: Your First Dataset Load
# ============================================================================
print("\n" + "="*60)
print("Section 2: Loading PLAsTiCC Dataset")
print("="*60)

# Load PLAsTiCC dataset in streaming mode
print("Loading PLAsTiCC dataset in streaming mode...")
dataset = load_dataset(
    "MultimodalUniverse/plasticc",
    split="train",
    streaming=True
)

print("✅ Dataset loaded successfully!")
print(f"Dataset type: {type(dataset)}")

# ============================================================================
# Section 3: Exploring Data Structure
# ============================================================================
print("\n" + "="*60)
print("Section 3: Exploring Data Structure")
print("="*60)

# Get the first example
example = next(iter(dataset))

print("\nAvailable fields in the dataset:")
print("-" * 40)
for key in example.keys():
    value = example[key]
    if isinstance(value, dict):
        print(f"{key:20s} : dict with keys {list(value.keys())}")
    elif isinstance(value, np.ndarray):
        print(f"{key:20s} : array with shape {value.shape}")
    elif isinstance(value, (int, float)):
        print(f"{key:20s} : {type(value).__name__} = {value}")
    else:
        print(f"{key:20s} : {type(value).__name__}")

# Examine the lightcurve structure
print("\nLightcurve structure:")
print("-" * 40)
print(f"Object ID: {example['object_id']}")
print(f"Object Type: {example['obj_type']}")
print(f"Redshift: {example['redshift']:.4f}")

# The lightcurve is stored as a dictionary
lightcurve = example['lightcurve']
print(f"Lightcurve keys: {list(lightcurve.keys())}")

# Extract the components
times = np.array(lightcurve['time'])
bands = np.array(lightcurve['band'])
flux = np.array(lightcurve['flux'])
flux_err = np.array(lightcurve['flux_err'])

print(f"\nNumber of observations: {len(times)}")
print(f"Unique bands: {np.unique(bands)}")
print(f"Time range: {times.min():.1f} to {times.max():.1f} days")

# ============================================================================
# Section 4: Creating Visualizations
# ============================================================================
print("\n" + "="*60)
print("Section 4: Creating Visualization")
print("="*60)

# Set up the plot
fig = plt.figure(figsize=(12, 6))

# Define colors for each band
band_colors = {'u': 'purple', 'g': 'blue', 'r': 'green',
               'i': 'orange', 'z': 'red', 'Y': 'darkred'}

# Plot each band separately
for band_name in np.unique(bands):
    # Get data for this band
    mask = bands == band_name

    if band_name in band_colors:
        color = band_colors[band_name]
    else:
        color = 'gray'

    # Plot with error bars
    plt.errorbar(times[mask], flux[mask], yerr=flux_err[mask],
                fmt='o', label=f'Band {band_name}',
                color=color, alpha=0.7, markersize=4)

plt.xlabel('Time (days)', fontsize=12)
plt.ylabel('Flux', fontsize=12)
plt.title(f'PLAsTiCC Light Curve - {example["obj_type"]} (z={example["redshift"]:.3f})',
         fontsize=14)
plt.legend(loc='best')
plt.grid(True, alpha=0.3)
plt.tight_layout()

# Save the plot
output_file = 'tutorial01_lightcurve.png'
plt.savefig(output_file, dpi=100, bbox_inches='tight')
print(f"✅ Saved plot to {output_file}")
plt.close()

# ============================================================================
# Section 5: Loading Multiple Examples
# ============================================================================
print("\n" + "="*60)
print("Section 5: Loading Multiple Examples")
print("="*60)

# Load 10 examples
examples = []
dataset_iter = iter(dataset)

print("Loading 10 examples...")
for i in range(10):
    example = next(dataset_iter)
    examples.append({
        'object_id': example['object_id'],
        'obj_type': example['obj_type'],
        'redshift': example['redshift'],
        'num_observations': len(example['lightcurve']['time'])
    })
    print(f"  Loaded object {i+1}/10: ID={example['object_id']}")

# Convert to DataFrame for easy viewing
df = pd.DataFrame(examples)
print("\nSummary of loaded objects:")
print(df.to_string())

# Show statistics
print("\nObject type distribution:")
print(df['obj_type'].value_counts())

# ============================================================================
# Section 6: Loading Without Streaming
# ============================================================================
print("\n" + "="*60)
print("Section 6: Loading a Fixed Subset")
print("="*60)

# Load first 100 examples (downloads to cache)
print("Downloading first 100 examples (this may take a moment)...")
dataset_subset = load_dataset(
    "MultimodalUniverse/plasticc",
    split="train[:100]"
)

print(f"✅ Downloaded {len(dataset_subset)} examples")
print("These are now cached locally for fast access")

# You can now access by index
first_example = dataset_subset[0]
print(f"\nFirst object ID: {first_example['object_id']}")
print(f"Object type: {first_example['obj_type']}")

# Show some statistics
obj_types = [dataset_subset[i]['obj_type'] for i in range(len(dataset_subset))]
unique_types = pd.Series(obj_types).value_counts()
print("\nObject types in subset:")
print(unique_types)

# ============================================================================
# Section 7: Saving Data for Offline Work
# ============================================================================
print("\n" + "="*60)
print("Section 7: Saving Data for Offline Work")
print("="*60)

# Convert subset to pandas DataFrame
print("Creating summary DataFrame...")
data_list = []
for i in range(min(20, len(dataset_subset))):  # Save first 20
    obj = dataset_subset[i]
    data_list.append({
        'object_id': obj['object_id'],
        'obj_type': obj['obj_type'],
        'redshift': obj['redshift'],
        'hostgal_photoz': obj['hostgal_photoz'],
        'num_observations': len(obj['lightcurve']['time'])
    })

df = pd.DataFrame(data_list)

# Save to CSV
output_file = 'plasticc_sample.csv'
df.to_csv(output_file, index=False)
print(f"✅ Saved {len(df)} objects to {output_file}")

# Save first 5 light curves as pickle (preserves structure)
light_curves = []
for i in range(min(5, len(dataset_subset))):
    lc = dataset_subset[i]['lightcurve']
    light_curves.append({
        'object_id': dataset_subset[i]['object_id'],
        'lightcurve': lc
    })

with open('plasticc_lightcurves.pkl', 'wb') as f:
    pickle.dump(light_curves, f)
print(f"✅ Saved {len(light_curves)} light curves to plasticc_lightcurves.pkl")

# ============================================================================
# Summary
# ============================================================================
print("\n" + "="*60)
print("Tutorial Complete!")
print("="*60)

print("""
Summary of what we learned:
✅ Loaded data from the Multimodal Universe
✅ Explored the PLAsTiCC data structure
✅ Created and saved a visualization
✅ Loaded multiple examples and created summaries
✅ Saved data locally for offline work

Key takeaways:
1. Use streaming=True for exploration without downloading everything
2. Use split="train[:N]" to download specific number of examples
3. PLAsTiCC uses dictionary structure for light curves
4. Data can be saved as CSV (metadata) and pickle (full structure)

Available datasets to explore:
- MultimodalUniverse/plasticc    # Time series
- MultimodalUniverse/gaia        # Catalog (ra, dec lowercase)
- MultimodalUniverse/tess        # Time series (RA, DEC uppercase)
- MultimodalUniverse/gz10        # Galaxies (rgb_image, gz10_label)
- MultimodalUniverse/sdss        # Spectra
- MultimodalUniverse/chandra     # X-ray

All datasets: https://huggingface.co/MultimodalUniverse/datasets
""")

print("\n✅ Tutorial 01 script completed successfully!")
print(f"Generated files: tutorial01_lightcurve.png, plasticc_sample.csv, plasticc_lightcurves.pkl")