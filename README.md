# AST424-2025: Multimodal Universe Tutorials and Examples

Welcome to the AST424 course repository! This collection of notebooks will help you explore the Multimodal Universe dataset for your research projects.

## Quick Start

### Option 1: Google Colab (Recommended)
1. Click on any notebook link below
2. Select "Open in Colab"
3. Run cells in order (Shift+Enter)
4. No installation required!

### Option 2: Local Jupyter
```bash
# Install required packages
pip install datasets numpy matplotlib pandas astropy scipy

# Clone this repository
git clone https://github.com/[your-username]/ast424-2025.git
cd ast424-2025

# Start Jupyter
jupyter notebook
```

## 📚 Tutorials

Start here to learn the basics:

| Notebook | Description | Time | Status |
|----------|-------------|------|--------|
| [01_getting_started.ipynb](tutorials/01_getting_started.ipynb) | First steps with Multimodal Universe, loading PLAsTiCC data | 5 min | ✅ Ready |
| [02_data_types.ipynb](tutorials/02_data_types.ipynb) | Working with images, spectra, time series, and catalogs | 10 min | ✅ Ready |
| [03_cross_matching.ipynb](tutorials/03_cross_matching.ipynb) | Cross-matching Gaia and SDSS catalogs | 10 min | ✅ Ready |

All tutorials are available as both `.py` scripts (tested) and `.ipynb` notebooks.

## 🔬 Research Examples

Complete project examples you can adapt (coming soon):

| Notebook | Research Area | Key Techniques | Status |
|----------|---------------|----------------|--------|
| stellar_variability_tess.ipynb | Variable Stars | Time series, period finding | 🚧 Planned |
| galaxy_morphology_decals.ipynb | Galaxy Classification | CNNs, transfer learning | 🚧 Planned |
| color_magnitude_gaia.ipynb | Stellar Populations | Parallaxes, CMDs | 🚧 Planned |
| structure_3d_sdss.ipynb | Large-Scale Structure | Redshifts, 3D visualization | 🚧 Planned |
| xray_optical_chandra.ipynb | Multi-wavelength | AGN classification | 🚧 Planned |

## 📊 Available Datasets

The Multimodal Universe includes data from 25+ astronomical surveys:

### Images
- **Galaxy10 DECaLS**: 17,736 labeled galaxy images
- **Legacy Survey**: Millions of galaxy images
- **HSC**: High-resolution galaxy images

### Spectra
- **SDSS**: Optical spectra of galaxies and stars
- **DESI**: Latest spectroscopic survey data
- **Gaia BP/RP**: Low-resolution stellar spectra

### Time Series
- **TESS**: Stellar light curves for exoplanet/variability studies
- **PLAsTiCC**: Simulated LSST light curves

### Catalogs
- **Gaia**: Positions, parallaxes, proper motions for 2 billion stars
- **2MASS**: Near-infrared photometry
- **Chandra**: X-ray source catalog

## 💡 Tips for Your Project

1. **Start Small**: Load just 100-1000 objects first to develop your code
2. **Use Streaming**: For large datasets, use `streaming=True` to avoid memory issues
3. **Save Intermediate Results**: Download and save subsets for offline development
4. **Cross-Match Carefully**: Use small radius (1-3 arcsec) for reliable matches
5. **Check Data Quality**: Look for NaN values and outliers before analysis

## ⚠️ Important Dataset Notes

### Field Name Variations
Different datasets use different conventions:
- **Gaia**: Uses lowercase `ra`, `dec`
- **TESS**: Uses uppercase `RA`, `DEC`
- **SDSS**: Spectrum uses `lambda` (not `wavelength`)
- **gz10**: Uses `rgb_image` (not `image`)

### Data Structures
Many fields are dictionaries:
- **Lightcurves**: Dictionary with `time`, `flux`, `flux_err`, etc.
- **Spectra**: Dictionary with `flux`, `lambda`, `ivar`, etc.
- **Astrometry**: Dictionary with `parallax`, `pmra`, `pmdec`, etc.

## 📖 How to Load Data

Basic pattern for all datasets:
```python
from datasets import load_dataset

# Load with streaming (for exploration)
dataset = load_dataset("MultimodalUniverse/[dataset_name]",
                       split="train",
                       streaming=True)

# Get one example
example = next(iter(dataset))

# Load specific number of examples
dataset = load_dataset("MultimodalUniverse/[dataset_name]",
                       split="train[:1000]")  # First 1000
```

## 🔗 Useful Resources

- [Multimodal Universe GitHub](https://github.com/MultimodalUniverse/MultimodalUniverse)
- [HuggingFace Datasets](https://huggingface.co/MultimodalUniverse)
- [Astropy Coordinates Guide](https://docs.astropy.org/en/stable/coordinates/)
- [Course Website](https://q.utoronto.ca/courses/395255)

## ⚠️ Common Issues

**Out of Memory**: Use `streaming=True` or load fewer examples

**Slow Loading**: First download is slow; data is cached for subsequent use

**Missing Values**: Many surveys have NaN values; always check your data

**Coordinate Matching**: Different surveys use different epochs; be careful with proper motions

## 📧 Support

- **Instructor**: Prof. Josh Speagle (Fridays 1-3 PM, AB 206)
- **TA**: Alicia Savelli (Tuesdays 2-4 PM, Zoom)
- **GitHub Issues**: Report bugs or request features

---

*Remember: The goal is to learn by doing. Start with the tutorials, then adapt the examples for your own research questions!*
