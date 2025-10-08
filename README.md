# AST424-2025: Multimodal Universe Tutorials and Examples

Welcome to the AST424 course repository! This collection of notebooks will help you explore the Multimodal Universe dataset for your research projects.

## Quick Start

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

| Notebook | Description | Time |
|----------|-------------|------|
| [01_getting_started.ipynb](tutorials/01_getting_started.ipynb) | First steps with Multimodal Universe, loading PLAsTiCC data | 5 min |
| [02_data_types.ipynb](tutorials/02_data_types.ipynb) | Working with images, spectra, time series, and catalogs | 10 min |
| [03_cross_matching.ipynb](tutorials/03_cross_matching.ipynb) | Cross-matching Gaia and SDSS catalogs | 10 min |
| [04_multisurvey_crossmatch.ipynb](tutorials/04_multisurvey_crossmatch.ipynb) | Advanced multi-survey cross-matching (Chandra + Gaia + SDSS) | 15 min |

All tutorials are fully tested and available as both `.py` scripts and `.ipynb` notebooks.

## 🔬 Research Examples

Complete project examples you can adapt for your own research:

| Notebook | Research Area | Key Techniques |
|----------|---------------|----------------|
| [stellar_variability_tess.ipynb](examples/stellar_variability_tess.ipynb) | Variable Stars | Time series analysis, Lomb-Scargle periodograms, phase folding |
| [color_magnitude_gaia.ipynb](examples/color_magnitude_gaia.ipynb) | Stellar Populations | Parallaxes, absolute magnitudes, HR diagrams |
| [xray_sources_chandra.ipynb](examples/xray_sources_chandra.ipynb) | Multi-wavelength Astronomy | X-ray sources, optical counterparts, source classification |
| [galaxy_morphology_gz10.ipynb](examples/galaxy_morphology_gz10.ipynb) | Galaxy Classification | Random Forest, CNNs, transfer learning (17K galaxies, 10 classes) |

All examples are fully tested with real data from the Multimodal Universe dataset.

## 📊 Available Datasets

The Multimodal Universe includes data from 25+ astronomical surveys, including:

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
- **TA**: Alicia Savelli (Tuesdays 2-4 PM, CITA Idealab)
- **GitHub Issues**: Report bugs or request features

## 🙏 Acknowledgements

These tutorials and examples were developed with assistance from [Claude Code](https://claude.com/claude-code), Anthropic's AI coding assistant. Claude Code helped create well-tested, reproducible code examples and ensured all notebooks executed correctly with the Multimodal Universe dataset.

The [Multimodal Universe dataset](https://github.com/MultimodalUniverse/MultimodalUniverse) is made available under CC BY 4.0 license by the MultimodalUniverse team and hosted by the Flatiron Institute.

---

*Remember: The goal is to learn by doing. Start with the tutorials, then adapt the examples for your own research questions!*
