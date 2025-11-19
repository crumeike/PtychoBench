# PtychoBench Dataset Access

## Overview

The PtychoBench dataset contains 391 expert-annotated ptychographic reconstructions from experiments conducted at the Advanced Photon Source (APS) at Argonne National Laboratory. Due to institutional policies and data sensitivity, an example dataset is available upon request (yjiang@anl.gov). The complete dataset is shared only through formal research collaboration.

## Dataset Contents

### What's Included
- **391 ptychographic reconstruction images** (PNG format)
- **Expert annotations** for artifact detection (12 artifact types)
- **Parameter recommendations**
- **Experimental metadata** (e.g., instruments, sample type)
- **Train/test splits** (80/20 for artifact detection, filtered subset for parameter recommendation)

### Dataset Statistics
- **Artifact Detection Task**: 312 train / 79 test samples
- **Parameter Recommendation Task**: 91 train / 44 test samples
- **Total artifact instances**: 173 in test set
- **Sample types**: 18 categories (Integrated circuits, NCM batteries, Biological specimens, etc.)
- **Instruments**: 6 APS beamlines (2-IDE-XFM, Velociprobe, Bionanoprobe, etc.)

### File Structure (Once Downloaded)
```
ptychobench_dataset/
├── images/
│   ├── af707e7b-object_ph_Niter400.png
│   └── ...
├── annotations/
│   ├── artifact_detection_split.json
│       ├── train.json
|       └── test.json
│   └── parameter_recommendation_split.json
│       ├── train.json
|       └── test.json
└── README_DATASET.txt
```

### Citation Requirements
All publications, presentations, or software using PtychoBench must cite:

```bibtex
@inproceedings{umeike2025ptychobench,
  title={Adapting general-purpose foundation models for X-ray Ptychography in Low-Data Regimes},
  author={Umeike, Robinson and Getty, Neil and Yin, Xiangyu and Jiang, Yi},
  booktitle={NeurIPS 2025 Workshop on AI for Science},
  year={2025}
}
```

## Technical Support

### Dataset Issues
For questions about the dataset content, format, or annotations:
- **Email**: yjiang@anl.gov
- **Include**: Dataset version, specific samples in question, description of issue

### Code/Model Issues
For questions about the codebase, models, or reproduction:
- **Email**: crumeike@crimson.ua.edu, ngetty@anl.gov, xyin@anl.gov

## Updates and Versioning

The dataset may be updated with:
- Additional annotated samples
- Improved annotations
- Bug fixes in metadata
- New experimental conditions

**Current Version**: 1.0.0 (Released: 2025)

Check for updates:
```bash
python scripts/check_dataset_version.py
```

## Acknowledgments

When using this dataset, please acknowledge:

> "This research used data from the PtychoBench dataset, acquired at the Advanced Photon Source, a U.S. Department of Energy (DOE) Office of Science User Facility operated for the DOE Office of Science by Argonne National Laboratory under Contract No. DE-AC02-06CH11357. Dataset curation was supported by the Laboratory Directed Research and Development Program at Argonne National Laboratory under Project Number 2025-0495."

## Contact Information

**Primary Contact (Dataset Access)**:
- Yi Jiang, Ph.D.
- Beamline data Scientist
- Advanced Photon Source, Argonne National Laboratory
- Email: yjiang@anl.gov

---

*Last Updated: November 2025*
*Dataset Version: 1.0.0*
