# PtychoBench Dataset Access

## Overview

The PtychoBench dataset contains 391 expert-annotated ptychographic reconstructions from experiments conducted at the Advanced Photon Source (APS) at Argonne National Laboratory. Due to institutional policies and data sensitivity, the dataset is available upon request and subject to approval.

## Dataset Contents

### What's Included
- **391 ptychographic reconstruction images** (PNG format)
- **Expert annotations** for artifact detection (12 artifact types)
- **Parameter recommendations** for 135 samples
- **Experimental metadata** (beam type, instrument, sample type, reconstruction parameters)
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

## Request Process

### Eligibility
The dataset is available to:
- Academic researchers at accredited institutions
- Scientists at national laboratories
- Industry researchers with legitimate scientific purposes

### How to Request Access

#### Step 1: Prepare Your Request
Create an email with the following information:

**Subject**: PtychoBench Dataset Access Request

**Include**:
1. **Your Information**:
   - Full name
   - Title/Position
   - Institution/Organization
   - Email address

2. **Research Purpose**:
   - Brief description of your research project (Less than 100 words)
   - Intended use of the PtychoBench dataset

3. **Data Use Commitment**:
   - Confirmation that you will comply with the data use agreement
   - Confirmation that you will cite the dataset properly
   - Confirmation that you will not redistribute the dataset

#### Step 2: Send Request
**Email to**: yjiang@anl.gov

**Subject**: PtychoBench Dataset Access Request

**CC**: ngetty@anl.gov, xyin@anl.gov

#### Step 3: Approval Process
1. **Initial Review** (1-2 weeks): Your request will be reviewed by the dataset custodians
2. **Institutional Approval** (1-2 weeks): May require approval from Argonne National Laboratory
3. **Data Use Agreement** (1 week): You will receive a data use agreement to sign
4. **Dataset Delivery** (1-2 days): Upon signed agreement, you will receive download instructions

**Total Expected Timeline**: 3-6 weeks from initial request

## Data Use Agreement Terms

By requesting access, you agree to:

### Permitted Uses
✅ Use for academic research and education

✅ Use for developing and evaluating machine learning models

✅ Use for publications in peer-reviewed venues

✅ Use for presentations at scientific conferences

### Prohibited Uses
❌ Commercial use without explicit written permission

❌ Redistribution or sharing with third parties

❌ Using data in ways that could identify individual beamline users

❌ Reverse-engineering experimental setups for competitive purposes


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

### Data Handling
- Store dataset securely with appropriate access controls
- Delete dataset upon completion of research or upon request
- Acknowledge Advanced Photon Source and DOE support

## Alternative: Synthetic Data

If you want to test the codebase before receiving the full dataset, we provide:
- **Sample images** (10 examples in `data/samples/`)
- **Synthetic annotations** for testing code
- **Data format specifications** for compatibility

Download sample data:
```bash
python scripts/download_samples.py
```

## Technical Support

### Dataset Issues
For questions about the dataset content, format, or annotations:
- **Email**: yjiang@anl.gov
- **Include**: Dataset version, specific samples in question, description of issue

### Code/Model Issues
For questions about the codebase, models, or reproduction:
- **GitHub Issues**: https://github.com/[username]/ptychobench/issues
- **Email**: crumeike@crimson.ua.edu, ngetty@anl.gov, xyin@anl.gov

### Access/Legal Issues
For questions about data use agreements or institutional approval:
- **Email**: yjiang@anl.gov
- **Subject**: "PtychoBench Access - Legal/Institutional Question"

## Frequently Asked Questions

### Q: How long does the approval process take?
**A**: Typically 1-2 weeks from initial request to dataset delivery.

### Q: What if I need data before my request is approved?
**A**: Use our synthetic sample data (`scripts/download_samples.py`) to test your code.

### Q: Can I publish images from the dataset?
**A**: Yes, with proper citation.

### Q: Is there a fee for accessing the dataset?
**A**: No, the dataset is provided free of charge for academic research.

### Q: Can industry researchers access the dataset?
**A**: Yes, but commercial use requires additional approval and potentially a licensing agreement.

### Q: What format are the images?
**A**: PNG format, with labels stored in JSON files.

### Q: Can I contribute new annotations to the dataset?
**A**: Yes! We welcome contributions. Contact yjiang@anl.gov to discuss.

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
- Beamline Scientist
- Advanced Photon Source, Argonne National Laboratory
- Email: yjiang@anl.gov

---

*Last Updated: November 2025*
*Dataset Version: 1.0.0*
