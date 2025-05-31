# Data Directory

This directory will contain the downloaded dataset files when you run the predictive maintenance pipeline.

## Dataset Information

- **Source**: AI4I 2020 Predictive Maintenance Dataset
- **Download**: Automatic via Kaggle API
- **License**: CC BY-NC-SA 4.0
- **Size**: ~500KB (ai4i2020.csv)

## Files Created Automatically

When you run `python predictive_maintenance.py`, the following file will be downloaded here:

- `ai4i2020.csv` - Main dataset file with sensor readings and failure labels

## Note

The actual data files are excluded from version control (see `.gitignore`) to:
- Keep repository size manageable
- Respect dataset licensing
- Ensure fresh downloads with latest versions

To get the data, simply run the main script - it will handle the download automatically if you have Kaggle API credentials configured.