**Important:** This repository uses a Git submodule (`pytorch-image-models`). To clone with all submodules:
```bash
git clone --recursive https://github.com/edgaremy/arthropod-classifier.git
```
If you've already cloned without `--recursive`, run:
```bash
git submodule update --init --recursive
```

Create a symlink to the arthropod dataset :
```bash
ln -s path/to/dataset/ dataset
```

Setup Conda environment:
```bash
conda create -n arthropod-classifier python=3.12
conda activate arthropod-classifier
pip install torch torchvision scikit-learn pyyaml
```