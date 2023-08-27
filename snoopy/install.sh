#GPU
wget https://repo.anaconda.com/miniconda/Miniconda3-py310_23.5.2-0-Linux-x86_64.sh
curl -o miniconda.sh -O https://repo.anaconda.com/miniconda/Miniconda3-py310_23.5.2-0-Linux-x86_64.sh && bash miniconda.sh -bfp /usr/local && rm miniconda.sh
bash Miniconda3-py310_23.5.2-0-Linux-x86_64.sh -u
git clone https://github.com/easeml/snoopy.git
# -------------
conda create --name snoopy python=3.7
conda activate snoopy
conda install -c pytorch pytorch torchvision cudatoolkit=10.1
conda install -c conda-forge spacy ftfy matplotlib colorlog scikit-learn seaborn
conda install -c conda-forge notebook iprogress ipywidgets==7.4.2
conda install -c conda-forge nb_conda_kernels
#TF must be installed via pip, because TF 2.2 is not available via conda
#TF 2.2 is needed, because text cannot be fed to TF Hub models if TF 2.1 is used
pip install --no-cache-dir tensorflow tensorflow-addons tfds-nightly tensorflow-hub transformers efficientnet_pytorch

# -------
conda create --name snoopy python=3.7 -y
conda activate snoopy
echo "conda activated snoopy-cpu"
echo "Installing batch 1/4 ..."
conda install -c pytorch pytorch torchvision cudatoolkit=10.1 -y
echo "Installing batch 2/4 ..."
conda install -c conda-forge spacy ftfy matplotlib colorlog scikit-learn seaborn -y
echo "Installing batch 3/4 ..."
conda install -c conda-forge notebook iprogress ipywidgets==7.4.2 -y
echo "Installing batch 4/4 ..."
conda install -c conda-forge nb_conda_kernels -y
#TF must be installed via pip, because TF 2.2 is not available via conda
#TF 2.2 is needed, because text cannot be fed to TF Hub models if TF 2.1 is used
echo "Final pip installs ..."
pip install --no-cache-dir tensorflow tensorflow-addons tfds-nightly tensorflow-hub transformers efficientnet_pytorch