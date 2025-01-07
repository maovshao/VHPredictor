#Create a virtual environment 
conda create -n vhpredictor python=3.12
conda activate vhpredictor

#conda environment
conda install -c conda-forge biopython
conda install -c conda-forge tqdm
conda install pytorch torchvision torchaudio pytorch-cuda=12.1 -c pytorch -c nvidia
conda install -c conda-forge pandas
conda install -c conda-forge seaborn
conda install -c conda-forge logzero
conda install -c conda-forge scikit-learn
conda install ipykernel --update-deps --force-reinstall
pip install fair-esm
pip install logzero
pip install matplotlib-venn
