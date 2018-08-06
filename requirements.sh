# This is for the server : huckleberry
# Set up conda first, by installing in home directory from ppc installer (architecture power pc )
# create a venv named 'venv'

mkdir resources/data
mkdir resources/data/data/original_paper
source activate venv
conda install gensim cython networkx biopython
pip install obonet
conda install tensorflow-gpu
conda install numpy 
conda install pandas
conda install textacy
conda install spacy
python -m spacy download en
 
