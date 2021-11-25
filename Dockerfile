FROM nvcr.io/nvidia/tensorflow:20.03-tf2-py3

# Install other packages you want (for example: seaborn)
RUN pip install --no-cache-dir --upgrade pip && pip install --no-cache-dir --upgrade \
    matplotlib \
    pycodestyle \
    autopep8 \
    pylint \
    scikit-learn \
    seaborn \
    pandas \
    numpy \
    h5py \
    pyyaml \
    tensorflow-addons==0.9.1

RUN pip install --no-cache-dir scipy

# Install jupyter-vim-binding
# RUN pip install --no-cache-dir --upgrade pip && pip install --no-cache-dir \
# 	jupyter_contrib_nbextensions && \
#     jupyter contrib nbextension install --system && \
#     mkdir -p /usr/local/share/jupyter/nbextensions && \
#     cd /usr/local/share/jupyter/nbextensions && \
#     git clone https://github.com/lambdalisue/jupyter-vim-binding vim_binding && \
#     chmod -R go-w vim_binding && \
#     jupyter nbextension enable vim_binding/vim_binding
