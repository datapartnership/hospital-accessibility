# Start from a core stack version
FROM jupyter/scipy-notebook:6d42503c684f

USER root


RUN apt-get update \
    && apt-get install software-properties-common -y \
    && apt-get install gdal-bin -y && apt-get install libgdal-dev -y \
    && apt-get install -y libproj-dev proj-data proj-bin libgeos-dev libspatialindex-dev  \
    && apt-get install -y osmctools \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*
    

USER $NB_ID


RUN export CPLUS_INCLUDE_PATH=/usr/include/gdal && export C_INCLUDE_PATH=/usr/include/gdal

RUN pip install GDAL==$(gdal-config --version | awk -F'[.]' '{print $1"."$2}') && \
    pip install pyrasterframes==0.8.5

# Install the Dask dashboard
# RUN pip install dask_labextension ; \
#    jupyter labextension install -y --clean \
#    dask-labextension

#RUN pip install --no-cache-dir notebook==5.*

RUN pip install 'affine==2.3.0' \
    'argon2-cffi==20.1.0' \
    'attrs==19.3.0' \
    'backcall==0.2.0' \
    'bleach==3.1.5' \
    'bokeh==2.1.1' \
    'boltons' \
    'Cartopy==0.18.0' \
    'certifi==2020.6.20' \
    'cffi==1.14.1' \
    'chardet==3.0.4' \
    'click==7.1.2' \
    'click-plugins==1.1.1' \
    'cligj==0.5.0' \
    'cloudpickle==1.5.0' \
    'colorcet==2.0.2' \
    'contextily==1.0.0' \
    'cycler==0.10.0' \
    'cython' \
    'dask==2.17.2' \
    'datashader==0.11.0' \
    'datashape==0.5.2' \
    'decorator==4.4.2' \
    'defusedxml==0.6.0' \
    'descartes==1.1.0' \
    'distributed==2.22.0' \
    'entrypoints==0.3' \
    'Fiona==1.8.13.post1' \
    'fsspec==0.8.0' \
    'geographiclib==1.50' \
    'geopandas==0.8.1' \
    #'geopy==2.0.0' \ #TODO GOSTNets needs an update for this to work
    'geopy==1.22.0' \
    'geoviews==1.8.1' \
    'GOSTnets==1.0.1' \
    'HeapDict==1.0.1' \
    'holoviews==1.13.3' \
    'holoviz==0.11.6' \
    'hvplot==0.6.0' \
    'idna==2.10' \
    'ipykernel==5.3.4' \
    'ipython==7.17.0' \
    'ipython-genutils==0.2.0' \
    'jedi==0.17.2' \
    'Jinja2==2.11.2' \
    'joblib==0.16.0' \
    'jsonschema==3.2.0' \
    'jupyter-client==6.1.6' \
    'jupyter-core==4.6.3' \
    'kiwisolver==1.2.0' \
    'llvmlite==0.31.0' \
    'locket==0.2.0' \
    'Markdown==3.2.2' \
    'MarkupSafe==1.1.1' \
    'matplotlib==3.3.0' \
    'mercantile==1.1.5' \
    'mistune==0.8.4' \
    'msgpack==1.0.0' \
    'multipledispatch==0.6.0' \
    'munch==2.5.0' \
    'nbconvert==5.6.1' \
    'nbformat==5.0.7' \
    'networkx==2.4' \
    'notebook==6.1.1' \
    'numba==0.48.0' \
    'numpy==1.19.1' \
    'osmnx==0.15.1' \
    'packaging==20.4' \
    'pandas==1.1.0' \
    'pandocfilters==1.4.2' \
    'panel==0.9.7' \
    'param==1.9.3' \
    'parso==0.7.1' \
    'partd==1.1.0' \
    'pexpect==4.8.0' \
    'pickleshare==0.7.5' \
    'Pillow==7.2.0' \
    'prometheus-client==0.8.0' \
    'prompt-toolkit==3.0.5' \
    'psutil==5.7.2' \
    'ptyprocess==0.6.0' \
    'pycparser==2.20' \
    'PyCRS==1.0.2' \
    'pyct==0.4.6' \
    'Pygments==2.6.1' \
    'pyparsing==2.4.7' \
    'pyproj==2.6.1.post1' \
    'pyrsistent==0.16.0' \
    'pyshp==2.1.0' \
    'python-dateutil==2.8.1' \
    'pytz==2020.1' \
    'pyviz-comms==0.7.6' \
    'PyYAML==5.3.1' \
    'pyzmq==19.0.2' \
    'rasterio==1.1.5' \
    'requests==2.24.0' \
    'rise==5.6.1' \
    'Rtree==0.9.4' \
    'scipy==1.5.2' \
    'Send2Trash==1.5.0' \
    'Shapely==1.7.0' \
    'six==1.15.0' \
    'snuggs==1.4.7' \
    'sortedcontainers==2.2.2' \
    'tblib==1.7.0' \
    'terminado==0.8.3' \
    'testpath==0.4.4' \
    'toolz==0.10.0' \
    'tornado==6.0.4' \
    'tqdm==4.48.2' \
    'traitlets==4.3.3' \
    'typing-extensions==3.7.4.2' \
    'urllib3==1.25.10' \
    'wcwidth==0.2.5' \
    'webencodings==0.5.1' \
    'xarray==0.15.1' \
    'zict==2.0.0'

# Dask Scheduler & Bokeh ports
EXPOSE 8787
EXPOSE 8786

#ENTRYPOINT ["jupyter", "lab", "--ip=0.0.0.0", "--allow-root"]
