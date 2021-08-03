# Use:
# $ setup.sh /path/to/python3.x /path/to/virtualenv/dir/smap-l4c

virtualenv -p $1 --system-site-packages $2

source $2/bin/activate

# Install a specific GDAL version, matching that of the GDAL system installation
pip install GDAL==$(gdal-config --version)
