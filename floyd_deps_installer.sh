# sym link to dataset:
ln -s /data .

# install ml_utils
git clone https://github.com/gjeusel/ml_utils
cd ml_utils
pip install -e .

# Update torchvision without uninstall old one (else error):
pip install --ignore-installed --no-dependencies torchvision==0.2.0
