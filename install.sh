cd rsl_rl && pip install -e . && cd ..
cd legged_gym && pip install -e . && cd ..
# Uninstall conflicting packages first
pip uninstall -y opencv-python opencv-contrib-python opencv-python-headless
# Upgrade spacy/thinc to versions compatible with pydantic v2
pip install --upgrade "spacy>=3.5.0" "thinc>=8.1.0"
# Install packages with compatible versions
pip install "numpy==1.23.0" pydelatin "pydantic>=2.0" "wandb>=0.16.0" tqdm opencv-python ipdb pyfqmr flask dill gdown hydra-core imageio[ffmpeg] mujoco mujoco-python-viewer isaacgym-stubs pytorch-kinematics rich termcolor
pip install redis[hiredis]
pip install pyttsx3 # for voice control
cd pose && pip install -e . && cd ..
