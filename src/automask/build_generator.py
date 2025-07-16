import os
from pathlib import Path

from hydra.utils import instantiate
import torch  

import automask

from segment_anything.build_sam import build_sam

from sam2.modeling.sam2_base import SAM2Base

if os.path.isdir(os.path.join(automask.__path__[0], "automask")):
    raise RuntimeError(
        "You're likely running Python from the parent directory of the automask repository "
        "(i.e. the directory where the automask folder is located). Please run the script from within the automask directory."
    )
    
def _load_checkpoint(model, ckpt_path):
    if ckpt_path is not None:
        sd = torch.load(ckpt_path, map_location="cpu", weights_only=True)["model"]
        missing_keys, unexpected_keys = model.load_state_dict(sd)
        if missing_keys:
            raise RuntimeError()
        if unexpected_keys:
            raise RuntimeError()

def build_generator(cfg, sam_model):
    generator = instantiate(cfg.generator, model=sam_model, _recursive_=False)
    return generator

def sam_factory(config_file, checkpoint_path, device="cuda") -> SAM2Base:

    checkpoint_path = Path(checkpoint_path)
    if not checkpoint_path.is_absolute():
        project_root = Path(os.environ.get("PROJECT_ROOT", Path.cwd()))
        checkpoint_path = project_root / checkpoint_path

    if "sam2" not in str(checkpoint_path):
        return build_sam(checkpoint=str(checkpoint_path)).to(device=device)
    else:
        print(config_file)
        model = instantiate(config_file, _recursive_=True)
        _load_checkpoint(model, str(checkpoint_path))
        model = model.to(device)
        model.eval()
        return model

