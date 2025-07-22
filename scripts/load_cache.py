print('Loading cache to ~/.boltz')

from pathlib import Path
import urllib.request
import tarfile

cache_dir = Path("~/.boltz").expanduser()
cache_dir.mkdir(parents=True, exist_ok=True)
checkpoint_path = cache_dir / "boltz2_aff.ckpt"
mol_tar_path = cache_dir / "mols.tar"
mol_dir = cache_dir / "mols"
ccd_path = cache_dir / "ccd.pkl"

if not mol_tar_path.exists():
    urllib.request.urlretrieve("https://huggingface.co/boltz-community/boltz-2/resolve/main/mols.tar", str(mol_tar_path))

if not mol_dir.exists():
    with tarfile.open(mol_tar_path, "r") as tar:
        tar.extractall(cache_dir)

if not checkpoint_path.exists():
    urllib.request.urlretrieve("https://huggingface.co/boltz-community/boltz-2/resolve/main/boltz2_aff.ckpt", str(checkpoint_path))

if not ccd_path.exists():
    urllib.request.urlretrieve("https://huggingface.co/boltz-community/boltz-1/resolve/main/ccd.pkl", str(ccd_path))

structure_checkpoint_path = cache_dir / "boltz2_conf.ckpt"

if not structure_checkpoint_path.exists():
    urllib.request.urlretrieve(
        "https://huggingface.co/boltz-community/boltz-2/resolve/main/boltz2_conf.ckpt",
        str(structure_checkpoint_path)
    )

print('Done.')
