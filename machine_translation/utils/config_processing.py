import os
import collections

import argparse
import yaml
from omegaconf import OmegaConf


def flatten_omegaconf(
        d,
        sep='_'
):
    d = OmegaConf.to_container(d)

    obj = collections.OrderedDict()

    def recurse(t, parent_key=''):

        if isinstance(t, list):
            for i in range(len(t)):
                recurse(t[i], parent_key + sep + str(i) if parent_key else str(i))
        elif isinstance(t, dict):
            for k, v in t.items():
                recurse(v, parent_key + sep + k if parent_key else k)
        else:
            obj[parent_key] = t

    recurse(d)
    obj = {k: v for k, v in obj.items() if type(v) in [int, float]}
    # obj = {k: v for k, v in obj.items()}

    return obj


def get_config(
        default="./configs/default.yaml",
        experiment=None,
        override=None
):
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default=default, help="path to config (YAML file)")
    parser.add_argument("--exp", type=str, default=experiment, help="path to experiment config (YAML file)")
    parser.add_argument("--ovr", type=str, default=override,
                        help="path to override config (YAML file) for some fast experiments")
    parser.add_argument("-f", "--fff", help="for jupyter notebook", default="1")

    args = parser.parse_args()

    with open(args.config) as f:
        config = yaml.full_load(f)

    cfg = OmegaConf.create(config)

    if args.exp is not None:
        with open(args.exp) as f:
            exp = yaml.full_load(f)
        exp = OmegaConf.create(exp)
        cfg = OmegaConf.merge(cfg, exp)

    if args.ovr is not None:
        with open(args.ovr) as f:
            ovr = yaml.full_load(f)
        ovr = OmegaConf.create(ovr)
        cfg = OmegaConf.merge(cfg, ovr)

    hparams = flatten_omegaconf(cfg)
    return args.config, cfg, hparams


def save_config(
        cfg,
        cfg_dir,
        cfg_fname
):
    os.makedirs(cfg_dir, exist_ok=True)
    with open(os.path.join(cfg_dir, cfg_fname), 'w') as f:
        OmegaConf.save(config=cfg, f=f.name)
