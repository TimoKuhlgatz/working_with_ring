from dataclasses import replace
import itertools
from typing import Optional
import os
import fire
import jax
import jax.numpy as jnp
import numpy as np
import ring
from ring.extras.randomize_sys import randomize_anchors
from ring.sys_composer.morph_sys import _autodetermine_new_parents
from ring.sys_composer.morph_sys import identify_system
from ring.sys_composer.morph_sys import Node
from ring.utils import dict_union
import platform


if platform.system() == "Windows":
    xml_file_path = os.path.join(
        r"C:\Users\Timo_Kuhlgatz\Seafile\Meine Bibliotheken\02_projects\04_IMU_Motion_capture\03_RING\ring_dev\archiv\train_xmls",
        "lam2.xml"
    )
else:
    xml_file_path = os.path.join(
        "/home/kuhlgatz/Projects/imu_motion_tracking/ring_dev/archiv/train_xmls/",
        "lam2.xml"
    )

with open(xml_file_path, 'r', encoding='utf-8') as f:
    next(f)
    sys_str = f.read()  # <-- This line reads the file's content

def _change_joint_type(sys: ring.System, name: str, dof: int | str):
    dof_joint_types = {
        "0": "frozen",
        "1": "rr",
        "1a": "rr",
        "1b": "rr_imp",
        "2": "rsaddle",
        "3": "spherical",
    }
    dof_joint_dampings = {
        "0": np.array([]),
        "1": np.array([3.0]),
        "1a": np.array([3.0]),
        "1b": np.array([3.0, 3.0]),
        "2": np.array([3.0, 3.0]),
        "3": np.array([5.0, 5.0, 5.0]),
    }
    dof = str(dof)
    return sys.change_joint_type(
        name, dof_joint_types[dof], new_damp=dof_joint_dampings[dof], warn=False
    )


def body_to_eps_rots(
    sys: ring.System,
    x: ring.Transform,
    sys_x: ring.System,
) -> dict[str, jax.Array]:
    # (time, nlinks, 4) -> (nlinks, time, 4)
    rots = x.rot.transpose((1, 0, 2))
    l_map = sys_x.idx_map("l")
    segments = sys.findall_segments()

    y = dict()

    def f(_, __, name: str):
        if name not in segments:
            return
        q_i = rots[l_map[name]]
        q_i = ring.maths.quat_inv(q_i)
        y[name] = q_i

    sys.scan(f, "l", sys.link_names)

    return y


def _lookup_new_index(structure: Node):
    "Returns link index in new indicies for joint params and dof"
    if not structure.parent_changed:
        return structure.link_idx_new_indices
    return structure.old_parent_new_indices


def finalize_fn_factory(sys_pred: ring.System, verbose=False):
    segs = set(sys_pred.findall_segments()) - set(
        [sys_pred.find_body_to_world(name=True)]
    )

    dof_map = dict(rr=1, rsaddle=2, spherical=3, rr_imp=1)

    def finalize_fn(key, q, x, sys: ring.System):
        anchor = sys.find_body_to_world(name=True)
        structures, *_ = identify_system(
            sys_pred,
            _autodetermine_new_parents(
                sys_pred.link_parents,
                sys_pred.name_to_idx(anchor),
            ),
        )

        X = {"dt": jnp.array(sys.dt)}
        for seg in segs:
            structure = structures[sys_pred.name_to_idx(seg)]
            i = _lookup_new_index(structure)
            if verbose:
                print(
                    f"For `{seg}` and anchor `{anchor}` use link `{sys.idx_to_name(i)}`"
                )
            X = dict_union(
                X,
                {
                    seg: dict(
                        dof=dof_map[sys.link_types[i]],
                        joint_params=sys.links[i].joint_params,
                        parent_changed=jnp.array(structure.parent_changed),
                    )
                },
            )

        y = body_to_eps_rots(sys_pred, x, sys)
        return X, y

    return finalize_fn


def _add_rom(mconfig: ring.MotionConfig) -> ring.MotionConfig:
    overwrites = dict(
        rom_halfsize=0.4,
        dpos_max=0.1,
        cor_pos_min=-0.05,
        cor_pos_max=0.05,
        cor_dpos_max=0.03,
        dang_max_free_spherical=0.8,
        t_max=5.0,
    )
    return replace(
        mconfig, joint_type_specific_overwrites=dict(cor=overwrites, free=overwrites)
    )


def main(
    size: int,  # 32 * n_mconfigs * n_gens (= n_anchors * 3**(N-1)) * X
    output_path: str,
    configs: list[str] = ["standard", "expSlow", "expFast", "hinUndHer"],
    seed: int = 1,
    anchors: Optional[list[str]] = None,
    mot_art: bool = False,
    dyn_sim: bool = False,
    sampling_rates: list[float] = [40, 60, 80, 100, 120, 140, 160, 180, 200],
    T: float = 150.0,
    dof_configuration: Optional[list[str]] = ["111"],
    embc_rom_limitation: bool = False,
    save_rcmg_to_json: Optional[str] = None,
    include_standstills_prob: float = 0.0,
):
    """
    Main function for generating motion sequences with customizable configurations.

    Parameters:
        size (int): Number of sequences to generate.
        output_path (str): Path to the folder where the generated sequences will be stored.
        configs (list[str], optional): List of MotionConfigs to use. Defaults to
            ["standard", "expSlow", "expFast", "hinUndHer"].
        seed (int, optional): Randomness seed. Defaults to 1.
        anchors (Optional[list[str]], optional): Anchors of the four-segment chain. If None,
            all segments are chosen as anchors. Defaults to None.
        mot_art (bool, optional): Whether to simulate motion artifacts (nonrigidly attached IMUs). Defaults to False.
        dyn_sim (bool, optional): Whether to perform a dynamic or only a kinematic forward simulation. Defaults to False.
        sampling_rates (list[float], optional): Sampling rates to simulate. Defaults to
            [40, 60, 80, 100, 120, 140, 160, 180, 200].
        T (float, optional): Maximum trial length in seconds. Higher sampling rates will be truncated
            to match the length corresponding to the lowest sampling rate (e.g., 40 Hz * 150s = 6000 samples).
            Defaults to 150.0.
        dof_configuration (Optional[list[str]], optional): List of DOF (Degrees of Freedom) configurations
            for the joints in the format ['111', '121', ...]. If None, all combinations of
            1D, 2D, and 3D joints are considered. Defaults to ['111'].
        embc_rom_limitation (bool, optional): If enabled then for each `MotionConfig` adds a second `MotionConfig` object
            where the global rotation is limited to stay within [-20°, 20°] from the initial random global orientation of
            the kinematic chain.
        save_rcmg_to_json (str, optional): Save a deserialised RCMG representation in a json file with name `save_rcmg_to_json`

    Returns:
        None: This function generates motion sequences and saves them to the specified output path.
    """  # noqa: E501
    sys = ring.System.create(sys_str)

    syss = []
    segs = [s for s in sys.findall_segments() if s != sys.find_body_to_world(name=True)]
    if dof_configuration is None:
        dof_combinations = list(itertools.product(*([[1, 2, 3]] * len(segs))))
    else:
        dof_combinations = [tuple((int(s) for s in comb)) for comb in dof_configuration]

    for dofs in dof_combinations:
        _sys = sys
        for seg, dof in zip(segs, dofs):
            _sys = _change_joint_type(_sys, seg, dof)
        syss.extend(randomize_anchors(_sys, anchors))

    if mot_art:
        dyn_sim = True

    configs = [replace(ring.MotionConfig.from_register(c), T=T) for c in configs]
    if embc_rom_limitation:
        _replace_rom = lambda mconfig, add: _add_rom(mconfig) if add else mconfig
        configs = [_replace_rom(c, add) for c in configs for add in [False, True]]
    if include_standstills_prob > 0.0:
        configs = [
            replace(c, include_standstills_prob=include_standstills_prob)
            for c in configs
        ]

    rcmg = ring.RCMG(
        syss,
        configs,
        add_X_imus=True,
        dynamic_simulation=dyn_sim,
        imu_motion_artifacts=mot_art,
        imu_motion_artifacts_kwargs=dict(
            prob_rigid=0.5,
            pos_min_max=0.05,
            all_imus_either_rigid_or_flex=True,
            disable_warning=True,
        ),
        randomize_joint_params=True,
        randomize_motion_artifacts=True,
        randomize_positions=True,
        randomize_hz=True,
        randomize_hz_kwargs=dict(sampling_rates=sampling_rates, add_dt=False),
        cor=True,
        finalize_fn=finalize_fn_factory(sys),
        sys_ml=sys,
    )

    if save_rcmg_to_json is not None:
        rcmg.serialise_to_json(
            ring.utils.parse_path(save_rcmg_to_json, extension="json")
        )

    rcmg.to_folder(output_path, size, seed, overwrite=False)


if __name__ == "__main__":
    fire.Fire(main)
