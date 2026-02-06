import os.path
import time
import json
from dataclasses import replace
from typing import Optional
import jax
import warnings
import pickle
from benchmark import benchmark as benchmark_fn
from benchmark import IMTP
import fire
import jax.numpy as jnp
import numpy as np
import optax
import qmt
import ring
from ring import maths
from ring import ml
from ring.extras.dataloader_torch import dataset_to_generator
from ring.extras.dataloader_torch import dataset_to_Xy
from ring.extras.dataloader_torch import FolderOfFilesDataset
from ring.extras.dataloader_torch import ShuffledDataset
import torch
from torch.utils.data import random_split
import wandb


def _cb_metrices_factory(warmup: int = 0):
    """
    Creates a dictionary of metric functions for the evaluation callback.

    Calculates the Mean Absolute Error (MAE) and inclination error in degrees.
    The first `warmup` time steps are ignored to allow the RNN to settle.

    Args:
        warmup (int): Number of initial time steps to ignore in each sequence.

    Returns:
        dict: A dictionary mapping names (str) to lambda functions.
    """
    return dict(
        mae_deg=lambda q, qhat: jnp.rad2deg(
            jnp.mean(maths.angle_error(q, qhat)[:, warmup:])
        ),
        incl_deg=lambda q, qhat: jnp.rad2deg(
            jnp.mean(maths.inclination_loss(q, qhat)[:, warmup:])
        ),
    )

def _params(unique_id: str = ring.ml.unique_id()) -> str:
    """
    Generates the file path for saving the optimized network parameters (weights).

    Automatically distinguishes between cluster environments and the local home directory.

    Args:
        unique_id (str): A unique identifier for the training run.

    Returns:
        str: Absolute path to the .pickle file for the parameters.
    """
    home = "/bigwork/nhkbkuhl/" if ring.ml.on_cluster() else os.getcwd()
    return os.path.join(home,f"params/{unique_id}.pickle")

def _model(unique_id: str = ring.ml.unique_id()) -> str:
    """
    Generates the file path for saving the entire model object.

    Args:
        unique_id (str): A unique identifier for the training run.

    Returns:
        str: Absolute path to the .pickle file for the model.
    """
    home = "/bigwork/nhkbkuhl/" if ring.ml.on_cluster() else os.getcwd()
    return os.path.join(home,f"models/{unique_id}.pickle")

def _checkpoints(unique_id: Optional[str] = None) -> str:
    """
    Generates the path for checkpoints (to resume training after a crash).

    Args:
        unique_id (str, optional): ID of a specific checkpoint. If None,
            returns the path to the checkpoint folder.

    Returns:
        str: Path to the checkpoint folder or a specific file.
    """
    home = "/bigwork/nhkbkuhl/" if ring.ml.on_cluster() else os.getcwd()
    if unique_id is not None:
        return os.path.join(home,f"ring_checkpoints/{unique_id}.pickle")
    else:
        return os.path.join(home,"ring_checkpoints")

class DumpModelCallback(ring.ml.training_loop.TrainingLoopCallback):
    """
    Callback that saves the trained model (architecture + weights) to disk.
    """
    def __init__(
        self,
        path: str,
        ringnet: ring.ml.ringnet.RING,
        overwrite: bool = False,
        dump_every: Optional[int] = None,
    ):
        """
        Initializes the callback.

        Args:
            path (str): Destination path for the save file.
            ringnet (ring.ml.ringnet.RING): The network object to save.
            overwrite (bool): Whether to overwrite an existing file.
            dump_every (int, optional): If set, the model is saved every X episodes.
        """
        self.path = ring.utils.parse_path(
            path,
            extension="pickle",
            file_exists_ok=overwrite,
        )
        self.ringnet = ringnet.unwrapped_deep
        self.params = None
        self.dump_every = dump_every

    def after_training_step(
        self, i_episode, metrices, params, grads, sample_eval, loggers, opt_state
    ):
        """
        Called after each training step. Periodically saves the model if configured.
        """
        self.params = params
        if self.dump_every is not None and ((i_episode % self.dump_every) == 0):
            self.close()

    def close(self):
        """
        Finalizes the saving process. Injects the current parameters into the model object
        and writes it to disk using pickle.
        """
        if self.params is not None:
            self.ringnet.params = self.params
            ring.utils.pickle_save(self.ringnet.nojit(), self.path, overwrite=True)

def act_fn_rnno(X):
    """
    Identity function serving as a placeholder for the RNNO activation.

    In this configuration, non-linearity is typically handled within the
    RNNO cells or by other layers.

    Args:
        X (jax.numpy.ndarray): Input tensor.

    Returns:
        jax.numpy.ndarray: Unmodified input tensor.
    """
    return X

def _make_net(lam, warmstart, rnn_w, rnn_d, lin_w, lin_d, layernorm, celltype, rnno):
    """
    Factory function to create and configure the neural network.

    Dynamically calculates the output dimension based on the length of the kinematic chain
    and configures either a standard RNN or an RNNO (Recursive Neural Network Operator).

    Args:
        lam (list): Topology list of the graph (e.g., [-1, 0]).
        warmstart (str): Path to pre-trained weights (or None).
        rnn_w (int): Width (hidden size) of the RNN layers.
        rnn_d (int): Depth (number) of the RNN layers.
        lin_w (int): Width of the linear layers.
        lin_d (int): Depth of the linear layers.
        layernorm (bool): Whether to use layer normalization.
        celltype (str): Type of RNN cell ('gru' or 'lstm').
        rnno (bool): If True, the RNNO architecture is used.

    Returns:
        Callable: A function that expects initialized (X, y) inputs and
                  returns the ready-to-use `ring.ml.RING` model.
    """
    dry_run = not ring.ml.on_cluster()
    if dry_run:
        rnn_w = 10
        rnn_d = 1
        lin_w = 10
        lin_d = 0

    # Output Dimension berechnen: Anzahl Segmente * 4 (Quaternion)
    # Bei lam=[-1, 0] ist len(lam) = 2 -> output_dim = 8
    output_dim = len(lam) * 4

    if rnno:
        kwargs = {
            "forward_factory": ring.ml.rnno_v1.rnno_v1_forward_factory,
            "rnn_layers": [rnn_w] * rnn_d,
            "linear_layers": [lin_w] * lin_d,
            "act_fn_rnn": act_fn_rnno,
            "output_dim": output_dim,  # HIER GEÄNDERT: Dynamisch statt fix 16
        }
    else:
        kwargs = {
            "hidden_state_dim": rnn_w,
            "stack_rnn_cells": rnn_d,
            "message_dim": lin_w,
            "send_message_n_layers": lin_d,
            # Hier müsste ggf. auch output_dim übergeben werden, falls RING genutzt wird
        }

    net = ring.ml.RING(
        params=_params(hex(warmstart)) if warmstart else None,
        celltype=celltype,
        lam=lam,
        layernorm=layernorm,
        **kwargs,
    )
    if rnno:
        net = ring.ml.base.NoGraph_FilterWrapper(net, quat_normalize=True)

    return net

def _loss_fn_ring_factory(lam):
    """
    Creates the loss function for training, adapted to the graph topology.

    The function distinguishes between:
    1. Root Segment (Parent -1): Uses `inclination_loss` (inclination relative to gravity),
       as absolute yaw is unobservable without a magnetometer.
    2. Joint Segments: Uses `angle_error` (geodesic distance on the sphere),
       as the relative rotation is fully observable.

    Args:
        lam (list): The list of parent indices (graph topology).

    Returns:
        Callable: A loss function `fn(y_hat, y) -> scalar`.
    """
    def _loss_fn_ring(q, qhat):
        "T, N, F -> Scalar"
        loss = jnp.array(0.0)
        for i, p in enumerate(lam):
            if p == -1:
                loss += jnp.mean(ring.maths.inclination_loss(q[:, i], qhat[:, i]) ** 2)
            else:
                loss += jnp.mean(ring.maths.angle_error(q[:, i], qhat[:, i]) ** 2)
        return loss / len(lam)

    return _loss_fn_ring


class Transform2Seg:
    """
    Transforms raw data of a 2-segment chain for RNNO training.

    Performs the following steps:
    1. Data Augmentation (random rotation of the entire system).
    2. Extraction and normalization of sensor data (Acc, Gyr).
    3. Integration of the time step (dt) as an additional feature.
    4. Calculation of Ground Truth:
       - Absolute orientation for the base segment.
       - Relative orientation (joint angle) for the child segment.
    """

    def __init__(self, imtp: IMTP, rand_ori: bool = True):
        """
        Initializes the transformation.

        Args:
            imtp (IMTP): Configuration object containing scaling factors and slices.
            rand_ori (bool): If True, applies data augmentation (random rotation).
        """
        self.imtp = imtp
        self.rand_ori = rand_ori
        self.link_names = ["seg3_2Seg", "seg4_2Seg"]
        # HILFSVARIABLE FÜR DEBUG PRINTS (nur 1x drucken)
        self._debug_printed = False
    def __call__(self, element):
        """
        Applies the transformation to a single data sequence.

        Args:
            element (tuple): A tuple (X_dict, Y_dict) from the dataset.

        Returns:
            tuple: (X, Y) tensors with shapes (Time, N_Links, Features) and (Time, N_Links, 4) respectively.
        """
        # element ist (X_dict, Y_dict) einer einzelnen Sequenz
        X_in_orig, Y_in_orig = element

        # # --- SPION START ---
        # if not self._debug_printed:
        #     print("\n>>> SPION IN TRANSFORM2SEG <<<")
        #     print(f"Keys in X: {X_in_orig.keys()}")
        #     print(f"Keys in Y: {Y_in_orig.keys()}")
        #
        #     if 'dt' in X_in_orig:
        #         print(f"dt gefunden: {X_in_orig['dt']}")
        #     else:
        #         print("dt NICHT gefunden! Nutze Fallback.")
        #
        #     # Erster Acc Wert checken
        #     acc_first = X_in_orig['seg3_2Seg']['acc'][0]
        #     gyr_first = X_in_orig['seg3_2Seg']['gyr'][0]
        #     print(f"Raw Acc (Zeile 0): {acc_first}")
        #     print(f"Raw Gyr (Zeile 0): {gyr_first}")
        #     print(f"Rand Ori aktiv: {self.rand_ori}")
        #     print(">>> SPION ENDE <<<\n")
        #     self._debug_printed = True
        # # --- SPION ENDE ---

        # --- Augmentation: Random Orientation ---
        if self.rand_ori:
            qrand = qmt.randomQuat()
            qrand_inv = qmt.qinv(qrand)
        else:
            qrand = None
        # implement random quaternion flip
        current_link_names = self.link_names.copy()
        if self.rand_ori and (np.random.random() < 0.5):
            current_link_names.reverse()

        imtp = self.imtp
        slices = imtp.getSlices()

        # Annahme: Alle Sensoren haben gleiche Länge T
        T = X_in_orig[current_link_names[0]]["acc"].shape[0]

        # Arrays initialisieren
        # X: (Features, N_Links, Time)
        # imtp.getF() sollte 7 sein (3 Acc + 3 Gyr + 1 dt)
        X = np.zeros((imtp.getF(), 2, T))
        Y = np.zeros((2, T, 4))

        qs_global = []

        # --- DT Extraktion ---
        # dt ist meist im X-Dictionary gespeichert oder kommt aus Config
        # Wir nehmen das dt des ersten Segments (sollte für alle gleich sein)
        # Wenn 'dt' im Dict ist, nutzen wir es, sonst imtp.Ts
        if 'dt' in X_in_orig:
            dt_val = X_in_orig['dt']
        elif 'seg3_2Seg' in X_in_orig and 'dt' in X_in_orig['seg3_2Seg']:
            # Manche Formate speichern dt im Sub-Dict
            dt_val = X_in_orig['seg3_2Seg']['dt']
        else:
            dt_val = imtp.Ts

        # dt Skalierung
        dt_norm = dt_val / imtp.scale_dt

        for i, name in enumerate(current_link_names):
            # --- INPUTS ---
            acc = X_in_orig[name]["acc"]
            gyr = X_in_orig[name]["gyr"]

            if self.rand_ori:
                acc = qmt.rotate(qrand, acc)
                gyr = qmt.rotate(qrand, gyr)

            # 1. Acc & Gyr füllen
            X[slices["acc"], i] = (acc.T / imtp.scale_acc)
            X[slices["gyr"], i] = (gyr.T / imtp.scale_gyr)

            # 2. DT füllen (wenn konfiguriert)
            if imtp.dt:
                # slices["dt"] gibt den Index für dt zurück (meist Index 6)
                X[slices["dt"], i] = dt_norm

            # --- TARGETS ---
            q_global = Y_in_orig[name]
            if self.rand_ori:
                q_global = qmt.qmult(q_global, qrand_inv)
            qs_global.append(q_global)

        # --- TARGET BERECHNUNG ---
        # Index 0 (Seg3, Root): Absolute Orientierung
        Y[0] = qs_global[0]
        # Index 1 (Seg4, Child): Relative Orientierung zu Seg3
        Y[1] = qmt.qrel(qs_global[0], qs_global[1])

        # Transponieren zu (Time, N_Links, Features) für das Netz
        return X.transpose((2, 1, 0)), Y.transpose((1, 0, 2))


class TransformedDataset(torch.utils.data.Dataset):
    """
    A wrapper for PyTorch Datasets that applies a transformation function 'on-the-fly'.

    This is necessary to ensure that Data Augmentation (like random rotation) is
    re-calculated at each access, rather than just once during loading.
    """

    def __init__(self, dataset, transform):
        """
        Args:
            dataset: The underlying dataset (e.g., FolderOfFilesDataset).
            transform: The transformation function (e.g., Transform2Seg).
        """
        self.dataset = dataset
        self.transform = transform

    def __len__(self):
        """Returns the number of sequences in the dataset."""
        return len(self.dataset)

    def __getitem__(self, idx):
        """
        Loads the element at index `idx` and applies the transformation.
        """
        return self.transform(self.dataset[idx])


class WandBEpochLogger(ring.ml.training_loop.TrainingLoopCallback):
    def __init__(self, steps_per_epoch):
        self.steps_per_epoch = steps_per_epoch

    def after_training_step(self, i_episode, metrices, params, grads, sample_eval, loggers, opt_state):
        # i_episode ist in ring der aktuelle Step-Counter
        current_epoch = i_episode / self.steps_per_epoch

        # Wir loggen 'epoch' an WandB.
        # commit=False ist wichtig, damit es zum aktuellen Step hinzugefügt wird
        # und keinen neuen Step erzwingt.
        if wandb.run is not None:
            wandb.log({"epoch": current_epoch}, commit=False)


def main(
        path_lam,  # path to the two-seg data
        bs: int,
        epochs: int,
        use_wandb: bool = False,
        wandb_project: str = "RING_2Seg",
        wandb_name: str = None,
        warmstart: str = None,
        seed: int = 1,
        lr: float = 1e-3,
        tbp: int = 1000,
        n_val: int = 256,
        rnno: bool = True,  # Standardmäßig True für diesen Anwendungsfall
        rnn_w: int = 400,
        rnn_d: int = 2,
        lin_w: int = 200,
        lin_d: int = 0,
        layernorm: bool = False,
        celltype: str = "gru",
        skip_first: bool = False,
        grad_accu: int = 1,
        rand_ori: bool = True,  # Empfohlen: True
	num_workers: int = 8,
        disable_dump_model=False,
        disable_checkpoint_model=False,
        disable_save_params=False,
        debug: bool = False,
        # Drop Params werden hier ignoriert oder können für Transform genutzt werden
):
    """
    Main function to train an RNNO on 2-segment chains.

    Executes the complete training process:
    1. Configuration of graph topology (World -> Seg3 -> Seg4).
    2. Loading and preparing data from the specified path.
    3. Initialization of the network and optimizer.
    4. Starting the training loop with logging (WandB) and checkpointing.

    Args:
        path_lam (str): Path to the folder containing training data (.pickle files).
        bs (int): Batch Size.
        epochs (int): Number of training epochs. #todo: Kontrollieren, ob das die Epochen oder die Update-Schritte sind
        use_wandb (bool): Whether to use Weights & Biases for logging.
        wandb_project (str): Name of the WandB project.
        wandb_name (str): Name of the run in WandB.
        warmstart (str): Path to a .pickle file for transfer learning / resuming.
        seed (int): Random seed for reproducibility.
        lr (float): Learning rate for the Adam optimizer.
        tbp (int): Truncated Backpropagation Through Time (chunk length).
        n_val (int): Number of files reserved for validation.
        rnno (bool): Whether to use RNNO (True) or classic RING (False).
        rnn_w (int): Width of RNN layers.
        rnn_d (int): Number of RNN layers.
        lin_w (int): Width of linear layers.
        lin_d (int): Number of linear layers.
        layernorm (bool): Use layer normalization.
        celltype (str): 'gru' or 'lstm'.
        skip_first (bool): Skips the first batch (due to RNN warmup).
        grad_accu (int): Gradient accumulation steps (for larger effective batch size).
        rand_ori (bool): Enables random rotations as data augmentation.
        disable_dump_model (bool): Disables saving the model at the end.
        disable_checkpoint_model (bool): Disables periodic checkpoints.
        disable_save_params (bool): Disables saving the pure parameters.
        debug (bool): Enables debug mode.
    """
    config_dict = locals().copy()

    np.random.seed(seed)

    if use_wandb:
        unique_id = ring.ml.unique_id()
        wandb.init(project=wandb_project, config=locals(), name=wandb_name)

    json_path = _model().split('.')[0]+"_config.json"
    folder_path = os.path.dirname(json_path)
    os.makedirs(folder_path, exist_ok=True)
    with open(json_path, 'w') as f:
        json.dump(config_dict, f, indent=4)

    print(f"Configuration file was saved under: {json_path}")


    # 1. GRAPH DEFINITION
    lam = [-1, 0]
    link_names = ["seg3_2Seg", "seg4_2Seg"]

    # 2. IMTP
    imtp = IMTP(
        dt=True,
        scale_acc=9.81,
        scale_gyr=2.2,
        scale_dt=0.01,
        segments=None,
        sparse=False,
        joint_axes_1d=False, joint_axes_1d_field=False,
        joint_axes_2d=False, joint_axes_2d_field=False,
        dof=False, dof_field=False,
        scale_ja=0.33,
    )
    '''
    When all Flags are set to True, the output dimension of the tensors are: (Batch-Size, Timesteps, #Segments, 19 Features)
    Those 19 Features are: acc[0:3], gyr[3:6], joint_axes_1d [6:9], joint_axes_2d [9:15], dof [15:18], dt[18]. If they 
    the flag is set, but the values are not present, they will be filled with zeros.
    '''

    # 3. DATEN LADEN
    ds_folder = FolderOfFilesDataset(path_lam)
    ds_raw = ShuffledDataset(ds_folder)

    if len(ds_raw) <= n_val:
        print(f"WARNUNG: Zu wenige Dateien ({len(ds_raw)}) für n_val={n_val}.")
        n_val = max(1, int(len(ds_raw) * 0.2))
        print(f"Setze n_val automatisch auf {n_val}")

    trafo = Transform2Seg(imtp, rand_ori=rand_ori)
    ds = TransformedDataset(ds_raw, trafo)
    ds_train, ds_val = random_split(ds, [len(ds) - n_val, n_val])

    # --- HIER: BERECHNUNG DER SCHRITTE (EPOCHS -> STEPS) ---
    try:
        n_train_sequences = len(ds_train)

        # Annahme aus dem Paper/Dataset: 60 Sekunden à 100 Hz = 6000 Samples
        sequence_length = 6000

        # Wie viele Updates pro Sequenz? (Truncated BPTT zerteilt die Sequenz)
        updates_per_sequence = sequence_length // tbp

        # Wie viele Batches pro Epoche?
        batches_per_epoch = n_train_sequences // bs

        # Gesamtzahl der Updates (das, was ring als 'episodes' will)
        total_steps = int(epochs * batches_per_epoch * updates_per_sequence)
        steps_per_epoch = int(batches_per_epoch * updates_per_sequence)
        print(f"\n--- TRAINING KALKULATION ---")
        print(f"Gewünschte Epochen: {epochs}")
        print(f"Trainings-Sequenzen: {n_train_sequences}")
        print(f"Batch Size: {bs} -> {batches_per_epoch} Batches/Epoche")
        print(f"Updates pro Sequenz: {updates_per_sequence} (bei tbp={tbp})")
        print(f"--> BERECHNETE TOTAL STEPS: {total_steps}")
        print(f"----------------------------\n")

    except Exception as e:
        print(f"FEHLER bei der Schritt-Berechnung: {e}")
        # Fallback, damit es nicht crasht, falls Länge unbekannt
        total_steps = epochs * 1000
        steps_per_epoch = 1000
        # -------------------------------------------------------
    shuffle_train = False if debug else True
    # Generator
    generator = dataset_to_generator(
        ds_train,
        batch_size=bs,
        seed=seed,
        drop_last=True,
        shuffle=shuffle_train,
        # num_workers=None if ring.ml.on_cluster() else 0,
	num_workers=num_workers,
    )

    # Validierung
    X_val, y_val = dataset_to_Xy(ds_val)

    # ------------------- FINALER DEBUG BLOCK -------------------
    if debug:
        print("\n=== DEBUG MODUS AKTIV ===")
        print(f"Optionen: rand_ori={rand_ori}, shuffle_train={shuffle_train}")
        debug_path = os.path.join(os.path.dirname(path_lam), 'ring_debug')
        os.makedirs(debug_path, exist_ok=True)
        print(f"Speichere Tensoren in: {debug_path}")
        # 1. Dateinamen rekonstruieren
        # ds_train ist ein 'Subset'. Es hat ein Attribut .indices, das auf das Original zeigt.
        # Da wir shuffle=False gesetzt haben, entspricht Batch 0 genau den ersten 'bs' Indizes von ds_train.
        print("\n--- DATEIEN IM ERSTEN BATCH (Training) ---")
        try:
            # Zugriff auf die Indizes, die für Training ausgewählt wurden
            train_indices = ds_train.indices

            # Wir nehmen nur die für den ersten Batch
            batch_indices = train_indices[:bs]

            # Wir müssen durch die Wrapper zum Original FolderDataset
            # ds (Transformed) -> ds_raw (Shuffled)
            ds_shuffled = ds.dataset

            # Mapping des ShuffledDataset auflösen
            # ShuffledDataset hat meist .indices oder .permutation, um die Ordnung zu ändern
            if hasattr(ds_shuffled, 'indices'):
                # Das ist die Permutation vom ShuffledDataset
                shuffled_map = ds_shuffled.indices
            else:
                # Fallback
                shuffled_map = np.arange(len(ds_folder))

            # Jetzt die echten Dateinamen holen
            file_names = []
            for i, idx_in_subset in enumerate(batch_indices):
                # idx_in_subset: Index im ds_shuffled
                # real_idx: Index im ds_folder (alphabetisch sortiert)
                real_idx = shuffled_map[idx_in_subset]
                fname = ds_folder.files[real_idx]
                file_names.append(os.path.basename(fname))
                print(f"Batch Index {i:<2}: {os.path.basename(fname)}")

            # Speichern der Liste
            debug_path = os.path.join(os.path.dirname(path_lam), 'ring_debug')
            os.makedirs(debug_path, exist_ok=True)
            with open(os.path.join(debug_path, "debug_filenames.txt"), "w") as f:
                for name in file_names:
                    f.write(f"{name}\n")
            print(f"--> Dateiliste gespeichert in debug_filenames.txt")

        except Exception as e:
            print(f"WARNUNG: Konnte Dateinamen nicht auflösen: {e}")

        # 1. Batch holen
        key = jax.random.PRNGKey(seed)

        gen_output = generator(key)

        # Sicherheits-Check und Zuweisung
        if isinstance(gen_output, (list, tuple)) and len(gen_output) == 2:
            X_train_sample = gen_output[0]
            Y_train_sample = gen_output[1]

            # Prüfen ob Shapes plausibel sind
            if hasattr(X_train_sample, 'shape') and hasattr(Y_train_sample, 'shape'):
                print(f"  -> X Shape gefunden: {X_train_sample.shape}")
                print(f"  -> Y Shape gefunden: {Y_train_sample.shape}")

                # Speichern
                timestamp = time.strftime("%Y-%m-%d_%H-%M-%S")
                np.save(os.path.join(debug_path, f"debug_X_train_{timestamp}.npy"), np.array(X_train_sample))
                np.save(os.path.join(debug_path, f"debug_Y_train_{timestamp}.npy"), np.array(Y_train_sample))
                print(f"--> Gespeichert: debug_X/Y_train_{timestamp}.npy")
            else:
                print("WARNUNG: Elemente im Batch haben keine 'shape'.")
        else:
            print(f"WARNUNG: Unerwartetes Generator-Format: {type(gen_output)}")
            if isinstance(gen_output, (list, tuple)):
                print(f"Länge: {len(gen_output)}")

        # Validation auch speichern
        timestamp = time.strftime("%Y-%m-%d_%H-%M-%S")
        np.save(os.path.join(debug_path, f"debug_X_val_{timestamp}.npy"), np.array(X_val))
        np.save(os.path.join(debug_path, f"debug_Y_val_{timestamp}.npy"), np.array(y_val))
        print(f"--> Gespeichert: debug_X/Y_val_{timestamp}.npy")

        print("=== DEBUG ENDE ===\n")
    # -----------------------------------------------------------

    # 4. NETZWERK
    net = _make_net(
        lam, warmstart, rnn_w, rnn_d, lin_w, lin_d, layernorm, celltype, rnno
    )

    # 5. OPTIMIZER
    if warmstart is not None:
        n_decay_steps = int(0.85 * total_steps)
        n_warmup_steps = int(0.15 * total_steps)
        n_steps_per_episode = int(6000 / tbp / grad_accu)
        optimizer = optax.MultiSteps(
            optax.lamb(
                optax.join_schedules(
                    [
                        optax.schedules.linear_schedule(
                            1e-7, lr, n_warmup_steps * n_steps_per_episode
                        ),
                        optax.schedules.cosine_decay_schedule(
                            lr, n_decay_steps * n_steps_per_episode
                        ),
                    ],
                    [n_warmup_steps * n_steps_per_episode],
                )
            ),
            grad_accu,
        )
    else:
        n_decay_steps = int(0.95 * total_steps)
        n_steps_per_episode = int(6000 / tbp / grad_accu)
        optimizer = optax.MultiSteps(
            ring.ml.make_optimizer(
                lr, n_decay_steps, n_steps_per_episode, adap_clip=0.5, glob_clip=None
            ),
            grad_accu,
        )

    callbacks = [
        ring.ml.callbacks.EvalXyTrainingLoopCallback(
            net,
            _cb_metrices_factory(),
            X_val,
            y_val,
            lam,
            "val",
            link_names=link_names,
        )
    ]
    if use_wandb:
        # Hier fügen wir den neuen Logger ein
        callbacks.append(WandBEpochLogger(steps_per_epoch))

    if not disable_dump_model:
        callbacks.append(
            DumpModelCallback(_model(), net, overwrite=True, dump_every=None)
        )
    if not disable_checkpoint_model:
        callbacks.append(
            ml.callbacks.CheckpointCallback(
                checkpoint_every=5, checkpoint_folder=_checkpoints()
            )
        )

    print("\n=== SPEICHERORTE ===")
    print(f"Model Architektur: {_model()}")
    print(f"Parameter (Weights): {_params()}")
    print(f"Checkpoints:       {_checkpoints()}")
    print("====================\n")

    # 6. TRAINING STARTEN
    ml.train_fn(
        generator,
        total_steps,
        net,
        optimizer=optimizer,
        callbacks=callbacks,
        callback_kill_if_nan=True,
        callback_kill_if_grads_larger=1e32,
        callback_save_params=False if disable_save_params else _params(),
        seed_network=seed,
        link_names=link_names,
        tbp=tbp,
        loss_fn=_loss_fn_ring_factory(lam),
        metrices=None,
        skip_first_tbp_batch=skip_first,
        callback_create_checkpoint=False,
    )


if __name__ == "__main__":
    fire.Fire(main)
