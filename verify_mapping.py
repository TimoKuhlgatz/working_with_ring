'''
This script is to verify the mapping between a tensor batch derived from train_EMBC_model.py (in debug mode) and raw
data derived from train_step1_generateData_v2.py. This script can be used to check whether the tensor batch contains
the correct samples and whether the preprocessing pipeline actually wrote the expected sequences into the tensor.
'''
# usage:
# python verify_mapping.py "C:\Users\Timo_Kuhlgatz\Seafile\Meine Bibliotheken\02_projects\04_IMU_Motion_capture\03_RING\ring_debug\debug_X_train_2025-12-18_14-14-50.npy" "C:\Users\Timo_Kuhlgatz\Seafile\Meine Bibliotheken\02_projects\04_IMU_Motion_capture\03_RING\EMBC_KC"

import numpy as np
import pickle
import glob
import os
import argparse
from tqdm import tqdm


def load_pickle(path):
    """
    Load a pickle file containing raw data.

    Why:
        The training pipeline may store raw data differently depending on the step/version
        (e.g., directly as a dict or as a tuple (X, Y)). For mapping verification, we need
        consistent access to the X/input part so that it can be compared with the tensor.
    """
    with open(path, 'rb') as f:
        return pickle.load(f)


def get_fingerprint(acc_data):
    """
    Generate a 'fingerprint' (signature) for a sequence of acceleration data.

    How:
        - Takes the first up to 1000 frames.
        - Computes the norm per frame ||acc|| = sqrt(x^2 + y^2 + z^2).

    Why:
        During training, data is often normalized, sampled, or mixed into batches.
        Instead of trying to track exact metadata IDs, we generate a robust signature
        from the values themselves. The norm is a highly compressed representation
        that often distinguishes sequences well enough to reassign batch elements
        to their corresponding pickle files (via an MSE comparison).
    """

    # Wir nehmen die ersten 1000 Frames
    n_samples = min(len(acc_data), 1000)
    acc_segment = acc_data[:n_samples]

    # Norm berechnen: sqrt(x^2 + y^2 + z^2)
    norm = np.linalg.norm(acc_segment, axis=1)
    return norm


def verify_systematic_mapping(tensor_path, data_folder):
    """
    Systematically verify the mapping between a .npy tensor batch and pickle raw data.

    What this tool checks:
        - Whether each sample in the stored tensor batch can plausibly be assigned to
          a raw data file.
        - Whether the training/preprocessing pipeline actually wrote the expected
          sequences into the tensor (instead of, e.g., off-by-one errors, wrong folder,
          wrong segment, or unintended shuffling without traceability).

    How it works (high level):
        1) Load the tensor batch (.npy).
        2) Index all pickle files in the raw data directory:
           - Extract the acceleration signal (e.g., seg3_2Seg or fallback).
           - Apply the same scaling as in training (acceleration in the tensor is
             typically divided by 9.81).
           - Fingerprint = norm of the acceleration vectors over the first frames.
        3) For each batch element in the tensor:
           - Compute the fingerprint from the tensor.
           - Find the best-matching pickle file via minimal MSE between fingerprints.
        4) Output the results in tabular form (including a warning if the MSE is too high).

    Why MSE on fingerprints:
        Exact 1:1 comparisons on raw values are often fragile in training pipelines
        (normalization, float rounding, optional augmentation). The fingerprint approach
        is fast, scalable, and in practice robust enough to uncover systematic
        mapping errors.
    """

    # 1. TENSOR LADEN
    print(f"Lade Tensor-Batch: {tensor_path}")
    try:
        X_batch = np.load(tensor_path)
        print(f"Tensor Shape: {X_batch.shape}")
    except Exception as e:
        print(f"Fehler beim Laden des Tensors: {e}")
        return

    # 2. ROHDATEN LADEN & FINGERABDRÜCKE ERSTELLEN
    print(f"\nLade und indiziere Rohdaten aus: {data_folder}/*.pickle")
    file_paths = glob.glob(os.path.join(data_folder, "*.pickle"))

    if not file_paths:
        print("KEINE DATEIEN GEFUNDEN! Pfad prüfen.")
        return

    file_db = []

    print("Erstelle Datenbank der Rohdaten...")
    for fp in tqdm(file_paths):
        try:
            raw_content = load_pickle(fp)

            # --- FIX: Tupel behandeln ---
            if isinstance(raw_content, tuple):
                # Wir nehmen an: (X_dict, Y_dict) -> X ist Index 0
                data = raw_content[0]
            else:
                data = raw_content
            # ----------------------------

            # Pfad zur Beschleunigung finden
            if 'seg3_2Seg' in data:
                acc_raw = data['seg3_2Seg']['acc']
            elif 'acc' in data:
                acc_raw = data['acc']
            else:
                # Fallback: Erstes verfügbares Segment
                first_key = list(data.keys())[0]
                acc_raw = data[first_key]['acc']

            # Skalierung anwenden (Tensor Werte sind durch 9.81 geteilt)
            acc_scaled = acc_raw / 9.81

            fingerprint = get_fingerprint(acc_scaled)
            file_db.append({
                'name': os.path.basename(fp),
                'fingerprint': fingerprint
            })

        except Exception as e:
            # Nur echte Fehler anzeigen, keine erwarteten Struktur-Probleme
            print(f"Überspringe {os.path.basename(fp)}: {e}")

    # 3. ABGLEICH (SYSTEMISCHE ZUORDNUNG)
    print("\n" + "=" * 80)
    print(f"{'BATCH IDX':<10} | {'ZUGEORDNETE DATEI':<35} | {'CONFIDENCE (MSE)'}")
    print("=" * 80)

    matches_found = 0

    for i in range(len(X_batch)):
        # Tensor Daten: Segment 0, Acc (0:3)
        tensor_acc = X_batch[i, :, 0, 0:3]
        tensor_fp = get_fingerprint(tensor_acc)

        best_match_name = "UNBEKANNT"
        best_mse = float('inf')

        for entry in file_db:
            file_fp = entry['fingerprint']

            length = min(len(tensor_fp), len(file_fp))

            # MSE Berechnung
            mse = np.mean((tensor_fp[:length] - file_fp[:length]) ** 2)

            if mse < best_mse:
                best_mse = mse
                best_match_name = entry['name']

        # Etwas tolerantere Schwelle setzen
        if best_mse < 1e-4:
            status = "OK"
            matches_found += 1
        else:
            status = "WARNUNG"

        print(f"{i:<10} | {best_match_name:<35} | {best_mse:.8f}  {status if best_mse > 1e-4 else ''}")

    print("=" * 80)
    print(f"Systemische Zuordnung abgeschlossen. {matches_found}/{len(X_batch)} Dateien identifiziert.")
    print(f'Bedenke bei der Zuordnung, dass die Tensoren bereits normiert wurden. Die Beschleunigung wurde mit 9,81 m/s², '
          f'die gyroskopischen Daten mit 2,2 rad/s und die Zeitschritte mit 0,01 normiert.')


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("tensor_file", help="Pfad zur .npy Datei")
    parser.add_argument("data_folder", default="../EMBC_KC", help="Pfad zum Rohdaten-Ordner")
    args = parser.parse_args()

    verify_systematic_mapping(args.tensor_file, args.data_folder)
