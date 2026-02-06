"""
inspect_batch.py — Tensor-to-raw-data audit tool for EMBC IMU training batches.

This script is meant for debugging and validating the data pipeline that creates `.npy`
input tensors for model training (e.g., produced by `train_EMBC_model.py` in a debug run).
It answers two practical questions that are otherwise hard to verify once batching,
shuffling and normalization are involved:

1) “Which raw `.pickle` file does a given tensor sample (batch index i) come from?”
2) “Did the preprocessing pipeline write the *correct values* into the tensor, and did it
   scale/normalize them as expected?”

How it works (high level):
    - Loads a stored tensor batch `X_batch` from a `.npy` file (shape: Batch × Time × Segments × Features).
    - Indexes all `.pickle` files in a raw-data folder and extracts the corresponding IMU signals.
    - Builds a lightweight “fingerprint” for each file based on the norm of acceleration
      (||acc|| over the first samples). This fingerprint is used to match each tensor sample
      to the most likely originating raw file via a minimum-MSE search.
    - After a match is found, prints a human-readable comparison:
        * dt: compares raw dt to tensor dt (after rescaling)
        * ||acc||: compares the acceleration norm over the full sequence (useful even when
          orientation augmentation is enabled)
        * acc & gyr: compares the first few frames component-wise after rescaling back to
          physical units (useful when no orientation randomization was applied)

Why this is useful:
    Training can “appear to work” even if the input tensors are wrong (wrong segment key,
    wrong feature order, wrong scaling constants, off-by-one windowing, or accidental mixing
    of files). This script provides a quick audit trail from tensor samples back to their
    raw sources and highlights inconsistencies early—before they turn into silent model
    performance issues.

Notes / limitations:
    - Component-wise acc/gyr comparisons are only expected to match exactly if no
      orientation randomization/augmentation was applied (often referred to as `rand_ori=False`).
      If augmentation rotates vectors, the norms (e.g., ||acc||) and dt are typically still
      meaningful sanity checks.
    - The scaling constants (SCALE_ACC, SCALE_GYR, SCALE_DT) must match the ones used
      when the tensor was generated, otherwise the “back-calculated” values will differ.

"""
# usage:
# python inspect_batch.py "C:\Users\Timo_Kuhlgatz\Seafile\Meine Bibliotheken\02_projects\04_IMU_Motion_capture\03_RING\ring_debug\debug_X_train_2025-12-18_14-14-50.npy" "C:\Users\Timo_Kuhlgatz\Seafile\Meine Bibliotheken\02_projects\04_IMU_Motion_capture\03_RING\EMBC_KC"


import numpy as np
import pickle
import glob
import os
import argparse
from tqdm import tqdm

# Konfiguration (muss mit Ihrem Training übereinstimmen!)
SCALE_ACC = 9.81
SCALE_GYR = 2.2
SCALE_DT = 0.01


def load_pickle(path):
    with open(path, 'rb') as f:
        data = pickle.load(f)
    # Tuple-Handling (X, Y) -> X
    if isinstance(data, tuple):
        return data[0]
    return data


def get_fingerprint(acc_data):
    # Norm der ersten 500 Samples für das Matching
    n = min(len(acc_data), 500)
    return np.linalg.norm(acc_data[:n], axis=1)


def find_matching_file(tensor_acc_segment, file_db):
    """Sucht die passende Datei anhand der Acc-Norm."""
    # Tensor zurückskalieren für Fingerprint (Norm ist skalierungsinvariant, aber sicher ist sicher)
    tensor_norm = np.linalg.norm(tensor_acc_segment, axis=1)

    best_match = None
    best_mse = float('inf')

    for entry in file_db:
        file_norm = entry['fingerprint']
        length = min(len(tensor_norm), len(file_norm))
        mse = np.mean((tensor_norm[:length] - file_norm[:length]) ** 2)

        if mse < best_mse:
            best_mse = mse
            best_match = entry

    return best_match, best_mse


def inspect_batch(tensor_path, data_folder):
    print(f"--- INSPEKTION: {os.path.basename(tensor_path)} ---")

    # 1. Tensor laden
    # Shape: (Batch, Time, Segments, Features)
    # Features: 0-2 Acc, 3-5 Gyr, 6 dt (bei 7 Features)
    X_batch = np.load(tensor_path)
    batch_size = X_batch.shape[0]

    # 2. Datenbank aufbauen
    print("Indiziere Rohdaten...")
    file_paths = glob.glob(os.path.join(data_folder, "*.pickle"))
    file_db = []

    for fp in file_paths:
        try:
            data = load_pickle(fp)
            # Finde Acc Daten (Segment 3 ist meist Index 0 im Tensor)
            if 'seg3_2Seg' in data:
                acc = data['seg3_2Seg']['acc']
                gyr = data['seg3_2Seg']['gyr']
                # dt holen (kann top-level oder im seg sein)
                dt = data.get('dt', data['seg3_2Seg'].get('dt', 0.01))
            else:
                # Fallback
                k = list(data.keys())[0]
                acc = data[k]['acc']
                gyr = data[k]['gyr']
                dt = data.get('dt', 0.01)

            # Wir speichern die Rohdaten direkt im RAM für den schnellen Vergleich
            file_db.append({
                'name': os.path.basename(fp),
                'fingerprint': get_fingerprint(acc / SCALE_ACC),  # Norm basierend auf skalierten Daten
                'acc_raw': acc,
                'gyr_raw': gyr,
                'dt_raw': dt
            })
        except:
            pass

    print(f"\nStarte Detail-Vergleich für {batch_size} Sequenzen...")
    print("Hinweis: Vergleich funktioniert nur exakt, wenn rand_ori=False war.")
    print("         Bei rand_ori=True stimmen nur dt und die Norm (Länge) überein.\n")

    # 3. Iteration durch den Batch
    for i in range(batch_size):
        print("=" * 80)
        print(f"BATCH INDEX {i}")

        # Tensor Daten extrahieren (Segment 0 / Root)
        # Acc: 0:3, Gyr: 3:6, dt: 6 (oder -1)
        t_acc = X_batch[i, :, 0, 0:3]
        t_gyr = X_batch[i, :, 0, 3:6]
        t_dt = X_batch[i, :, 0, 6]

        # Match finden
        match, mse = find_matching_file(t_acc, file_db)

        if match is None or mse > 1e-4:
            print(f"WARNUNG: Kein eindeutiger Match gefunden! (MSE: {mse})")
            continue

        print(f"DATEI: {match['name']} (Confidence MSE: {mse:.8f})")
        print("-" * 80)
        print(f"{'TYPE':<10} | {'ROHDATEN (aus Datei)':<30} | {'TENSOR (Rückgerechnet)':<30} | {'DIFF'}")
        print("-" * 80)

        # A) DT Vergleich
        # Tensor dt ist Array, Roh dt ist float. Wir nehmen den Mittelwert des Tensors.
        t_dt_val = np.mean(t_dt) * SCALE_DT
        r_dt_val = match['dt_raw']
        print(f"{'dt':<10} | {r_dt_val:<30.6f} | {t_dt_val:<30.6f} | {abs(r_dt_val - t_dt_val):.6f}")

        # ... existing code ...

        # A2) ACC-Norm Vergleich (über die komplette Sequenz, in Tensor-Skala)
        # Wichtig für rand_ori=True: Komponenten stimmen nicht, aber Norm sollte sehr ähnlich bleiben.
        r_acc_scaled = match['acc_raw'] / SCALE_ACC
        t_norm = np.linalg.norm(t_acc, axis=1)
        r_norm = np.linalg.norm(r_acc_scaled, axis=1)
        length = min(len(t_norm), len(r_norm))

        norm_diff = t_norm[:length] - r_norm[:length]
        norm_mse = float(np.mean(norm_diff ** 2))
        norm_mae = float(np.mean(np.abs(norm_diff)))
        norm_max = float(np.max(np.abs(norm_diff)))

        r_norm_str = f"m={float(np.mean(r_norm[:length])):.6f}"
        t_norm_str = f"m={float(np.mean(t_norm[:length])):.6f}"
        diff_str = f"MSE={norm_mse:.2e}, MAE={norm_mae:.2e}, MAX={norm_max:.2e}"
        print(f"{'||acc||':<10} | {r_norm_str:<30} | {t_norm_str:<30} | {diff_str}")

        # B) ACC Vergleich (Erste 3 Zeitschritte)
        print("-" * 80)
        for t in range(3):  # Die ersten 3 Frames
            # Rückrechnung: Tensor * Scale
            t_vec = t_acc[t] * SCALE_ACC
            r_vec = match['acc_raw'][t]

            diff = np.linalg.norm(t_vec - r_vec)

            # Formatierung für saubere Ausgabe
            r_str = f"[{r_vec[0]:.2f}, {r_vec[1]:.2f}, {r_vec[2]:.2f}]"
            t_str = f"[{t_vec[0]:.2f}, {t_vec[1]:.2f}, {t_vec[2]:.2f}]"

            print(f"{f'Acc t={t}':<10} | {r_str:<30} | {t_str:<30} | {diff:.4f}")

        # C) GYR Vergleich
        print("-" * 80)
        for t in range(3):
            t_vec = t_gyr[t] * SCALE_GYR
            r_vec = match['gyr_raw'][t]
            diff = np.linalg.norm(t_vec - r_vec)

            r_str = f"[{r_vec[0]:.2f}, {r_vec[1]:.2f}, {r_vec[2]:.2f}]"
            t_str = f"[{t_vec[0]:.2f}, {t_vec[1]:.2f}, {t_vec[2]:.2f}]"

            print(f"{f'Gyr t={t}':<10} | {r_str:<30} | {t_str:<30} | {diff:.4f}")

        print("\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("tensor_file", help="Pfad zur .npy Datei")
    parser.add_argument("data_folder", default="../EMBC_KC", help="Pfad zum Rohdaten-Ordner")
    args = parser.parse_args()

    inspect_batch(args.tensor_file, args.data_folder)