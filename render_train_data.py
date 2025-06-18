import ring
import cv2
from ring.algorithms import kinematics
from ring.sys_composer import inject_system
import jax
import jax.numpy as jnp
import math
import pickle
import numpy as np
import os
import time
import argparse
from typing import List, Dict, Any, Tuple
import mediapy

# --- Constants ---
# Default grid layout columns
N_COLS = 8
# Default spacing between models
OFFSET = 0.8
# The base model definition for a single kinematic chain
SYS_STR = """
<x_xy model="lam2_viz">
  <options dt="0.01" gravity="0.0 0.0 9.81"/>
  <worldbody>
    <body joint="free" name="seg3_2Seg" pos="0 0 0">
      <geom pos="0.1 0.0 0.0" mass="1.0" color="dustin_exp_blue" type="box" dim="0.2 0.05 0.05"/>
      <body joint="frozen" name="imu3_2Seg" pos="0.1 0.0 0.035">
        <geom mass="0.1" color="dustin_exp_orange" type="box" dim="0.05 0.03 0.02"/>
      </body>
      <body joint="spherical" name="seg4_2Seg" pos="0.2 0.0 0.0">
        <geom pos="0.1 0.0 0.0" mass="1.0" color="dustin_exp_white" type="box" dim="0.2 0.05 0.05"/>
        <body joint="frozen" name="imu4_2Seg" pos="0.1 0.0 0.035">
          <geom mass="0.1" color="dustin_exp_orange" type="box" dim="0.05 0.03 0.02"/>
        </body>
      </body>
    </body>
  </worldbody>
</x_xy>
"""


def show_video_with_opencv(frames: List[np.ndarray], dt: float):
    """Displays a list of frames as a video using OpenCV."""
    for frame in frames:
        frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        cv2.imshow('Movement Visualization', frame_bgr)
        if cv2.waitKey(int(dt * 1000)) & 0xFF == ord('q'):
            break
    cv2.destroyAllWindows()


def load_motion_data(data_path: str) -> List[Any]:
    """Loads all pickle files from a specified directory."""
    print(f"Loading motion data from: {data_path}")
    if not os.path.exists(data_path):
        print(f"ERROR: Data directory not found at {data_path}")
        return []

    kc_data = []
    try:
        kc_data_files = [f for f in os.listdir(data_path) if f.endswith('.pickle')]
        for file in kc_data_files:
            with open(os.path.join(data_path, file), 'rb') as f:
                kc_data.append(pickle.load(f))
        print(f"Successfully loaded {len(kc_data)} data files.")
    except Exception as e:
        print(f"An error occurred while loading data: {e}")
    return kc_data


def build_combined_system(sys_str: str, num_trials: int) -> ring.System:
    """Builds a single ring.System containing multiple kinematic chains."""
    print("Building the combined system... (This may take a long time and consume a lot of memory)")
    if num_trials == 0:
        raise ValueError("Cannot build a system with zero trials.")

    # Start with the first system, giving it a suffix "_0" for consistency
    combined_sys = ring.System.create(sys_str).add_prefix_suffix(suffix="_0")
    # Add flush=True to ensure the output is shown immediately
    print(f"  - Injected system for trial 0 (Base system)", flush=True)

    # Loop through the rest of the trials and inject them
    for i in range(1, num_trials):
        system_to_add = ring.System.create(sys_str).add_prefix_suffix(suffix=f"_{i}")
        combined_sys = inject_system(combined_sys, system_to_add)
        # Add flush=True inside the loop to track progress
        print(f"  - Injected system for trial {i}", flush=True)

    print(f"Finished building system with {combined_sys.num_links()} total segments.", flush=True)
    return combined_sys


def get_camera_xml(positions: np.ndarray) -> Tuple[str, str]:
    """Generates the camera XML string for an overview shot."""
    x_center = np.max(positions[:, 0]) / 2
    y_center = np.max(positions[:, 1]) / 2
    n_rows = int(np.ceil(len(positions) / N_COLS))

    cam_pos_x = x_center
    cam_pos_y = y_center - n_rows  # Position camera based on number of rows
    cam_pos_z = n_rows  # Elevate camera for a good viewing angle
    cam_pos_str = f"{cam_pos_x} {cam_pos_y} {cam_pos_z}"

    # Tilt camera down by ~45 degrees
    cam_xyaxes_str = "1 0 0 0 0.707 0.707"
    camera_name = "overview_cam"

    camera_xml = (
        f'<camera name="{camera_name}" pos="{cam_pos_str}" '
        f'xyaxes="{cam_xyaxes_str}" fovy="45" />'
    )
    return camera_xml, camera_name


def preprocess_data_for_jax(kc_data: List[Any], num_timesteps: int) -> Tuple[jnp.ndarray, jnp.ndarray]:
    """Pre-processes raw motion data into batched JAX arrays."""
    print("\nPreprocessing motion data for JAX...", flush=True)
    all_quat_seg1, all_quat_seg2 = [], []
    for trial_data in kc_data:
        motion_data_y = trial_data[1]
        all_quat_seg1.append(motion_data_y['seg3_2Seg'][:num_timesteps])
        all_quat_seg2.append(motion_data_y['seg4_2Seg'][:num_timesteps])

    # Transpose to (num_timesteps, num_trials, 4) for easier processing
    all_quat_seg1 = jnp.transpose(jnp.array(all_quat_seg1), (1, 0, 2))
    all_quat_seg2 = jnp.transpose(jnp.array(all_quat_seg2), (1, 0, 2))
    return all_quat_seg1, all_quat_seg2


def calculate_kinematics(
        system: ring.System,
        q1_data: jnp.ndarray,
        q2_data: jnp.ndarray,
        positions: jnp.ndarray
) -> List[ring.base.Transform]:
    """Performs the JAX-accelerated forward kinematics calculation."""
    print(f"\nProcessing {q1_data.shape[0]} timesteps for {q1_data.shape[1]} chains using JAX...", flush=True)
    initial_state = ring.State.create(system)

    def build_q(q1, q2, pos):
        pos_expanded = jnp.tile(pos, (q1.shape[0], 1, 1))
        q_free = jnp.concatenate([q1, pos_expanded], axis=2)
        q_combined = jnp.concatenate([q_free, q2], axis=2)
        return q_combined.reshape(q_combined.shape[0], -1)

    @jax.jit
    def jitted_fk_step(q: jnp.ndarray) -> ring.base.Transform:
        state = initial_state.replace(q=q)
        _, new_state = kinematics.forward_kinematics(system, state)
        return new_state.x

    q_all_timesteps = build_q(q1_data, q2_data, positions)
    vmapped_fk = jax.vmap(jitted_fk_step)

    start_time = time.time()
    xs_batch = jax.block_until_ready(vmapped_fk(q_all_timesteps))
    end_time = time.time()
    print(f"  JAX computation finished in {end_time - start_time:.4f} seconds.", flush=True)

    # Un-batch the batched Transform object into a list of Transforms
    return [ring.base.Transform(pos=xs_batch.pos[i], rot=xs_batch.rot[i]) for i in range(xs_batch.pos.shape[0])]


def main(args):
    """Main execution function."""
    # --- DIAGNOSTIC PRINT ---
    # Print the arguments the script received to verify the configuration
    print("--- Script Starting ---")
    print(f"Arguments received: {args}")
    print("-----------------------")

    try:
        kc_data = load_motion_data(args.data_dir)
        if not kc_data:
            return

        num_trials = len(kc_data)
        combined_sys = build_combined_system(SYS_STR, num_trials)

        # Determine grid positions for the kinematic chains
        positions = np.array([[i % N_COLS * OFFSET, i // N_COLS * OFFSET, 0.0] for i in range(num_trials)])

        # Determine number of timesteps to render
        min_len = min(len(trial[1]['seg3_2Seg']) for trial in kc_data)

        if args.timesteps is None:
            num_timesteps = min_len
        else:
            num_timesteps = min(args.timesteps, min_len)
        print(f"Rendering {num_timesteps} timesteps.", flush=True)

        # Prepare data and run kinematics
        q1_data, q2_data = preprocess_data_for_jax(kc_data, num_timesteps)
        xs = calculate_kinematics(combined_sys, q1_data, q2_data, positions)

        # Get camera setup
        camera_xml, camera_name = get_camera_xml(positions)

        # Render the final animation
        print("\nRendering animation...", flush=True)
        frames = combined_sys.render(
            xs=xs,
            width=args.width,
            height=args.height,
            add_cameras={-1: camera_xml},
            camera=camera_name,
        )

        if args.save_intermediates:
            with open('system_variable.pkl', 'wb') as f:
                pickle.dump(combined_sys, f)
            print("System object saved to 'system_variable.pkl'", flush=True)
            with open('xs.pkl', 'wb') as f:
                pickle.dump(xs, f)
            print("Forward kinematics saved to 'xs.pkl'", flush=True)

        if args.output_path:
            print(f"\nSaving video to {args.output_path}...", flush=True)
            output_dir = os.path.dirname(args.output_path)
            if output_dir:
                os.makedirs(output_dir, exist_ok=True)

            fps = 1 / combined_sys.dt
            mediapy.write_video(args.output_path, frames, fps=fps)
            print("Video saved successfully.", flush=True)

        if not args.no_display:
            print("\nPlaying animation. Press 'q' to quit.", flush=True)
            show_video_with_opencv(frames, combined_sys.dt)

    except Exception as e:
        print("\n--- AN ERROR OCCURRED ---")
        print(f"Error Type: {type(e).__name__}")
        print(f"Error Details: {e}")
        import traceback
        traceback.print_exc()
        print("-------------------------")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Visualize kinematic chain motion from pickle files.")
    parser.add_argument(
        '--data-dir',
        type=str,
        default=r'C:\Users\Timo_Kuhlgatz\Seafile\Meine Bibliotheken\02_projects\04_IMU_Motion_capture\03_RING\EMBC_KC',
        help='Directory containing the motion data pickle files.'
    )
    parser.add_argument(
        '--output-path',
        type=str,
        default=None,
        help='Path to save the output video (e.g., output/animation.mp4).'
    )
    parser.add_argument(
        '--timesteps',
        type=int,
        default=None,
        help='Number of timesteps to render. Defaults to the full length of the shortest sequence.'
    )
    parser.add_argument('--width', type=int, default=3000, help='Width of the output video.')
    parser.add_argument('--height', type=int, default=2160, help='Height of the output video.')
    parser.add_argument(
        '--save-intermediates',
        action='store_true',
        help='If set, save the combined system and kinematics data to pickle files.'
    )
    parser.add_argument(
        '--no-display',
        action='store_true',
        help='If set, do not display the video with OpenCV.'
    )

    args = parser.parse_args()
    main(args)
