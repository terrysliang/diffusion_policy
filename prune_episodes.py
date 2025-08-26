#!/usr/bin/env python3
import argparse, os, shutil, zarr, tempfile
from pathlib import Path
from diffusion_policy.common.replay_buffer import ReplayBuffer


def find_episode_ends(root: zarr.hierarchy.Group):
    # Root-level?
    if "episode_ends" in root.array_keys():
        return root["episode_ends"][:]
    # meta/episode_ends ?
    if "meta" in root.group_keys():
        meta = root["meta"]
        if "episode_ends" in meta.array_keys():
            return meta["episode_ends"][:]
    # brute-force search for a 1D int array with monotonic increase
    for path, arr in walk_arrays(root).items():
        if arr.ndim == 1 and arr.dtype.kind in ("i", "u"):
            data = arr[:]
            if data.size > 0 and (data[1:] >= data[:-1]).all():
                # heuristic: usually the max equals the length of time dimension
                return data
    raise KeyError("Could not find episode_ends. Use --inspect to list arrays.")

def walk_arrays(root: zarr.hierarchy.Group, base=""):
    out = {}
    for k in root.array_keys():
        out[(base + k)] = root[k]
    for g in root.group_keys():
        sub = root[g]
        out.update(walk_arrays(sub, base + g + "/"))
    return out

def list_store(root):
    print("Groups:", root.group_keys())
    print("Arrays:", root.array_keys())
    print("All arrays (recursive):")
    for path in walk_arrays(root).keys():
        print("  -", path)


def list_arrays(root):
    # all array keys except episode_ends (we’ll slice those)
    return [k for k in root.array_keys() if k != 'episode_ends']

def episode_slices(episode_ends):
    prev = 0
    for end in episode_ends:
        yield slice(prev, end)
        prev = end

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("dataset_dir", help="Path containing replay_buffer.zarr and videos/")
    ap.add_argument("--delete", nargs="+", type=int, required=True,
                    help="Episode indices to delete (space separated)")
    ap.add_argument("--dry_run", action="store_true", help="Show plan only")
    args = ap.parse_args()

    ds = Path(args.dataset_dir)

    zarr_path = ds/"replay_buffer.zarr"
    root_old = zarr.open(str(zarr_path), mode="r")

    # Optional: an --inspect flag if you want
    # if args.inspect:
    #     list_store(root_old); return

    ep_ends = find_episode_ends(root_old)
    n_eps = len(ep_ends)

    to_delete = sorted(set([e for e in args.delete if 0 <= e < n_eps]))
    keep = [i for i in range(n_eps) if i not in to_delete]

    print(f"Found {n_eps} episodes. Deleting: {to_delete}. Keeping: {keep}.")
    if args.dry_run:
        return

    # Work in a temp dir then swap
    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir = Path(tmpdir)
        zarr_new = tmpdir/"replay_buffer.zarr"
        videos_new = tmpdir/"videos"
        videos_new.mkdir(parents=True, exist_ok=True)

        # Create new replay buffer
        rb_new = ReplayBuffer.create_from_path(str(zarr_new), mode="a")

        # Build slices once
        slices = list(episode_slices(ep_ends))
        arr_keys = list_arrays(root_old)

        # Copy kept episodes into new RB and videos with reindexed ids
        new_idx = 0
        for old_idx in keep:
            sl = slices[old_idx]
            # episode dict: slice every array along the time axis
            episode = {k: root_old[k][sl] for k in arr_keys}
            rb_new.add_episode(episode, compressors='disk')

            # copy videos/<old_idx> -> videos/<new_idx> if exists
            src = vids_dir/str(old_idx)
            if src.exists():
                dst = videos_new/str(new_idx)
                shutil.copytree(src, dst)
            new_idx += 1

        # Atomically replace originals
        # Move old aside (backup), then move new in place
        backup = ds/"backup_before_prune"
        if backup.exists():
            shutil.rmtree(backup)
        backup.mkdir()
        shutil.move(str(zarr_path), str(backup/"replay_buffer.zarr"))
        shutil.move(str(vids_dir), str(backup/"videos"))

        shutil.move(str(zarr_new), str(zarr_path))
        shutil.move(str(videos_new), str(vids_dir))
        print("Prune complete. Backup saved at:", backup)

    # Invalidate any dataset caches created by RealPushTImageDataset (if used)
    # They look like: <dataset_dir>/<md5_of_shape_meta>.zarr.zip
    for p in Path(args.dataset_dir).glob("*.zarr.zip"):
        try:
            os.remove(p)
            print("Removed cache:", p)
        except Exception:
            pass

if __name__ == "__main__":
    main()
