"""Optional local installation/API check. No training or evaluation is run."""
import argparse
import importlib
import inspect
import sys


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scene", action="store_true", help="Also create the original cube and block scenes")
    args = parser.parse_args()
    print("Python:", sys.version.split()[0], sys.executable)
    errors = []
    for name in ("numpy", "torch", "diffusers", "accelerate", "pydantic", "mujoco", "unified_planning", "tampanda"):
        try:
            module = importlib.import_module(name)
            print(name, getattr(module, "__version__", "installed"), module.__file__)
        except ImportError as error:
            errors.append(f"{name}: {error}")
    if errors:
        raise SystemExit("\n".join(errors) + "\nInstall requirements and the course TAMPanda checkout in this interpreter.")
    import torch
    from model.unet1d import unet
    from tampanda.planners.grasp_planner import GraspCandidate
    print("CUDA available:", torch.cuda.is_available())
    print("Custom model parameters:", sum(p.numel() for p in unet(17, 10, 64, 64).parameters()))
    print("Native GraspCandidate:", inspect.signature(GraspCandidate))
    if args.scene:
        from data.collect_data import collect_data, close_env
        from data.processing import train_dataset, candidate_from_vector
        from pddl.checks import table_height
        for shape in (0, 1):
            records, _, env, _, _ = collect_data(1, shape_index=shape)
            try:
                sample, condition = train_dataset(records)[0]
                candidate_from_vector(sample[:, 0].numpy())
                print("shape", shape, "half_size", records[0][1], "table_height", table_height(env),
                      "tensor_shapes", tuple(sample.shape), tuple(condition.shape))
                detach = getattr(env, "detach_object_from_ee", None) or getattr(env, "detach_object", None)
                if not callable(detach):
                    raise RuntimeError("Course environment needs detach_object_from_ee or detach_object for the added place stage")
                print("Detach API:", inspect.signature(detach))
            finally:
                close_env(env)
    print("Setup check completed.")


if __name__ == "__main__":
    main()
