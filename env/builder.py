from dataclasses import dataclass
from pathlib import Path
import numpy as np

TABLE_Z = 0.27
OBJECT_NAME = "block_a"


@dataclass(frozen=True)
class BlockShape:
    name: str
    half_size: tuple[float, float, float]
    shape_class: int  # 0 = cube, 1 = short rectangular prism
    held_out: bool = False


SHAPES = (
    BlockShape("cube40", (0.02, 0.02, 0.02), 0),
    BlockShape("cube60", (0.03, 0.03, 0.03), 0),
    BlockShape("prism60x40x60", (0.03, 0.02, 0.03), 1),
    BlockShape("prism40x60x60", (0.02, 0.03, 0.03), 1),
    BlockShape("cube50", (0.025, 0.025, 0.025), 0, True),
    BlockShape("prism70x40x60", (0.035, 0.02, 0.03), 1, True),
)
SHAPE_BY_NAME = {s.name: s for s in SHAPES}


def make_builder():
    from tampanda import ArmSceneBuilder
    from tampanda.scenes import TABLE_SYMBOLIC_TEMPLATE
    builder = ArmSceneBuilder()
    builder.add_resource("table", TABLE_SYMBOLIC_TEMPLATE)
    for shape in SHAPES:
        builder.add_resource(shape.name, str(Path(__file__).parent / "templates" / f"{shape.name}.xml"))
    builder.add_object("table", name="table", pos=[0.75, 0.80, 0.0])
    return builder


class FastRate:
    """Keep simulator dt; omit wall clock sleeping for headless runs."""
    def __init__(self, dt):
        self.dt = dt

    def sleep(self):
        pass


def make_env(scene, realtime=False):
    import mujoco
    builder = make_builder()
    builder.add_object(scene["shape_name"], name=OBJECT_NAME, pos=scene["object_pos"],
                       quat=scene.get("object_quat", [1, 0, 0, 0]))
    for i, obstacle in enumerate(scene.get("obstacles", [])):
        builder.add_object(obstacle["shape_name"], name=f"obstacle_{i}", pos=obstacle["object_pos"])
    env = builder.build_env(rate=200.0)
    if not realtime:
        env.rate = FastRate(env.rate.dt)
    mujoco.mj_forward(env.model, env.data)
    if not np.allclose(env.get_object_half_size(OBJECT_NAME), scene["half_size"], atol=1e-6):
        env.close()
        raise ValueError("Scene half_size does not match the block template")
    return env
