from tampanda import ArmSceneBuilder
from dataclasses import dataclass
from tampanda.scenes import BLOCK_SMALL_TEMPLATE, BLOCK_MEDIUM_TEMPLATE, TABLE_TEMPLATE, TABLE_SYMBOLIC_TEMPLATE
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
def make_builder() -> ArmSceneBuilder:

    # register most necessary/basic simulation template separately
    builder = ArmSceneBuilder()
    builder.add_resource("table",  TABLE_SYMBOLIC_TEMPLATE)
    builder.add_resource("cube", BLOCK_SMALL_TEMPLATE)
    builder.add_resource("block", BLOCK_MEDIUM_TEMPLATE)
    builder.add_resource("pudding_box", {"type": "ycb", "name": "pudding_box"})
    builder.add_resource("gelatin_box", {"type": "ycb", "name": "gelatin_box"})
    builder.add_resource("brick", {"type": "ycb", "name": "foam_brick"})
    builder.add_resource("wood_block", {"type": "ycb", "name": "wood_block"})
    builder.add_object("table", name="table", pos=[0.75,  0.80, 0.00])

    return builder
        
