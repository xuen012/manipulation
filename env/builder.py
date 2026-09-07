from tampanda import ArmSceneBuilder
from tampanda.scenes import BLOCK_SMALL_TEMPLATE, BLOCK_MEDIUM_TEMPLATE, TABLE_TEMPLATE, TABLE_SYMBOLIC_TEMPLATE

def make_builder() -> ArmSceneBuilder:
    
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
        
