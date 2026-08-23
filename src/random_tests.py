from hydra_zen import store, make_custom_builds_fn, zen

from game_library import inventory, Character

builds = make_custom_builds_fn(populate_full_signature=True)

# Create inventory configs
InventoryConf = builds(inventory)
starter_gear = InventoryConf(gold=10, weapon="stick", costume="{weapon}")
advanced_gear = InventoryConf(gold=500, weapon="wand", costume="${player.inventory.weapon}")
hard_mode_gear = InventoryConf(gold=0, weapon="inner thoughts", costume="rags")

# Register inventory configs under group: player/inventory
inv_store = store(group="player/inventory")

# inventory can take one of the 3 values, given by an alias
# names don't feature in actual config, as I stated
inv_store(starter_gear, name="starter")
inv_store(advanced_gear, name="advanced")
inv_store(hard_mode_gear, name="hard_mode")

# Create player-profile configs
CharConf = builds(Character, inventory=starter_gear)

brinda_conf = CharConf(
    name="brinda",
    level=47,
    inventory=InventoryConf(costume="cape", weapon="flute", gold=52),
)

rakesh_conf = CharConf(
    name="rakesh",
    level=300,
    inventory=InventoryConf(costume="PJs", weapon="pillow", gold=41),
)

# Register player-profile configs under group: player
player_store = store(group="playe")

player_store(CharConf, name="base")
player_store(brinda_conf, name="brinda")
player_store(rakesh_conf, name="rakesh")


# The `hydra_defaults` field is specified in our task function's config.
# It instructs Hydra to use the player config that named 'base' in our
# config store as the default config for our app.
@store(name="my_app")
def task_function(playe: Character):

    print(playe)

    with open("player_log.txt", "a") as f:
        f.write("Game session log:\n")
        f.write(f"Player: {playe}\n")

    return playe


if __name__ == "__main__":
    store.add_to_hydra_store()
    zen(task_function).hydra_main(config_name="my_app",
                                  version_base="1.3",
                                  config_path=".",
                                  )