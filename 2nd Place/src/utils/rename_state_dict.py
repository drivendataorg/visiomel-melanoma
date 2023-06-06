import torch
from collections import OrderedDict

def key_transformation(key_name):
    return key_name.replace("module.","")

def rename_state_dict_keys(source, key_transformation, target=None):
    """
    source             -> Path to the saved state dict.
    key_transformation -> Function that accepts the old key names of the state
                          dict as the only argument and returns the new key name.
    target (optional)  -> Path at which the new state dict should be saved
                          (defaults to `source`)
    Example:
    Rename the key `layer.0.weight` `layer.1.weight` and keep the names of all
    other keys.
    ```py
    def key_transformation(old_key):
        if old_key == "layer.0.weight":
            return "layer.1.weight"
        return old_key
    rename_state_dict_keys(state_dict_path, key_transformation)
    ```
    """
    if target is None:
        target = source

    state_dict = torch.load(source,map_location=torch.device('cpu'))
    new_state_dict = OrderedDict()

    for key, value in state_dict.items():
        new_key = key_transformation(key)
        new_state_dict[new_key] = value

    torch.save(new_state_dict, target)

model_path1 = "/Users/Happpyyyyyyy/Documents/VisioMel/trained_models/best_model_loss_ulceration_regular.pth"
rename_state_dict_keys(model_path1, key_transformation, target="/Users/Happpyyyyyyy/Documents/VisioMel/trained_models/remapped_best_model_loss_ulceration_regular.pth")
