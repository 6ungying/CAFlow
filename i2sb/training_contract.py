"""Reject incompatible legacy objectives instead of silently reinterpreting weights."""

OBJECTIVE = "i2sb_residual_v1"


def training_contract(opt):
    return {
        "objective": OBJECTIVE,
        "timestep": "discrete_index",
        "interval": opt.interval,
        "beta_max": opt.beta_max,
        "ot_ode": opt.ot_ode,
        "use_single_dem": opt.use_single_dem,
        "depth_encoding": "white_dry",
        "single_max_depth": opt.single_max_depth,
        "train_rain_ids": opt.train_rain_ids,
        "val_rain_ids": opt.val_rain_ids,
        "test_dem_list": opt.test_dem_list,
        "test_rain_list": opt.test_rain_list,
        "val_nfe": opt.val_nfe,
        "validation_seed": opt.validation_seed,
    }


def check_checkpoint_contract(checkpoint, opt):
    if checkpoint.get("training_contract") != training_contract(opt):
        raise ValueError(
            "Checkpoint objective/data split/validation configuration is incompatible. "
            "Legacy continuous checkpoints cannot resume or sample with discrete I2SB. "
            "Start a new experiment; existing checkpoints are preserved."
        )
