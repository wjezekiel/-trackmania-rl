import logging
import os
import wandb
import random
import atexit


# # =========  WANDB CONFIG  ===========
wandb_run_id = "run"  # name for your run
wandb_resume="checkpoints"
wandb_project = "trackmania"  # name of the wandb project in which your run will appear
wandb_entity = "neeraja10"  # wandb account
wandb_key = "1fda8441e6e7ee859f59dc7743ce68725fc67161"  # wandb API key

os.environ['WANDB_API_KEY'] = wandb_key  # this line sets your wandb API key as the active key

# start a new wandb run to track this script
wandb.init(
    # set the wandb project where this run will be logged
    project="my-awesome-project",
    id="run1",
    resume="checkpoints"
)

# simulate training
epochs = 10
offset = random.random() / 5
for epoch in range(2, epochs):
    acc = 1 - 2 ** -epoch - random.random() / epoch - offset
    loss = 2 ** -epoch + random.random() / epoch + offset
    
    # log metrics to wandb
    wandb.log({"acc": acc, "loss": loss})
    
# [optional] finish the wandb run, necessary in notebooks
# wandb.finish()


# # ============================================================================
# # ============================================================================

# def run_with_wandb(entity, project, run_id, interface, run_cls, checkpoint_path: str = None, dump_run_instance_fn=None, load_run_instance_fn=None, updater_fn=None):
#     """
#     Main training loop (remote).

#     saves config and stats to https://wandb.com
#     """
#     dump_run_instance_fn = dump_run_instance_fn or dump_run_instance
#     load_run_instance_fn = load_run_instance_fn or load_run_instance
#     wandb_dir = tempfile.mkdtemp()  # prevent wandb from polluting the home directory
#     atexit.register(shutil.rmtree, wandb_dir, ignore_errors=True)  # clean up after wandb atexit handler finishes
#     logging.debug(f" run_cls: {run_cls}")
#     config = partial_to_dict(run_cls)
#     config['environ'] = log_environment_variables()
#     # config['git'] = git_info()  # TODO: check this for bugs
#     resume = checkpoint_path and exists(checkpoint_path)
#     wandb_initialized = False
#     err_cpt = 0
#     while not wandb_initialized:
#         try:
#             wandb.init(dir=wandb_dir, entity=entity, project=project, id=run_id, resume=resume, config=config)
#             wandb_initialized = True
#         except Exception as e:
#             err_cpt += 1
#             logging.warning(f"wandb error {err_cpt}: {e}")
#             if err_cpt > 10:
#                 logging.warning(f"Could not connect to wandb, aborting.")
#                 exit()
#             else:
#                 time.sleep(10.0)
#     # logging.info(config)
#     for stats in iterate_epochs_tm(run_cls, interface, checkpoint_path, dump_run_instance_fn, load_run_instance_fn, 1, updater_fn):
#         [wandb.log(json.loads(s.to_json())) for s in stats]