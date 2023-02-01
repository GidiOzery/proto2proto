import numpy as np
import sys
from src.mgr import manager
from src.dataset_loader import init_dataset
from src.services import init_service
import torch
import random
import os
import ray

# Ray Initialization
# num_cpus = os.cpu_count()
# ray.init(num_cpus = num_cpus)

def main(runName, newRun, serviceType, randomRun, ablationType):
    # runName, newRun, serviceType, randomRun, ablationType = sys.argv[1:]
    newRun = newRun.lower() == "true"
    if runName.strip().lower() == "none":
        runName = None
    manager(runName, newRun, serviceType, randomRun, ablationType)
  
    # Fix Randomness
    seed = manager.settingsConfig.train.seed
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)

    dataset_loader = init_dataset(manager.dataConfig.loaderName)
    service = init_service(serviceType, manager.service_name, dataset_loader)

    # print("returning prematurely")
    # return

    starter, ender = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    starter.record()

    service()

    ender.record()
    torch.cuda.synchronize()
    curr_time = starter.elapsed_time(ender)
    seconds = curr_time / 1000.0
    minutes = seconds / 60.0
    hours = minutes / 60.0
    print()
    print(f"time elapsed for training: {round(seconds, 3)} seconds -> {round(minutes, 3)} minutes -> {round(hours,3)} hours.")


if __name__ == "__main__":
    import os
    import yaml
    import csv
    from sklearn.model_selection import GridSearchCV
    from src.utils.argument_parser import read_from_file, save_to_file
    from datetime import datetime
    runName, newRun, serviceType, randomRun, ablationType = sys.argv[1:]
    base_dir = os.path.join('Experiments', ablationType, runName)
    yaml_path = os.path.join(base_dir, 'args.yaml')
    with open(yaml_path, 'r') as fp:
        cfg_dict = yaml.safe_load(fp)
    # print(cfg_dict)

    csv_summary_path = os.path.join(base_dir, 'grid_search_summary.csv')
    if not os.path.exists(csv_summary_path):
        with open(csv_summary_path, 'w') as fp:
            print('Creating CSV Summary')
            csvwriter = csv.writer(fp)
            csvwriter.writerow(["Run Name", "New Run", "Service Type", "Random Run", "Ablation Type",
            "Seed", 'lrNet', 'lrBlock', 'lrProto', 'lrLastLayer', 'weightDecay', 'Start Time', 'Run Time (m)'
                    ])  # Headers
            fp.flush()

    seed = 4100
    weight_decays = [
        0.05]
    lr_nets = [2e-4]
    lr_blocks = [0.001, 0.003, 0.01, 0.03, 0.05]
    lr_protos = [0.02, 0.05, 0.08]
    lr_last_layers = [0.0005]

    print(f'{len(weight_decays) * len(lr_nets) * len(lr_blocks) * len(lr_protos) * len(lr_last_layers)} Estimated Runs')

    for weight_decay in weight_decays:
        for lr_net in lr_nets:
            for lr_block in lr_blocks:
                for lr_proto in lr_protos:
                    for lr_last_layer in lr_last_layers:
                        print(f'Weight Decay {weight_decay}\tlr_net{lr_net}\tlr_block{lr_block}\tlr_last_layer{lr_last_layer}')
                        for seed_rep in range(1):
                            run_start_time = datetime.now()
                            start_time_strimg = run_start_time.strftime("%Y %d %m  %H:%M")
                            seed_dir = os.path.join(base_dir, str(seed))
                            try:
                                os.mkdir(seed_dir)
                            except OSError:
                                print(f'Warning seed {seed} already exists')
                            cfg_dict['settingsConfig']['train']['seed'] = seed
                            cfg_dict['settingsConfig']['train']['lrNet'] = lr_net
                            cfg_dict['settingsConfig']['train']['lrBlock'] = lr_block
                            cfg_dict['settingsConfig']['train']['lrProto'] = lr_proto
                            cfg_dict['settingsConfig']['train']['lrLastLayer'] = lr_last_layer
                            cfg_dict['settingsConfig']['train']['weightDecay'] = weight_decay

                            yaml_out_path = os.path.join(seed_dir, 'args.yaml')
                            with open(yaml_out_path, 'w') as file:
                                yaml.dump(cfg_dict, file)
                            # save_to_file(yaml_out_path, yaml_contents)
                            main(runName, newRun, serviceType, seed, ablationType)

                            time_end = datetime.now()
                            elapsed_time_minutes = (time_end-run_start_time)/60
                            
                            # Write to CSV
                            with open(csv_summary_path, 'a') as fp:
                                writer = csv.writer(fp)
                                writer.writerow([runName, newRun, serviceType, randomRun, ablationType,
        str(seed), str(lr_net), str(lr_block), str(lr_proto), str(lr_last_layer), str(weight_decay), 
        start_time_strimg, elapsed_time_minutes])
                                fp.flush()
                            seed += 1
