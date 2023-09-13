import os
import pandas as pd


def select_root(file_in):
    data_in = pd.read_csv(file_in)

    if 'naive' not in data_in['pred'].tolist():
        raise Exception("Naive B cell to select as a root does not exists")

    root_candidate = data_in.loc[data_in['pred'] == 'naive']
    if 'distance_to_germline' in data_in.columns.values.tolist():
        root_candidate = root_candidate.loc[root_candidate['distance_to_germline'] == min(root_candidate['distance_to_germline'])]
    else:
        print("distance_to_germline does not considered in selecting root due to absence of corresponding information")

    return root_candidate['sequence_id'].tolist()


def reroot(args, root_id):
    for root in root_id:
        run_id = 'reroot_%s' % root
        cmd_igphyml = '%s -i %s -m HLP --root %s --run_id %s' % (args.igphyml_path, args.file_phylo, root, run_id)
        os.system(cmd_igphyml)