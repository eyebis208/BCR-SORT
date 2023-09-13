import os
import logging
import pandas as pd
import numpy as np
import torch

from tqdm import tqdm
from functools import partial
from torch.utils.data import DataLoader
from torch import optim
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts
from pathlib import Path
from model import load_model
from data_loader import load_dataset
from data_utils import sequence_padding, generate_dict, decode_sequence, decode_categorical, generate_seq_mask


def predict(args):
    file_in = args.input_file
    file_out = args.output_file
    if file_out is None:
        file_out = file_in.replace('.csv', '_bcr-sort.csv')

    model_path = args.model_path
    device = args.device
    if device is not None:
        torch.cuda.set_device(device)
        print("Use GPU: {} for loading model".format(torch.cuda.current_device()))

    # load data
    DatasetBCR = load_dataset(file_in, mode='predict')
    sequence_padder = partial(sequence_padding, mode='predict')
    DataLoaderBCR = DataLoader(DatasetBCR, batch_size=1024, num_workers=0, pin_memory=True, drop_last=False,
                               shuffle=False, collate_fn=sequence_padder)

    # load model
    model = load_model()
    model.load_state_dict(torch.load(model_path))
    model.to(device)
    model.eval()

    # setup output vars
    encoding_dict = generate_dict()
    ntest = len(DataLoaderBCR.dataset)
    num_class = 3
    pred = np.zeros((ntest, num_class))
    col_list = ['cdr3_aa', 'V_gene', 'J_gene', 'isotype', 'pred', 'p_naive', 'p_memory', 'p_diff']
    data_out = {}
    for col in col_list:
        data_out[col] = []

    # prediction
    dim_start = 0
    for i, dataloader in enumerate(DataLoaderBCR, 0):
        inputs, lengths, _ = dataloader
        sequence, vgene, jgene, isotype = inputs
        sequence = sequence.type(torch.cuda.LongTensor).to(device)
        vgene = vgene.type(torch.cuda.LongTensor).to(device)
        jgene = jgene.type(torch.cuda.LongTensor).to(device)
        isotype = isotype.type(torch.cuda.LongTensor).to(device)

        outputs, _, _ = model(sequence, vgene, jgene, isotype, lengths, False)

        # collect outputs
        dim_end = dim_start + outputs.shape[0]
        pred[dim_start:dim_end, ] = outputs.cpu().detach().numpy()

        sequence_out = decode_sequence(sequence.cpu().detach().tolist())
        vgene = vgene.squeeze().cpu().detach().tolist()
        jgene = jgene.squeeze().cpu().detach().tolist()
        isotype = isotype.squeeze().cpu().detach().tolist()

        if not isinstance(vgene, list):
            vgene = [vgene]
            jgene = [jgene]
            isotype = [isotype]

        vgene_out = [decode_categorical(f, encoding_dict['V_gene']) for f in vgene]
        jgene_out = [decode_categorical(f, encoding_dict['J_gene']) for f in jgene]
        isotype_out = [decode_categorical(f, encoding_dict['isotype']) for f in isotype]

        data_out['cdr3_aa'].extend(sequence_out)
        data_out['V_gene'].extend(vgene_out)
        data_out['J_gene'].extend(jgene_out)
        data_out['isotype'].extend(isotype_out)
        dim_start = dim_end

    # save outputs
    pred_out = np.argmax(pred, axis=1).tolist()
    data_out['pred'] = [decode_categorical(pred, encoding_dict['label']) for pred in pred_out]
    data_out['p_naive'] = pred[:, 0].tolist()
    data_out['p_memory'] = pred[:, 1].tolist()
    data_out['p_diff'] = pred[:, 2].tolist()

    Path(file_out).parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(data_out).to_csv(file_out, index=False)


def train(args):
    """

    :param dataset_dict:
    :param param_input:
    :param model:
    :param dir_out:
    :param seed:
    :param device:
    :param include_status:
    :return:
    """

    # set up vars
    file_in = args.input_file
    dir_out = args.output_dir
    if dir_out is None:
        dir_out = os.path.join(os.path.dirname(file_in), 'train')

    device = args.device
    if device is not None:
        torch.cuda.set_device(device)
        print("Use GPU: {} for loading model".format(torch.cuda.current_device()))

    batch_size = args.batch_size
    learning_rate = args.learning_rate
    weight_decay = args.weight_decay
    scaling_loss_aux = args.loss_scaling
    num_epoch = args.num_epoch

    # file out
    Path(dir_out).mkdir(parents=True, exist_ok=True)
    file_wt = os.path.join(dir_out, "best_wt.pt")
    file_loss = os.path.join(dir_out, 'loss.csv')
    file_acc = os.path.join(dir_out, 'accuracy.csv')
    file_log = os.path.join(dir_out, 'log.log')
    logging.basicConfig(filename=file_log, level=logging.INFO)
    print("Project log directory: %s" % file_log)

    # load data
    TrainDatasetBCR, ValDatasetBCR, TestDatasetBCR = load_dataset(file_in, mode='train', test_ratio=0.01)
    sequence_padder = partial(sequence_padding, mode='train')
    TrainDataLoaderBCR = DataLoader(TrainDatasetBCR, batch_size=batch_size, num_workers=0, pin_memory=True,
                                    drop_last=True, shuffle=True, collate_fn=sequence_padder)
    ValDataLoaderBCR = DataLoader(ValDatasetBCR, batch_size=batch_size, num_workers=0, pin_memory=True,
                                  drop_last=True, shuffle=True, collate_fn=sequence_padder)
    TestDataLoaderBCR = DataLoader(TestDatasetBCR, batch_size=batch_size, num_workers=0, pin_memory=True,
                                   drop_last=True, shuffle=True, collate_fn=sequence_padder)

    # load model
    model = load_model()
    model.to(device)

    criterion = torch.nn.CrossEntropyLoss()
    optimizer_ft = optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    scheduler = CosineAnnealingWarmRestarts(optimizer_ft, T_0=10)

    phase_list = ['train', 'val', 'test']
    phase_loader = {'train': TrainDataLoaderBCR, 'val': ValDataLoaderBCR, 'test': TestDataLoaderBCR}
    epoch_loss_log = {'train': [], 'val': [], 'test': []}
    epoch_accuracy = {'train': [], 'val': [], 'test': []}
    best_acc_test = 0.0
    with tqdm(total=num_epoch) as pbar:
        for epoch in range(0, num_epoch):

            for phase in phase_list:

                if phase == 'train':
                    model.train()
                    grad_flag = True
                else:
                    model.eval()
                    grad_flag = False

                with torch.set_grad_enabled(grad_flag):

                    epoch_loss = 0.0
                    epoch_total = 0
                    epoch_correct = 0
                    batch_counter = 0

                    for i, dataloader in enumerate(phase_loader[phase], 0):
                        optimizer_ft.zero_grad()

                        inputs, lengths, labels = dataloader
                        sequence, vgene, jgene, isotype = inputs
                        sequence = sequence.type(torch.cuda.LongTensor).to(device)
                        vgene = vgene.type(torch.cuda.LongTensor).to(device)
                        jgene = jgene.type(torch.cuda.LongTensor).to(device)
                        isotype = isotype.type(torch.cuda.LongTensor).to(device)
                        labels = labels.to(device)

                        if phase == 'train':
                            sequence_masked, labels_aux = generate_seq_mask(sequence, lengths)
                            sequence = sequence_masked.to(device)
                            labels_aux = labels_aux.to(device)

                        outputs, outputs_aux, _ = model(sequence, vgene, jgene, isotype, lengths, True)
                        loss = criterion(outputs, torch.argmax(labels, dim=1))

                        if phase == 'train':
                            loss_aux = criterion(outputs_aux, labels_aux)
                            loss_total = loss + scaling_loss_aux * loss_aux

                            loss_total.backward()
                            optimizer_ft.step()
                            scheduler.step()

                        epoch_loss += loss.item()
                        epoch_total += outputs.shape[0]
                        pred = torch.argmax(outputs, dim=1)
                        epoch_correct += torch.sum(pred == torch.argmax(labels, dim=1))
                        batch_counter += 1

                        pbar.set_description("epoch: %d, mode: train, batch: %s, loss: %s, acc:%s" %
                                             (epoch + 1, str(batch_counter), str(loss.cpu().detach().item()),
                                              str((epoch_correct / epoch_total).cpu().detach().item())))

                loss_log = epoch_loss / batch_counter
                epoch_loss_log[phase].append(loss_log)

                acc = (epoch_correct / epoch_total).cpu().detach().item()
                epoch_accuracy[phase].append(acc)
                logging.info('[Epoch: %d, Mode: %s, Total: %d Accuracy: %.4f' % (epoch + 1, phase, epoch_total, acc))

                if phase == 'test' and acc > best_acc_test:
                    best_acc_test = acc
                    best_wt = model.state_dict()
                    torch.save(best_wt, file_wt)

            pbar.update()

    data_loss = pd.DataFrame({'train': epoch_loss_log['train'], 'val': epoch_loss_log['val'], 'test': epoch_loss_log['test']})
    data_loss.groupby(level=0).max().to_csv(file_loss, index_label='epoch')
    data_acc = pd.DataFrame({'train': epoch_accuracy['train'], 'val': epoch_accuracy['val'], 'test': epoch_accuracy['test']})
    data_acc.groupby(level=0).max().to_csv(file_acc, index_label='epoch')

