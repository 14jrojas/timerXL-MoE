import numpy as np
import torch
import matplotlib.pyplot as plt
import plotly.graph_objects as go
import pandas as pd
import torch.distributed as dist
plt.switch_backend('agg')


def adjust_learning_rate(optimizer, epoch, args):
    if args.lradj == 'type1':
        lr_adjust = {epoch: args.learning_rate * (0.1 ** epoch)}
    elif args.lradj == 'type2':
        lr_adjust = {epoch: args.learning_rate * (0.5 ** epoch)}
    elif args.lradj == 'type3':
        lr_adjust = {epoch: args.learning_rate * (0.9 ** epoch)}


    if epoch in lr_adjust.keys():
        lr = lr_adjust[epoch]
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr
        print('Updating learning rate to {}'.format(lr))


class EarlyStopping:
    def __init__(self, args, verbose=False, delta=0):
        self.patience = args.patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.inf
        self.delta = delta
        self.dp = args.dp
        self.ddp = args.ddp
        if self.ddp:
            self.local_rank = args.local_rank
        else:
            self.local_rank = None

    def __call__(self, val_loss, model, path):
        score = -val_loss
        if self.best_score is None:
            self.best_score = score
            if self.verbose:
                print(
                    f'Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}).')
            self.val_loss_min = val_loss
            if self.ddp:
                if self.local_rank == 0:
                    self.save_checkpoint(val_loss, model, path)
                dist.barrier()
            else:
                self.save_checkpoint(val_loss, model, path)
        elif score < self.best_score + self.delta:
            self.counter += 1
            print(
                f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            if self.ddp:
                if self.local_rank == 0:
                    self.save_checkpoint(val_loss, model, path)
                dist.barrier()
            else:
                self.save_checkpoint(val_loss, model, path)
            if self.verbose:
                print(
                    f'Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}).')
            self.val_loss_min = val_loss
            self.counter = 0

    def save_checkpoint(self, val_loss, model, path):
        if self.dp:
            model = model.module
        param_grad_dic = {
            k: v.requires_grad for (k, v) in model.named_parameters()
        }
        state_dict = model.state_dict()
        for k in list(state_dict.keys()):
            if k in param_grad_dic.keys() and not param_grad_dic[k]:
                # delete parameters that do not require gradient
                del state_dict[k]
        torch.save(state_dict, path + '/' + f'checkpoint.pth')

def visual(true, preds=None, name='./pic/test.pdf'):
    """
    Results visualization
    """
    plt.figure()
    plt.plot(true, label='GroundTruth', linewidth=2)
    if preds is not None:
        plt.plot(preds, label='Prediction', linewidth=2)
    plt.legend()
    plt.savefig(name, bbox_inches='tight')

def visual_with_lookback(true, preds=None, lookback=None, name='./pic/test.html'):
    """
    Results visualization with optional lookback window using Plotly
    Saves the output to an interactive HTML file.
    """
    true = np.array(true)
    fig = go.Figure()

    if lookback is not None:
        lookback = np.array(lookback)
        full_series_gt = np.concatenate([lookback, true])
        x_axis = np.arange(len(full_series_gt))

        if preds is not None:
            preds = np.array(preds)
            full_series_pred = np.concatenate([np.full(len(lookback), np.nan), preds])
            fig.add_trace(go.Scatter(x=x_axis, y=full_series_pred,
                                     mode='lines',
                                     name='Prediction'))

        fig.add_trace(go.Scatter(x=x_axis, y=full_series_gt,
                                 mode='lines',
                                 name='GroundTruth'))

    else:
        x_axis = np.arange(len(true))
        fig.add_trace(go.Scatter(x=x_axis, y=true,
                                 mode='lines',
                                 name='GroundTruth'))

        if preds is not None:
            preds = np.array(preds)
            fig.add_trace(go.Scatter(x=x_axis, y=preds,
                                     mode='lines',
                                     name='Prediction'))

    fig.update_layout(
        title='Forecast vs GroundTruth',
        xaxis_title='Time Step',
        yaxis_title='Value',
        legend=dict(x=0.01, y=0.99),
        template='seaborn',
        xaxis=dict(range=[2500, len(x_axis)])
    )

    fig.write_html(name)
    fig.write_image(name.replace('.html', '.png'))



# def visual_with_lookback(true, preds=None, lookback=None, name='./pic/test.pdf'):
#     """
#     Results visualization with optional lookback window
#     """
#     plt.figure()
#     if lookback is not None:
#         lookback = np.array(lookback)
#         full_series_gt = np.concatenate([lookback, true])
#         full_series_pred = np.concatenate([np.full(len(lookback), np.nan), preds]) if preds is not None else None
#         x_axis = np.arange(len(full_series_gt))

#         plt.plot(x_axis, full_series_gt, label='GroundTruth', linewidth=2)
#         if full_series_pred is not None:
#             plt.plot(x_axis, full_series_pred, label='Prediction', linewidth=2)
#         plt.plot(np.arange(len(lookback)), lookback, label='Lookback Input', linestyle='--', linewidth=2)
#     else:
#         plt.plot(true, label='GroundTruth', linewidth=2)
#         if preds is not None:
#             plt.plot(preds, label='Prediction', linewidth=2)

#     plt.legend()
#     plt.savefig(name, bbox_inches='tight')