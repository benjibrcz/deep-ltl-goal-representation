import argparse
import json
import os
import pickle

import pandas as pd
import torch

import utils
from preprocessing import VOCAB, reset_vocab


class ModelStore:
    def __init__(self, env: str, name: str, seed: int):
        self.path = utils.get_experiment_path(env, name, seed)
        self.eval_results_path = utils.get_eval_results_path(env, name, seed)

    @classmethod
    def from_config(cls, config: argparse.Namespace) -> 'ModelStore':
        exp = config.experiment
        return cls(exp.env, exp.name, exp.seed)

    def save_training_status(self, status: dict[str, any]):
        torch.save(status, f'{self.path}/status.pth')

    def save_eval_training_status(self, status: dict[str, any]):
        torch.save(status, f'{self.path}/eval/{status["num_steps"]}.pth')

    def save_ltl_net(self, ltl_net: dict[str, any]):
        torch.save(ltl_net, f'{self.path}/ltl_net.pth')

    def save_vocab(self):
        with open(f'{self.path}/vocab.pkl', 'wb+') as f:
            pickle.dump(VOCAB, f)

    def load_vocab(self):
        reset_vocab()
        with open(f'{self.path}/vocab.pkl', 'rb') as f:
            v = pickle.load(f)
        VOCAB.update(v)

    def load_training_status(self, map_location=None) -> dict[str, any]:
        if not os.path.exists(f'{self.path}/status.pth'):
            raise FileNotFoundError(f'No training status found at {self.path}/status.pth')
        return torch.load(f'{self.path}/status.pth', map_location=map_location)

    def load_best_model(self, map_location=None) -> dict[str, any]:
        if not os.path.exists(self.eval_results_path):
            raise FileNotFoundError(f'No eval results found at {self.eval_results_path}')
        with open(self.eval_results_path) as f:
            df = pd.read_csv(f)
        # Get the best model by success rate and return
        best_model_steps = df[df['success_rate'] >= df['success_rate'].max() - 0.02].sort_values('return', ascending=False).iloc[0]['num_steps']
        print(best_model_steps)
        best_model_file = f'{self.path}/eval/{int(best_model_steps)}.pth'
        return torch.load(best_model_file, map_location=map_location)

    def load_eval_training_statuses(self, map_location=None) -> list[dict[str, any]]:
        eval_dir = f'{self.path}/eval'
        if not os.path.exists(eval_dir):
            raise FileNotFoundError(f'No eval models found at {eval_dir}')
        eval_models = []
        for file in os.listdir(eval_dir):
            eval_models.append(torch.load(f'{eval_dir}/{file}', map_location=map_location))
        final_model = self.load_training_status(map_location)
        eval_models.append(final_model)
        return sorted(eval_models, key=lambda x: x['num_steps'])

