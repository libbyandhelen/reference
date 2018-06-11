import math
import shutil

import os

import numpy as np
import torch
from torch import nn
from torch.autograd import Variable

import features
import go
import symmetries
import preprocessing

EXAMPLES_PER_GENERATION = 100000

TRAIN_BATCH_SIZE = 16


class DualNetwork():
    def __init__(self, save_file, **hparams):
        self.save_file = save_file
        self.hparams = get_default_hyperparams(**hparams)
        self.inference_input = None
        self.inference_output = None
        # config = tf.ConfigProto()
        # config.gpu_options.allow_growth = True
        # self.sess = tf.Session(graph=tf.Graph(), config=config)
        self.model = None
        self.initialize_graph()

    def initialize_graph(self):
        self.model = Model(self.hparams)
        if self.save_file is not None:
            self.model.load_state_dict(torch.load(self.save_file))

    def run(self, position, use_random_symmetry=True):
        probs, values = self.run_many([position],
                                      use_random_symmetry=use_random_symmetry)
        return probs[0], values[0]

    def run_many(self, positions, use_random_symmetry=True):
        processed = list(map(features.extract_features, positions))
        if use_random_symmetry:
            syms_used, processed = symmetries.randomize_symmetries_feat(
                processed)

        processed = np.array(processed)
        processed = np.moveaxis(processed, -1, 1)
        processed = torch.from_numpy(processed)
        probabilities, value = self.model(processed.float())
        if use_random_symmetry:
            probabilities = symmetries.invert_symmetries_pi(
                syms_used, probabilities)

        return probabilities, value


class Model(nn.Module):
    def __init__(self, params):
        super(Model, self).__init__()
        self.params = params
        self.initial_layer = nn.Sequential(
            nn.Conv2d(
                in_channels=features.NEW_FEATURES_PLANES,
                out_channels=params['k'],
                kernel_size=(3, 3),
                padding=(3 - 1) / 2,
            ),
            nn.BatchNorm2d(
                num_features=params['k'],
                eps=1e-5,
                momentum=.997,
            ),
            nn.ReLU(inplace=True),
        )
        self.res_layer = nn.Sequential(
            nn.Conv2d(
                in_channels=params['k'],
                out_channels=params['k'],
                kernel_size=(3, 3),
                padding=(3-1)/2,
            ),
            nn.BatchNorm2d(
                num_features=params['k'],
                eps=1e-5,
                momentum=.997,
            ),
            nn.ReLU(inplace=True),

            nn.Conv2d(
                in_channels=params['k'],
                out_channels=params['k'],
                kernel_size=(3, 3),
                padding=(3 - 1) / 2,
            ),
            nn.BatchNorm2d(
                num_features=params['k'],
                eps=1e-5,
                momentum=.997,
            ),
        )
        self.relu = nn.Sequential(
            nn.ReLU(inplace=True),
        )
        self.policy_conv = nn.Sequential(
            nn.Conv2d(
                in_channels=params['k'],
                out_channels=2,
                kernel_size=(1, 1),
                padding=0,
            ),
            nn.BatchNorm2d(
                num_features=params['k'],
                eps=1e-5,
                momentum=.997,
            ),
            nn.ReLU(inplace=True),
        )
        self.logits = nn.Sequential(
            nn.Linear(in_features=2*go.N*go.N, out_features=go.N * go.N + 1),
        )
        self.softmax = nn.Sequential(
            nn.Softmax(),
        )
        self.value_conv = nn.Sequential(
            nn.Conv2d(
                in_channels=params['k'],
                out_channels=1,
                kernel_size=(1, 1),
                padding=0,
            ),
            nn.BatchNorm2d(
                num_features=params['k'],
                eps=1e-5,
                momentum=.997,
            ),
            nn.ReLU(inplace=True),
        )
        self.fc_hidden = nn.Sequential(
            nn.Linear(in_features=go.N*go.N, out_features=params['fc_width']),
            nn.ReLU(inplace=True),

            nn.Linear(in_features=params['fc_width'], out_features=1),
            nn.Tanh()
        )

    def forward(self, features):
        initial_output = self.initial_layer(features)
        shared_output = initial_output

        # the shared stack
        for i in range(self.params['num_shared_layers']):
            tmp_output = self.res_layer(shared_output)
            print(tmp_output.shape)
            print(shared_output.shape)
            shared_output = self.relu(shared_output+tmp_output)

        # policy head
        policy_conv = self.policy_conv(shared_output)
        logits = self.logits(policy_conv.view(-1, 2*go.N*go.N))
        policy_output = self.softmax(logits)

        # value head
        value_conv = self.value_conv(shared_output)
        value_output = self.fc_hidden(value_conv.view(-1, go.N*go.N))

        return policy_output, value_output, logits

        # # train ops
        # loss = nn.CrossEntropyLoss()
        # policy_cost = torch.mean(loss(logits, labels['pi_tensor']))
        # value_cost = torch.mean((value_output - labels['value_tensor'])**2)
        #
        # combined_cost = policy_cost + value_cost
        # policy_entropy = -torch.mean(torch.sum(policy_output * torch.log(policy_output), dim=0))
        # opimizer = torch.optim.SGD(
        #     model.parameters(),
        #     lr=1.5e-6,
        #     momentum=self.params['momentum'],
        #     weight_decay=self.params['l2_strength'],
        # )


def get_default_hyperparams(**overrides):
    """Returns the hyperparams for the neural net.
    In other words, returns a dict whose parameters come from the AGZ
    paper:
      k: number of filters (AlphaGoZero used 256). We use 128 by
        default for a 19x19 go board.
      fc_width: Dimensionality of the fully connected linear layer
      num_shared_layers: number of shared residual blocks.  AGZ used both 19
        and 39. Here we use 19 because it's faster to train.
      l2_strength: The L2 regularization parameter.
      momentum: The momentum parameter for training
    """
    k = _round_power_of_two(go.N ** 2 / 3)  # width of each layer
    hparams = {
        'k': k,  # Width of each conv layer
        'fc_width': 2 * k,  # Width of each fully connected layer
        'num_shared_layers': go.N,  # Number of shared trunk layers
        'l2_strength': 1e-4,  # Regularization strength
        'momentum': 0.9,  # Momentum used in SGD
    }
    hparams.update(**overrides)
    return hparams


def _round_power_of_two(n):
    """Finds the nearest power of 2 to a number.
    Thus 84 -> 64, 120 -> 128, etc.
    """
    return 2 ** int(round(math.log(n, 2)))


def train(working_dir, tf_records, generation_num, **hparams):
    assert generation_num > 0, "Model 0 is random weights"
    hparams = get_default_hyperparams(**hparams)
    model = Model(hparams)

    loader = preprocessing.get_input_tensors(TRAIN_BATCH_SIZE, tf_records)
    optimizer = torch.optim.SGD(
        model.parameters(),
        lr=1.5e-6,
        momentum=hparams['momentum'],
        weight_decay=hparams['l2_strength'],
    )

    for epoch in range(10):
        for step, (features, pi, outcome) in enumerate(loader):
            features = Variable(features.float())
            pi = Variable(pi.float())
            outcome = Variable(outcome.float())

            policy_output, value_output, logits = model(features)

            loss = nn.CrossEntropyLoss()
            policy_cost = torch.mean(loss(logits, pi))
            value_cost = torch.mean((value_output - outcome)**2)

            combined_cost = policy_cost + value_cost
            policy_entropy = -torch.mean(torch.sum(policy_output * torch.log(policy_output), dim=0))

            optimizer.zero_grad()
            combined_cost.backward()
            optimizer.step()

            print("epoch: %s | step: %s | loss: %s" % (epoch, step, loss.data[0]))
        torch.save(model.state_dict(), working_dir)


def bootstrap(working_dir, **hparams):
    hparams = get_default_hyperparams(**hparams)

    estimator_initial_checkpoint_name = 'model.ckpt-1'
    print("working_dir", working_dir)
    save_file = os.path.join(working_dir, estimator_initial_checkpoint_name)
    model = Model(hparams)
    torch.save(model.state_dict(), save_file)
    return estimator_initial_checkpoint_name


def export_model(working_dir, model_name, model_path):
    shutil.copy2(os.path.join(working_dir, model_name), model_path)
    print("model_path", model_path)
