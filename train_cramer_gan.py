# -*- coding: utf-8 -*-
"""Train Critic and Generator using Cramer GAN."""

import argparse
import chainer
import numpy as np
import os
from chainer import Variable, cuda, using_config
from chainer import computational_graph as graph
from chainer import functions as F
from datetime import datetime as dt

from batch_generator import ImageBatchGenerator
from commons import ModelOptimizerSet
from commons import load_module
from commons import init_model, init_optimizer
from commons import l2_norm


def parse_arguments():
    """Define and parse positional/optional arguments."""
    parser = argparse.ArgumentParser()

    parser.add_argument(
        'config',
        help='a python module in which networks, optimizers and hyper params are defined'
    )
    parser.add_argument(
        'dataset',
        help='a text file in which image files are listed'
    )
    parser.add_argument(
        '-c', '--computational_graph', action='store_true',
        help='if specified, build computational graph'
    )
    parser.add_argument(
        '-g', '--gpu', default=-1, type=int,
        help='GPU ID (negative value indicates CPU)'
    )
    parser.add_argument(
        '-o', '--output', default='./',
        help='a directory in which output files will be stored'
    )
    parser.add_argument(
        '-p', '--param_cri', default=None,
        help='trained parameters for Critic saved as serialized array (.npz | .h5)'
    )
    parser.add_argument(
        '-P', '--param_gen', default=None,
        help='trained parameters for Generator saved as serialized array (.npz | .h5)'
    )
    parser.add_argument(
        '-s', '--state_cri', default=None,
        help='optimizer state for Critic saved as serialized array (.npz | .h5)'
    )
    parser.add_argument(
        '-S', '--state_gen', default=None,
        help='optimizer state for Generator saved as serialized array (.npz | .h5)'
    )

    return parser.parse_args()


def f_critic(h, h_g):
    return l2_norm(h - h_g) - l2_norm(h)


if __name__ == '__main__':
    # parse arguments
    args = parse_arguments()
    config = load_module(args.config)
    out_dir = args.output
    gpu_id = args.gpu

    # make output directory, if needed
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)
        print('mkdir ' + out_dir)
    assert os.path.isdir(out_dir)

    # setup network model, optimizer, and constant values to control training
    z_vec_dim = config.Z_VECTOR_DIM
    batch_size = config.BATCH_SIZE
    update_max = config.UPDATE_MAX
    update_save_params = config.UPDATE_SAVE_PARAMS
    update_cri_per_gen = getattr(config, 'UPDATE_CRI_PER_GEN', 1)
    gp_lambda = getattr(config, 'LAMBDA', 10)

    model_cri = config.Critic()
    optimizer_cri = config.OPTIMIZER_CRI
    optimizer_cri.setup(model_cri)
    model_opt_set_cri = ModelOptimizerSet(model_cri, optimizer_cri)

    model_gen = config.Generator()
    optimizer_gen = config.OPTIMIZER_GEN
    optimizer_gen.setup(model_gen)
    model_opt_set_gen = ModelOptimizerSet(model_gen, optimizer_gen)

    # setup batch generator
    with open(args.dataset, 'r') as f:
        input_files = [line.strip() for line in f.readlines()]
    batch_generator = ImageBatchGenerator(input_files, batch_size,
                                          config.HEIGHT, config.WIDTH,
                                          channel=config.CHANNEL,
                                          shuffle=True,
                                          flip_h=getattr(config, 'FLIP_H', False))
    sample_num = batch_generator.n_samples

    # show some settings
    print('sample num = {}'.format(sample_num))
    print('mini-batch size = {}'.format(batch_size))
    print('max update count = {}'.format(update_max))
    print('updates per saving params = {}'.format(update_save_params))
    print('critic updates per generator update = {}'.format(update_cri_per_gen))
    print('lambda for gradient penalty = {}'.format(gp_lambda))

    # save or load initial parameters for Critic
    if not init_model(model_cri, param=args.param_cri):
        model_opt_set_cri.save_model('cri', out_dir=out_dir)
    # save or load initial optimizer state for Critic
    if not init_optimizer(optimizer_cri, state=args.state_cri):
        model_opt_set_cri.save_optimizer('cri', out_dir=out_dir)
    # save or load initial parameters for Generator
    if not init_model(model_gen, param=args.param_gen):
        model_opt_set_gen.save_model('gen', out_dir=out_dir)
    # save or load initial optimizer state for Generator
    if not init_optimizer(optimizer_gen, state=args.state_gen):
        model_opt_set_gen.save_optimizer('gen', out_dir=out_dir)

    # set current device and copy model to it
    xp = np
    if gpu_id >= 0:
        cuda.get_device(gpu_id).use()
        model_gen.to_gpu(device=gpu_id)
        model_cri.to_gpu(device=gpu_id)
        xp = cuda.cupy

    # set global configuration
    chainer.global_config.enable_backprop = True
    chainer.global_config.enable_double_backprop = True
    chainer.global_config.train = True

    print('** chainer global configuration **')
    chainer.global_config.show()

    initial_t = optimizer_gen.t
    sum_count = 0
    sum_loss_cri = 0.
    sum_loss_gen = 0.

    # training loop
    while optimizer_gen.t < update_max:

        for idx in range(update_cri_per_gen):
            x_t = next(batch_generator)
            if gpu_id >= 0:
                x_t = cuda.to_gpu(x_t, device=gpu_id)

            z1 = xp.random.normal(loc=0., scale=1.,
                                  size=(batch_size, z_vec_dim)).astype('float32')
            z2 = xp.random.normal(loc=0., scale=1.,
                                  size=(batch_size, z_vec_dim)).astype('float32')

            if idx == (update_cri_per_gen - 1):
                x_g1 = model_gen(z1)
                x_g2 = model_gen(z2)
            else:
                with using_config('enable_backprop', False):
                    x_g1 = model_gen(z1)
                    x_g2 = model_gen(z2)

            h_t = model_cri(x_t)
            h_g1 = model_cri(x_g1.data)
            h_g2 = model_cri(x_g2.data)

            loss_s = F.sum(f_critic(h_t, h_g2)) - F.sum(f_critic(h_g1, h_g2))

            # calculate gradient penalty
            alpha = xp.random.uniform(low=0., high=1.,
                                      size=((batch_size,) +
                                            (1,) * (x_t.ndim - 1))
                                      ).astype('float32')
            interpolates = Variable((1. - alpha) * x_t + alpha * x_g1.data)
            h_i = model_cri(interpolates)
            gradients = chainer.grad([f_critic(h_i, h_g2)],
                                     [interpolates],
                                     enable_double_backprop=True)[0]
            slopes = l2_norm(gradients)
            gradient_penalty = gp_lambda * F.sum((slopes - 1.) * (slopes - 1.))

            loss_cri = gradient_penalty - loss_s
            loss_cri /= batch_size

            # update Critic
            model_cri.cleargrads()
            loss_cri.backward()
            optimizer_cri.update()

            sum_loss_cri += float(loss_cri.data) / update_cri_per_gen

        h_t = model_cri(x_t)
        h_g1 = model_cri(x_g1)
        h_g2 = model_cri(x_g2)

        loss_gen = F.sum(l2_norm(h_t - h_g1)) + \
            F.sum(l2_norm(h_t - h_g2)) - F.sum(l2_norm(h_g1 - h_g2))
        loss_gen /= batch_size

        # update Generator
        model_gen.cleargrads()
        loss_gen.backward()
        optimizer_gen.update()

        sum_loss_gen += float(loss_gen.data)
        sum_count += 1

        # show losses
        print('{0}: update # {1:09d}: C loss = {2:6.4e}, G loss = {3:6.4e}'.format(
            str(dt.now()), optimizer_gen.t,
            float(loss_cri.data), float(loss_gen.data)))

        # output computational graph, if needed
        if args.computational_graph and optimizer_gen.t == (initial_t + 1):
            with open(os.path.join(out_dir, 'graph.dot'), 'w') as f:
                f.write(graph.build_computational_graph((loss_cri, loss_gen)).dump())
            print('graph generated')

        # show mean losses, save interim trained parameters and optimizer states
        if optimizer_gen.t % update_save_params == 0:
            print('{0}: mean of latest {1:06d} in {2:09d} updates : C loss = {3:7.5e}, G loss = {4:7.5e}'.format(
                str(dt.now()), sum_count, optimizer_gen.t, sum_loss_cri / sum_count, sum_loss_gen / sum_count))
            sum_count = 0
            sum_loss_cri = 0.
            sum_loss_gen = 0.

            model_opt_set_gen.save('gen', out_dir=out_dir)
            model_opt_set_cri.save('cri', out_dir=out_dir)

    # save final trained parameters and optimizer states
    model_opt_set_gen.save('gen', out_dir=out_dir)
    model_opt_set_cri.save('cri', out_dir=out_dir)
