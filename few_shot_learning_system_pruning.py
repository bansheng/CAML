import os

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

# from meta_neural_network_architectures import VGGReLUNormNetwork
from inner_loop_optimizers import LSLRGradientDescentLearningRule
from models.meta_model_darts import MetaNASNetwork as MetaPruningNetwork
from models.meta_model_darts_adaption import MetaNetwork as MetaPrunedPretrainedNetwork

import meta_genotype

def set_torch_seed(seed):
    """
    Sets the pytorch seeds for current experiment run
    :param seed: The seed (int)
    :return: A random number generator to use
    """
    rng = np.random.RandomState(seed=seed)
    torch_seed = rng.randint(0, 999999)
    torch.manual_seed(seed=torch_seed)

    return rng


class MAMLFewShotClassifier(nn.Module):
    def __init__(self, im_shape, device, args):
        """
        Initializes a MAML few shot learning system
        :param im_shape: The images input size, in batch, c, h, w shape
        :param device: The device to use to use the model on.
        :param args: A namedtuple of arguments specifying various hyperparameters.
        """
        super(MAMLFewShotClassifier, self).__init__()
        self.args = args
        self.device = device
        self.batch_size = args.batch_size
        self.use_cuda = args.use_cuda
        self.im_shape = im_shape
        self.current_epoch = 0

        self.rng = set_torch_seed(seed=args.seed)

        if not self.args.retrain:
            print('search for optimal arch for meta learning')
            self.classifier = MetaPruningNetwork(args=args, in_channels=im_shape[1], init_channels=args.init_channels,
                num_classes=self.args.num_classes_per_set, layers=self.args.layers, device=device).to(device=self.device)

        else: # for retrain
            print('retraining the arch', args.arch, eval('meta_genotype.{}'.format(args.arch)))
            self.classifier = MetaPrunedPretrainedNetwork(args=args, genotype=eval('meta_genotype.{}'.format(args.arch)), in_channels=im_shape[1], init_channels=args.init_channels, num_classes=self.args.num_classes_per_set, device=device).to(device=self.device)


        # self.task_learning_rate = args.task_learning_rate #0.1
        # self.task_learning_rate = args.init_inner_loop_learning_rate #0.1

        self.inner_loop_optimizer = LSLRGradientDescentLearningRule(
            device=device,
            init_learning_rate=args.init_inner_loop_learning_rate,
            init_learning_rate_arch=args.init_inner_arch_loop_learning_rate,
            total_num_inner_loop_steps=self.args.number_of_training_steps_per_iter,
            use_learnable_learning_rates=self.args.learnable_per_layer_per_step_inner_loop_learning_rate
        )
        if self.args.retrain: # only weights
            self.inner_loop_optimizer.initialise(
                names_weights_dict=self.get_inner_loop_parameter_dict(
                    params=self.classifier.named_parameters(),
                )
            )
        else: # arch + weights
            self.inner_loop_optimizer.initialise(
                names_weights_dict=self.get_inner_loop_parameter_dict(
                    params=self.classifier.named_parameters(),
                    arch_params=self.classifier.named_arch_parameters()
                )
            )

        self.use_cuda = args.use_cuda
        self.device = device
        self.args = args
        self.to(device)

        # print("Inner Loop parameters")
        # for key, value in self.inner_loop_optimizer.named_parameters():
        #     print(key, value.shape)

        # print("Outer Loop parameters")
        # for name, param in self.named_parameters():
        #     if param.requires_grad:
        #         print(name, param.shape, param.device, param.requires_grad)

        self.optimizer = optim.Adam(self.trainable_parameters(), lr=args.meta_learning_rate, amsgrad=False)
        if not self.args.retrain: # train the arch
            self.optimizer_arch = optim.Adam(self.classifier.arch_parameters(),
            lr=args.init_meta_arch_loop_learning_rate,
            betas=(0.5, 0.999),
            weight_decay=args.meta_arch_weights_decay
            )
            self.scheduler_arch = optim.lr_scheduler.CosineAnnealingLR(optimizer=self.optimizer_arch, T_max=self.args.total_epochs, eta_min=self.args.min_arch_learning_rate)

        self.scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer=self.optimizer, T_max=self.args.total_epochs,
                                                              eta_min=self.args.min_learning_rate)

        self.device = torch.device('cpu')
        if torch.cuda.is_available():
            # print('torch.cuda.device_count() = ', torch.cuda.device_count())
            if torch.cuda.device_count() > 1:
                self.to(torch.cuda.current_device())
                self.classifier = nn.DataParallel(module=self.classifier)
            else:
                self.to(torch.cuda.current_device())

            self.device = torch.cuda.current_device()

    def get_per_step_loss_importance_vector(self):
        """
        Generates a tensor of dimensionality (num_inner_loop_steps) indicating the importance of each step's target
        loss towards the optimization loss.
        :return: A tensor to be used to compute the weighted average of the loss, useful for
        the MSL (Multi Step Loss) mechanism.
        """
        loss_weights = np.ones(shape=(self.args.number_of_training_steps_per_iter)) * (
                1.0 / self.args.number_of_training_steps_per_iter)
        decay_rate = 1.0 / self.args.number_of_training_steps_per_iter / self.args.multi_step_loss_num_epochs
        min_value_for_non_final_losses = 0.03 / self.args.number_of_training_steps_per_iter
        for i in range(len(loss_weights) - 1):
            curr_value = np.maximum(loss_weights[i] - (self.current_epoch * decay_rate), min_value_for_non_final_losses)
            loss_weights[i] = curr_value

        curr_value = np.minimum(
            loss_weights[-1] + (self.current_epoch * (self.args.number_of_training_steps_per_iter - 1) * decay_rate),
            1.0 - ((self.args.number_of_training_steps_per_iter - 1) * min_value_for_non_final_losses))
        loss_weights[-1] = curr_value
        loss_weights = torch.Tensor(loss_weights).to(device=self.device)
        return loss_weights

    def get_inner_loop_parameter_dict(self, params=None, arch_params=None):
        """
        Returns a dictionary with the parameters to use for inner loop updates.
        :param params: A dictionary of the network's parameters.
        :return: A dictionary of the parameters to use for the inner loop optimization process.
        """
        param_dict = dict()
        if params is not None:
            for name, param in params:
                if param.requires_grad:
                    if self.args.enable_inner_loop_optimizable_bn_params: # enable bn params to be optimized
                        param_dict[name] = param.to(device=self.device)
                        # param_dict[name] = param
                    else:
                        if "norm_layer" not in name:
                            param_dict[name] = param.to(device=self.device)
                            # param_dict[name] = param
        if arch_params is not None:
            for name, param in arch_params:
                if param.requires_grad:
                    param_dict[name] = param.to(device=self.device)
        return param_dict

    def apply_inner_loop_update(self, loss, names_weights_copy, use_second_order, current_step_idx):
        """
        Applies an inner loop update given current step's loss, the weights to update, a flag indicating whether to use
        second order derivatives and the current step's index.
        :param loss: Current step's loss with respect to the support set.
        :param names_weights_copy: A dictionary with names to parameters to update.
        :param use_second_order: A boolean flag of whether to use second order derivatives.
        :param current_step_idx: Current step's index.
        :return: A dictionary with the updated weights (name, param)
        """
        num_gpus = torch.cuda.device_count()
        if num_gpus > 1:
            self.classifier.module.zero_grad(params=names_weights_copy)
        else:
            self.classifier.zero_grad(params=names_weights_copy)

        grads = torch.autograd.grad(loss, names_weights_copy.values(),
                                    create_graph=use_second_order, allow_unused=True)
        names_grads_copy = dict(zip(names_weights_copy.keys(), grads))

        names_weights_copy = {key: value[0] for key, value in names_weights_copy.items()}

        for key, grad in names_grads_copy.items():
            if grad is None: # some grads are not needed.
                raise Exception('Grads not found for inner loop parameter', key)
            else:
                names_grads_copy[key] = names_grads_copy[key].sum(dim=0)


        names_weights_copy = self.inner_loop_optimizer.update_params(names_weights_dict=names_weights_copy,
                                                                     names_grads_wrt_params_dict=names_grads_copy,
                                                                     num_step=current_step_idx)

        num_devices = torch.cuda.device_count() if torch.cuda.is_available() else 1
        names_weights_copy = {
            name.replace('module.', ''): value.unsqueeze(0).repeat(
                [num_devices] + [1 for i in range(len(value.shape))]) for
            name, value in names_weights_copy.items()}

        return names_weights_copy

    def get_across_task_loss_metrics(self, total_losses, total_accuracies):
        losses = dict()

        losses['loss'] = torch.mean(torch.stack(total_losses)) #use loss average
        losses['accuracy'] = np.mean(total_accuracies)

        return losses

    def forward(self, data_batch, epoch, use_second_order, use_multi_step_loss_optimization, num_steps, training_phase, param='weight'):
        """
        Runs a forward outer loop pass on the batch of tasks using the MAML/++ framework.
        :param data_batch: A data batch containing the support and target sets.
        :param epoch: Current epoch's index
        :param use_second_order: A boolean saying whether to use second order derivatives.
        :param use_multi_step_loss_optimization: Whether to optimize on the outer loop using just the last step's
        target loss (True) or whether to use multi step loss which improves the stability of the system (False)
        :param num_steps: Number of inner loop steps.
        :param training_phase: Whether this is a training phase (True) or an evaluation phase (False)
        :param param: 0 for weights. 1 for alpha
        :return: A dictionary with the collected losses of the current outer forward propagation.
        """
        x_support_set, x_target_set, y_support_set, y_target_set = data_batch

        total_losses = []
        total_accuracies = []
        per_task_target_preds = [[] for i in range(len(x_target_set))]
        self.classifier.zero_grad()
        for task_id, (x_support_set_task, y_support_set_task, x_target_set_task, y_target_set_task) in \
                enumerate(zip(x_support_set,
                              y_support_set,
                              x_target_set,
                              y_target_set)):
            task_losses = []
            task_accuracies = []
            per_step_loss_importance_vectors = self.get_per_step_loss_importance_vector()

            names_weights_copy = None
            if param == 'weight':
                names_weights_copy = self.get_inner_loop_parameter_dict(
                    self.classifier.named_parameters(),
                )
            else: # for alpha
                names_weights_copy = self.get_inner_loop_parameter_dict(
                    arch_params=self.classifier.named_arch_parameters(),
                )

            num_devices = torch.cuda.device_count() if torch.cuda.is_available() else 1

            names_weights_copy = {
                name.replace('module.', ''): value.unsqueeze(0).repeat(
                    [num_devices] + [1 for i in range(len(value.shape))]) for
                name, value in names_weights_copy.items()}

            n, s, c, h, w = x_target_set_task.shape

            x_support_set_task = x_support_set_task.view(-1, c, h, w)
            y_support_set_task = y_support_set_task.view(-1)
            x_target_set_task = x_target_set_task.view(-1, c, h, w)
            y_target_set_task = y_target_set_task.view(-1)

            task_max_acc = 0.0
            for num_step in range(num_steps):

                support_loss, support_preds = self.net_forward(x=x_support_set_task,
                                                               y=y_support_set_task,
                                                               weights=names_weights_copy,
                                                               backup_running_statistics=
                                                               True if (num_step == 0) else False,
                                                               training=True, num_step=num_step)

                names_weights_copy = self.apply_inner_loop_update(loss=support_loss,
                                                                  names_weights_copy=names_weights_copy,
                                                                  use_second_order=use_second_order,
                                                                  current_step_idx=num_step)

                if use_multi_step_loss_optimization and training_phase and epoch < self.args.multi_step_loss_num_epochs:
                    target_loss, target_preds = self.net_forward(x=x_target_set_task,
                                                                 y=y_target_set_task, weights=names_weights_copy,
                                                                 backup_running_statistics=False, training=True,
                                                                 num_step=num_step)

                    task_losses.append(per_step_loss_importance_vectors[num_step] * target_loss)
                else:
                    if num_step == (self.args.number_of_training_steps_per_iter - 1):
                        target_loss, target_preds = self.net_forward(x=x_target_set_task,
                                                                    y=y_target_set_task, weights=names_weights_copy,
                                                                    backup_running_statistics=False, training=True,
                                                                    num_step=num_step)
                        task_losses.append(target_loss)

            per_task_target_preds[task_id] = target_preds.detach().cpu().numpy()
            _, predicted = torch.max(target_preds.data, 1)

            accuracy = predicted.float().eq(y_target_set_task.data.float()).cpu().float()
            task_losses = torch.sum(torch.stack(task_losses))
            total_losses.append(task_losses)
            total_accuracies.extend(accuracy)

            if not training_phase:
                self.classifier.restore_backup_stats()

        losses = self.get_across_task_loss_metrics(total_losses=total_losses,
                                                   total_accuracies=total_accuracies)

        # for idx, item in enumerate(per_step_loss_importance_vectors):
        #     losses['loss_importance_vector_{}'.format(idx)] = item.detach().cpu().numpy()

        return losses, per_task_target_preds

    def net_forward(self, x, y, weights, backup_running_statistics, training, num_step):
        """
        A base model forward pass on some data points x. Using the parameters in the weights dictionary. Also requires
        boolean flags indicating whether to reset the running statistics at the end of the run (if at evaluation phase).
        A flag indicating whether this is the training session and an int indicating the current step's number in the
        inner loop.
        :param x: A data batch of shape b, c, h, w
        :param y: A data targets batch of shape b, n_classes
        :param weights: A dictionary containing the weights to pass to the network.
        :param backup_running_statistics: A flag indicating whether to reset the batch norm running statistics to their
         previous values after the run (only for evaluation)
        :param training: A flag indicating whether the current process phase is a training or evaluation.
        :param num_step: An integer indicating the number of the step in the inner loop.
        :return: the crossentropy losses with respect to the given y, the predictions of the base model.
        """
        preds = self.classifier.forward(x=x, params=weights,
                                        training=training,
                                        backup_running_statistics=backup_running_statistics, num_step=num_step,
                                        )

        loss = F.cross_entropy(input=preds, target=y)

        return loss, preds

    def trainable_parameters(self):
        """
        Returns an iterator over the trainable parameters of the model.
        """
        for param in self.parameters():
            if param.requires_grad:
                yield param

    def train_forward_prop(self, data_batch, epoch, param='weight'):
        """
        Runs an outer loop forward prop using the meta-model and base-model.
        :param data_batch: A data batch containing the support set and the target set input, output pairs.
        :param epoch: The index of the currrent epoch.
        :return: A dictionary of losses for the current step.
        """
        losses, per_task_target_preds = self.forward(data_batch=data_batch, epoch=epoch,
                                                     use_second_order=self.args.second_order and
                                                                      epoch > self.args.first_order_to_second_order_epoch,
                                                     use_multi_step_loss_optimization=self.args.use_multi_step_loss_optimization,
                                                     num_steps=self.args.number_of_training_steps_per_iter,
                                                     training_phase=True, param=param)
        return losses, per_task_target_preds

    def train_arch_forward_prop(self, data_batch, epoch, param='arch'):
        """
        Runs an outer loop forward prop using the meta-model and base-model.
        :param data_batch: A data batch containing the support set and the target set input, output pairs.
        :param epoch: The index of the currrent epoch.
        :return: A dictionary of losses for the current step.
        """
        losses, per_task_target_preds = self.forward(data_batch=data_batch, epoch=epoch,
                                                    use_second_order=False,
                                                    use_multi_step_loss_optimization=self.args.use_multi_step_loss_optimization,
                                                    num_steps=self.args.number_of_training_steps_per_iter,
                                                    training_phase=True, param=param)
        return losses, per_task_target_preds

    def evaluation_forward_prop(self, data_batch, epoch):
        """
        Runs an outer loop evaluation forward prop using the meta-model and base-model.
        :param data_batch: A data batch containing the support set and the target set input, output pairs.
        :param epoch: The index of the currrent epoch.
        :return: A dictionary of losses for the current step.
        """
        losses, per_task_target_preds = self.forward(data_batch=data_batch, epoch=epoch, use_second_order=False,
                                                     use_multi_step_loss_optimization=True,
                                                     num_steps=self.args.number_of_evaluation_steps_per_iter,
                                                     training_phase=False)

        return losses, per_task_target_preds

    def meta_update_weights(self, loss):
        """
        Applies an outer loop update on the meta-parameters of the model.
        :param loss: The current crossentropy loss.
        """
        self.optimizer.zero_grad()
        # self.optimizer_arch.zero_grad()

        loss.backward()
        if 'imagenet' in self.args.dataset_name:
            for name, param in self.classifier.named_parameters():
                if param.requires_grad:
                    param.grad.data.clamp_(-10, 10)  # not sure if this is necessary, more experiments are needed

        self.optimizer.step()
        # self.optimizer_arch.step()

    def meta_update_alphas(self, loss):
        """
        Applies an outer loop update on the meta-parameters of the model.
        :param loss: The current crossentropy loss.
        """
        self.optimizer_arch.zero_grad()
        loss.backward()
        self.optimizer_arch.step()

    def batch_helper(self, data_batch, split=False):
        x_support_set, x_target_set, y_support_set, y_target_set = data_batch

        x_support_set = torch.Tensor(x_support_set).float().to(device=self.device)
        x_target_set = torch.Tensor(x_target_set).float().to(device=self.device)
        y_support_set = torch.Tensor(y_support_set).long().to(device=self.device)
        y_target_set = torch.Tensor(y_target_set).long().to(device=self.device)

        if not split:
            data_batch = (x_support_set, x_target_set, y_support_set, y_target_set)
            return data_batch
        else: #split the data
            task_num_split = len(x_target_set) // 2
            data_batch_arch = (x_support_set[:task_num_split], x_target_set[:task_num_split], y_support_set[:task_num_split], y_target_set[:task_num_split])
            data_batch_weight = (x_support_set[task_num_split:], x_target_set[task_num_split:], y_support_set[task_num_split:], y_target_set[task_num_split:])
            return data_batch_arch, data_batch_weight


    def run_train_iter(self, data_batch, sample_idx, epoch):
        """
        Runs an outer loop update step on the meta-model's parameters.
        :param data_batch: input data batch containing the support set and target set input, output pairs
        :param epoch: the index of the current epoch
        :return: The losses of the ran iteration.
        """
        epoch = int(epoch)

        # self.step = sample_idx
        self.scheduler.step(epoch=epoch)
        if not self.args.retrain:
            self.scheduler_arch.step(epoch=epoch)
        if self.current_epoch != epoch:
            self.current_epoch = epoch
            # print('learning_rate = ', self.scheduler.get_last_lr()[0])
            # self.scheduler.step()

        if not self.training:
            self.train()


        if not self.args.retrain: # search process
            data_batch_arch, data_batch_weight = self.batch_helper(data_batch, split=True)

            train_prop_weight = self.train_arch_forward_prop
            if epoch > self.args.min_search_epoch:
                if torch.sum(self.classifier.normal_candidate_flags.int()) > 0:
                    losses, _ = self.train_arch_forward_prop(data_batch=data_batch_arch, epoch=epoch, param='arch')
                    self.meta_update_alphas(loss=losses['loss'])
                    self.optimizer.zero_grad()
                    self.optimizer_arch.zero_grad()
                    self.zero_grad()
                else: # no need for optimize arch
                    train_prop_weight = self.train_forward_prop

            losses, per_task_target_preds = train_prop_weight(data_batch=data_batch_weight, epoch=epoch, param='weight')
            self.meta_update_weights(loss=losses['loss'])
            losses['learning_rate'] = self.scheduler.get_last_lr()[0]
            self.optimizer.zero_grad()
            self.optimizer_arch.zero_grad()
            self.zero_grad()

        else: # different batch
            data_batch = self.batch_helper(data_batch)
            losses, per_task_target_preds = self.train_forward_prop(data_batch=data_batch, epoch=epoch, param='weight')
            self.meta_update_weights(loss=losses['loss'])
            losses['learning_rate'] = self.scheduler.get_last_lr()[0]
            # print("current_learning_rate =", losses['learning_rate'])
            self.optimizer.zero_grad()
            self.zero_grad()


        return losses, per_task_target_preds

    def run_test_iter(self, data_batch):
        """
        Runs an outer loop evaluation step on the meta-model's parameters.
        :param data_batch: input data batch containing the support set and target set input, output pairs
        :param epoch: the index of the current epoch
        :return: The losses of the ran iteration.
        """

        if self.training:
            self.eval()

        x_support_set, x_target_set, y_support_set, y_target_set = data_batch

        x_support_set = torch.Tensor(x_support_set).float().to(device=self.device)
        x_target_set = torch.Tensor(x_target_set).float().to(device=self.device)
        y_support_set = torch.Tensor(y_support_set).long().to(device=self.device)
        y_target_set = torch.Tensor(y_target_set).long().to(device=self.device)

        data_batch = (x_support_set, x_target_set, y_support_set, y_target_set)

        losses, per_task_target_preds = self.evaluation_forward_prop(data_batch=data_batch, epoch=self.current_epoch)

        # losses['loss'].backward() # uncomment if you get the weird memory error
        # self.zero_grad()
        # self.optimizer.zero_grad()

        return losses, per_task_target_preds

    def run_check_iter(self, data_batch, logging):
        """
        check weather all the weights still require grad.
        """
        x_support_set, x_target_set, y_support_set, y_target_set = data_batch

        x_support_set = torch.Tensor(x_support_set).float().to(device=self.device)
        x_target_set = torch.Tensor(x_target_set).float().to(device=self.device)
        y_support_set = torch.Tensor(y_support_set).long().to(device=self.device)
        y_target_set = torch.Tensor(y_target_set).long().to(device=self.device)

        if not self.training:
            self.train()

        for task_id, (x_support_set_task, y_support_set_task, x_target_set_task, y_target_set_task) in \
                enumerate(zip(x_support_set,
                              y_support_set,
                              x_target_set,
                              y_target_set)):
            names_weights_copy = self.get_inner_loop_parameter_dict(self.classifier.named_parameters())

            num_devices = torch.cuda.device_count() if torch.cuda.is_available() else 1
            names_weights_copy = {
                name.replace('module.', ''): value.unsqueeze(0).repeat(
                    [num_devices] + [1 for i in range(len(value.shape))]) for
                name, value in names_weights_copy.items()}

            n, s, c, h, w = x_target_set_task.shape

            x_support_set_task = x_support_set_task.view(-1, c, h, w)
            y_support_set_task = y_support_set_task.view(-1)
            x_target_set_task = x_target_set_task.view(-1, c, h, w)
            y_target_set_task = y_target_set_task.view(-1)

            num_step = 0
            target_loss, target_preds = self.net_forward(x=x_target_set_task,
                                                        y=y_target_set_task, weights=names_weights_copy,
                                                        backup_running_statistics=False, training=True,
                                                        num_step=num_step)

            self.optimizer.zero_grad()

            target_loss.backward()
            for name, param in self.classifier.named_parameters():
            # for name, param in self.named_parameters():
                if param.requires_grad:
                    if param.grad is not None:
                        # print(name, param.grad.shape)
                        if torch.sum(param.grad) == 0:
                            if 'op' in name:
                                logging.info('{} requires_grad = False'.format(name))
                                param.requires_grad = False
                    else:
                        if 'op' in name:
                            logging.info('{} requires_grad = False'.format(name))
                            param.requires_grad = False
            self.optimizer.zero_grad()
            self.zero_grad()
            break # only one iter

    def save_model(self, model_save_dir, state):
        """
        Save the network parameter state and experiment state dictionary.
        :param model_save_dir: The directory to store the state at.
        :param state: The state containing the experiment state and the network. It's in the form of a dictionary
        object.
        """
        state['network'] = self.state_dict()
        state['classifier'] = self.classifier.state_dict()
        torch.save(state, f=model_save_dir)

    def load_model(self, model_save_dir, model_name, model_idx):
        """
        Load checkpoint and return the state dictionary containing the network state params and experiment state.
        :param model_save_dir: The directory from which to load the files.
        :param model_name: The model_name to be loaded from the direcotry.
        :param model_idx: The index of the model (i.e. epoch number or 'latest' for the latest saved model of the current
        experiment)
        :return: A dictionary containing the experiment state and the saved model parameters.
        """
        filepath = os.path.join(model_save_dir, "{}_{}".format(model_name, model_idx))
        state = torch.load(filepath)
        state_dict_loaded = state['network']
        if not self.args.retrain:
            self.classifier.load_alphas(state)
        self.load_state_dict(state_dict=state_dict_loaded)
        return state

    def load_pretrained_model_init(self, pretrained_mode_path):
        state = torch.load(pretrained_mode_path)
        state_dict_loaded = state['network']
        state_dict_loaded = {k[len('classifier.'):]: v for k, v in state_dict_loaded.items() \
                            if k.startswith('classifier.') and k[len('classifier.'):] in self.classifier.state_dict()}
        # pretrained_model = dict()
        for k, v in state_dict_loaded.items():
            if self.args.per_step_bn_statistics:
                if 'norm' in k and len(v.shape) == 1:
                    # print(k)
                    state_dict_loaded[k] = v.unsqueeze(0).repeat(
                    [self.args.number_of_training_steps_per_iter] + [1 for i in range(len(v.shape))])
            else:
                if 'norm' in k and len(v.shape) == 2:
                    # unuse the per_step_bn_statistics
                    state_dict_loaded[k] = v[0]
        self.classifier.load_state_dict(state_dict_loaded)

