import glob
import logging
import os
import sys
import time
from math import e

import numpy as np
from numpy.lib import utils
import torch
import tqdm
from tensorboardX import SummaryWriter
from torch import le

from meta_utils import count_parameters_in_KB, r_makedir
from utils.storage import (build_experiment_folder, save_statistics,
                           save_to_json)


class PruningExperimentBuilder(object):
    def __init__(self, args, data, model, device):
        """
        Initializes an experiment builder using a named tuple (args), a data provider (data), a meta learning system
        (model) and a device (e.g. gpu/cpu/n)
        :param args: A namedtuple containing all experiment hyperparameters
        :param data: A data provider of instance MetaLearningSystemDataLoader
        :param model: A meta learning system instance
        :param device: Device/s to use for the experiment
        """
        self.args, self.device = args, device

        self.model = model

        if self.args.retrain:
            self.args.save_dir = os.path.join(self.args.save_dir, self.args.arch)
            r_makedir(self.args.save_dir.split('/'))

        self.saved_models_filepath, self.logs_filepath, self.visual_filepath = build_experiment_folder(
            experiment_name=self.args.experiment_name,
            save_dir=self.args.save_dir,
            scripts_to_save=glob.glob("*.py") + glob.glob("*.sh") + glob.glob("*.json")
        )

        self._init_log()
        self.logging.info(str(args))
        self.logging.info("parameter size = {}KB".format(count_parameters_in_KB(self.model.classifier)))

        self.total_losses = dict()
        self.state = dict()
        self.state["best_val_acc"] = 0.0
        self.state["best_val_iter"] = 0
        self.state["current_iter"] = 0
        # self.state['current_iter'] = 0
        self.start_epoch = 0
        self.max_models_to_save = self.args.max_models_to_save
        self.create_summary_csv = False


        if self.args.continue_from_epoch == "from_scratch":
            self.create_summary_csv = True

        elif self.args.continue_from_epoch == "latest":
            checkpoint = os.path.join(self.saved_models_filepath, "train_model_latest")
            self.logging.info("attempting to find existing checkpoint")
            if os.path.exists(checkpoint):
                self.state = self.model.load_model(
                    model_save_dir=self.saved_models_filepath,
                    model_name="train_model",
                    model_idx="latest",
                )
                self.start_epoch = int(
                    self.state["current_iter"] / self.args.total_iter_per_epoch
                )
                self.logging.info("existing checkpoint found")
            elif self.args.retrain and self.args.use_pretrained_model:
                self.logging.info("try to load pretrained initialization")
                self.model.load_pretrained_model_init(self.args.pretrained_model)
                self.create_summary_csv = True
                self.logging.info("loaded pretrained initialization")
            else:
                self.args.continue_from_epoch = "from_scratch"
                self.create_summary_csv = True
                self.logging.info("existing checkpoint not found")

        elif int(self.args.continue_from_epoch) >= 0:
            self.state = self.model.load_model(
                model_save_dir=self.saved_models_filepath,
                model_name="train_model",
                model_idx=self.args.continue_from_epoch,
            )
            self.start_epoch = int(
                self.state["current_iter"] / self.args.total_iter_per_epoch
            )


        self.data = data(args=args, current_iter=self.state["current_iter"])

        self.logging.info(
            "train_seed {}, val_seed: {}, at start time".format(
                self.data.dataset.seed["train"], self.data.dataset.seed["val"]
            )
        )

        self.total_epochs_before_pause = self.args.total_epochs_before_pause
        self.state["best_epoch"] = int(
            self.state["best_val_iter"] / self.args.total_iter_per_epoch
        )
        self.epoch = int(self.state["current_iter"] / self.args.total_iter_per_epoch)
        self.augment_flag = (
            True if "omniglot" in self.args.dataset_name.lower() else False
        )
        self.start_time = time.time()
        self.epochs_done_in_this_run = 0
        self.logging.info(
            "current iter = {}, total iter = {}".format(
                self.state["current_iter"],
                int(self.args.total_iter_per_epoch * self.args.total_epochs),
            )
        )

    def _init_log(self):
        cur_time = time.strftime("%Y%m%d-%H%M%S")

        self.writer = SummaryWriter(self.visual_filepath)

        log_format = "%(asctime)s %(message)s"
        logging.basicConfig(
            stream=sys.stdout,
            level=logging.INFO,
            format=log_format,
            datefmt="%m/%d %I:%M:%S %p",
        )
        self.logging = logging.getLogger("Meta learning with Architecture Searching")
        fh = logging.FileHandler(
            os.path.join(
                self.args.save_dir, self.args.experiment_name, "log_{}.txt".format(cur_time)
            )
        )
        fh.setFormatter(logging.Formatter(log_format))
        self.logging.addHandler(fh)

    def build_summary_dict(self, total_losses, phase, summary_losses=None):
        """
        Builds/Updates a summary dict directly from the metric dict of the current iteration.
        :param total_losses: Current dict with total losses (not aggregations) from experiment
        :param phase: Current training phase
        :param summary_losses: Current summarised (aggregated/summarised) losses stats means, stdv etc.
        :return: A new summary dict with the updated summary statistics information.
        """
        if summary_losses is None:
            summary_losses = dict()

        for key in total_losses:
            summary_losses["{}_{}_mean".format(phase, key)] = np.mean(total_losses[key])
            summary_losses["{}_{}_std".format(phase, key)] = np.std(total_losses[key])

        return summary_losses

    def build_loss_summary_string(self, summary_losses):
        """
        Builds a progress bar summary string given current summary losses dictionary
        :param summary_losses: Current summary statistics
        :return: A summary string ready to be shown to humans.
        """
        output_update = ""
        for key, value in zip(
            list(summary_losses.keys()), list(summary_losses.values())
        ):
            # if "loss" in key or "accuracy" in key: # no vector
            if "mean" in key: # no vector
                value = float(value)
                output_update += "{}: {:.4f}, ".format(key, value)

        return output_update

    def merge_two_dicts(self, first_dict, second_dict):
        """Given two dicts, merge them into a new dict as a shallow copy."""
        z = first_dict.copy()
        z.update(second_dict)
        return z

    def train_iteration(
        self, train_sample, sample_idx, epoch_idx, total_losses, pbar_train
    ):   #current_iter
        """
        Runs a training iteration, updates the progress bar and returns the total and current epoch train losses.
        :param train_sample: A sample from the data provider
        :param sample_idx: The index of the incoming sample, in relation to the current training run.
        :param epoch_idx: The epoch index.
        :param total_losses: The current total losses dictionary to be updated.
        :param current_iter: The current training iteration in relation to the whole experiment.
        :param pbar_train: The progress bar of the training.
        :return: Updates total_losses, train_losses, current_iter
        """
        # train_sample, val_sample = train_val_sample
        x_support_set, x_target_set, y_support_set, y_target_set, seed = train_sample
        data_batch_train = (x_support_set, x_target_set, y_support_set, y_target_set)

        if sample_idx == 0:
            self.logging.info(
                "shape of data {} {} {} {}".format(
                    x_support_set.shape,
                    x_target_set.shape,
                    y_support_set.shape,
                    y_target_set.shape,
                )
            )
        losses, _ = self.model.run_train_iter(data_batch=data_batch_train,
                                            epoch=epoch_idx, sample_idx=sample_idx)

        for key, value in zip(list(losses.keys()), list(losses.values())):
            if key not in total_losses:
                total_losses[key] = [float(value)]
            else:
                total_losses[key].append(float(value))

        train_losses = self.build_summary_dict(total_losses=total_losses, phase="train")
        train_output_update = self.build_loss_summary_string(train_losses)

        pbar_train.update(1)
        pbar_train.set_description("training {} / {} -> {}".format(
            self.epoch, self.args.total_epochs, train_output_update))

        return train_losses, total_losses  # , current_iter

    def validation_iteration(self, val_sample, total_losses, phase, pbar_val):
        """
        Runs a validation iteration, updates the progress bar and returns the total and current epoch val losses.
        :param val_sample: A sample from the data provider
        :param total_losses: The current total losses dictionary to be updated.
        :param pbar_val: The progress bar of the val stage.
        :return: The updated val_losses, total_losses
        """
        x_support_set, x_target_set, y_support_set, y_target_set, seed = val_sample
        data_batch = (x_support_set, x_target_set, y_support_set, y_target_set)

        losses, _ = self.model.run_test_iter(data_batch=data_batch)
        for key, value in zip(list(losses.keys()), list(losses.values())):
            if key not in total_losses:
                total_losses[key] = [float(value)]
            else:
                total_losses[key].append(float(value))

        val_losses = self.build_summary_dict(total_losses=total_losses, phase=phase)
        val_output_update = self.build_loss_summary_string(val_losses)

        pbar_val.update(1)
        pbar_val.set_description(
            "val {} / {} -> {}".format(self.epoch, self.args.total_epochs, val_output_update)
        )

        return val_losses, total_losses

    def test_evaluation_iteration(
        self, val_sample, model_idx, sample_idx, per_model_per_batch_preds, pbar_test
    ):
        """
        Runs a validation iteration, updates the progress bar and returns the total and current epoch val losses.
        :param val_sample: A sample from the data provider
        :param total_losses: The current total losses dictionary to be updated.
        :param pbar_test: The progress bar of the val stage.
        :return: The updated val_losses, total_losses
        """
        x_support_set, x_target_set, y_support_set, y_target_set, seed = val_sample
        data_batch = (x_support_set, x_target_set, y_support_set, y_target_set)

        losses, per_task_preds = self.model.run_test_iter(data_batch=data_batch)

        # print(np.array(per_task_preds).shape)

        per_model_per_batch_preds[model_idx].extend(list(per_task_preds))

        test_output_update = self.build_loss_summary_string(losses)

        pbar_test.update(1)
        pbar_test.set_description(
            "test_phase {} -> {}".format(self.epoch, test_output_update)
        )

        return per_model_per_batch_preds


    def save_models(self, model, epoch, state):
        """
        Saves two separate instances of the current model. One to be kept for history and reloading later and another
        one marked as "latest" to be used by the system for the next epoch training. Useful when the training/val
        process is interrupted or stopped. Leads to fault tolerant training and validation systems that can continue
        from where they left off before.
        :param model: Current meta learning model of any instance within the few_shot_learning_system.py
        :param epoch: Current epoch
        :param state: Current model and experiment state dict.
        """
        model.save_model(
            model_save_dir=os.path.join(
                self.saved_models_filepath, "train_model_{}".format(int(epoch))
            ),
            state=state,
        )

        model.save_model(
            model_save_dir=os.path.join(
                self.saved_models_filepath, "train_model_latest"
            ),
            state=state,
        )

        self.logging.info("saved models to %s", self.saved_models_filepath)

    def pack_and_save_metrics(
        self, create_summary_csv, train_losses, val_losses, state
    ):
        """
        Given current epochs start_time, train losses, val losses and whether to create a new stats csv file, pack stats
        and save into a statistics csv file. Return a new start time for the new epoch.
        :param start_time: The start time of the current epoch
        :param create_summary_csv: A boolean variable indicating whether to create a new statistics file or
        append results to existing one
        :param train_losses: A dictionary with the current train losses
        :param val_losses: A dictionary with the currrent val loss
        :return: The current time, to be used for the next epoch.
        """
        epoch_summary_losses = self.merge_two_dicts(
            first_dict=train_losses, second_dict=val_losses
        )

        if "per_epoch_statistics" not in state:
            state["per_epoch_statistics"] = dict()

        for key, value in epoch_summary_losses.items():

            if key not in state["per_epoch_statistics"]:
                state["per_epoch_statistics"][key] = [value]
            else:
                state["per_epoch_statistics"][key].append(value)

        epoch_summary_string = self.build_loss_summary_string(epoch_summary_losses)
        epoch_summary_losses["epoch"] = self.epoch

        if create_summary_csv:
            self.summary_statistics_filepath = save_statistics(
                self.logs_filepath, list(epoch_summary_losses.keys()), create=True
            )
            self.create_summary_csv = False

        self.logging.info(
            "epoch {} -> epoch time = {} || total time = {}".format(
                self.epoch,
                time.time() - self.start_time,
                time.time() - self.total_start
            )
        )
        self.start_time = time.time()
        self.logging.info(
            "epoch {} -> {}".format(epoch_summary_losses["epoch"], epoch_summary_string)
        )

        self.summary_statistics_filepath = save_statistics(
            self.logs_filepath, list(epoch_summary_losses.values())
        )
        return state

    def evaluated_test_set_using_the_best_models(self, top_n_models):

        if self.args.retrain: # evaluation process
            per_epoch_statistics = self.state["per_epoch_statistics"]
            val_acc = np.copy(per_epoch_statistics["val_accuracy_mean"])
            val_idx = np.array([i for i in range(len(val_acc))])
            sorted_idx = np.argsort(val_acc, axis=0).astype(dtype=np.int32)[::-1]
            sorted_idx = sorted_idx[:top_n_models]

            sorted_val_acc = val_acc[sorted_idx]
            val_idx = val_idx[sorted_idx]
            top_n_models = len(sorted_idx) # maybe top_n_models is smaller
            top_n_idx = val_idx[:top_n_models]

            self.logging.info(sorted_idx)
            self.logging.info(sorted_val_acc)
        else: # search process
            # sorted_idx = [s for s in sorted_idx if s > self.args.warmup_dec_epoch + 7 * self.args.decision_freq][
            #     :top_n_models
            # ]
            sorted_idx = range(self.args.total_epochs - top_n_models, self.args.total_epochs)
            top_n_idx = sorted_idx


        per_model_per_batch_preds = [[] for i in range(top_n_models)]
        per_model_per_batch_targets = [[] for i in range(top_n_models)]

        test_losses = dict()
        max_acc = 0
        for idx, model_idx in enumerate(top_n_idx):
            self.state = self.model.load_model(
                model_save_dir=self.saved_models_filepath,
                model_name="train_model",
                model_idx=model_idx,
            )
            with tqdm.tqdm(
                total=int(self.args.num_evaluation_tasks / self.args.batch_size)
            ) as pbar_test:
                for sample_idx, test_sample in enumerate(
                    self.data.get_test_batches(
                        total_batches=int(
                            self.args.num_evaluation_tasks / self.args.batch_size
                        ),
                        augment_images=False,
                    )
                ):
                    if sample_idx == 0:
                        self.check_operations(test_sample)
                    per_model_per_batch_targets[idx].extend(np.array(test_sample[3]))
                    per_model_per_batch_preds = self.test_evaluation_iteration(
                        val_sample=test_sample,
                        sample_idx=sample_idx,
                        model_idx=idx,
                        per_model_per_batch_preds=per_model_per_batch_preds,
                        pbar_test=pbar_test,
                    )

        per_batch_preds = np.mean(per_model_per_batch_preds, axis=0)
        per_batch_max = np.argmax(per_batch_preds, axis=2)
        per_batch_targets = np.array(per_model_per_batch_targets[0]).reshape(
            per_batch_max.shape
        )
        accuracy = np.mean(np.equal(per_batch_targets, per_batch_max))
        accuracy_std = np.std(np.equal(per_batch_targets, per_batch_max))

        test_losses = {
            "test_accuracy_mean": accuracy,
            "test_accuracy_std": accuracy_std,
        }

        _ = save_statistics(
            self.logs_filepath,
            list(test_losses.keys()),
            create=True,
            filename="test_summary.csv",
        )

        summary_statistics_filepath = save_statistics(
            self.logs_filepath,
            list(test_losses.values()),
            create=False,
            filename="test_summary.csv",
        )
        self.logging.info(test_losses)
        self.logging.info("saved test performance at %s", summary_statistics_filepath)

    def check_operations(self, train_sample):

        x_support_set, x_target_set, y_support_set, y_target_set, seed = train_sample
        data_batch = (x_support_set, x_target_set, y_support_set, y_target_set)

        self.model.run_check_iter(data_batch, self.logging)

    def validation(self):
        """
        run full experiment
        """
        total_losses = dict()
        val_losses = dict()
        with tqdm.tqdm(
            total=int(self.args.num_evaluation_tasks / self.args.batch_size)
        ) as pbar_val:

            for val_iter, val_sample in enumerate(
                # self.data.get_val_batches(
                self.data.get_test_batches(
                    total_batches=int(
                        self.args.num_evaluation_tasks
                        / self.args.batch_size
                    ),
                    augment_images=False,
                )
            ):
                val_losses, total_losses = self.validation_iteration(
                    val_sample=val_sample,
                    total_losses=total_losses,
                    phase="val",
                    pbar_val=pbar_val,
                )
        if val_losses["val_accuracy_mean"] > self.state["best_val_acc"]:
            self.state["best_val_acc"] = val_losses["val_accuracy_mean"]
            self.state["best_val_iter"] = self.state["current_iter"]
            self.state["best_epoch"] = self.epoch

        self.logging.info("Curr epoch {}\t|| {}".format(self.epoch, val_losses["val_accuracy_mean"]))
        self.logging.info("Best epoch {}\t|| {}".format(self.state["best_epoch"], self.state["best_val_acc"]))

        return val_losses


    def run_experiment(self):
        """
        Runs a full training experiment with evaluations of the model on the val set at every epoch. Furthermore,
        will return the test set evaluation results on the best performing validation model.
        """
        self.total_start = time.time()
        if not self.args.retrain: # search_process
            self.model.classifier.check_weights(self.logging)
        while (self.state["current_iter"] < (self.args.total_epochs * self.args.total_iter_per_epoch)) \
            and (self.args.evaluate_on_test_set_only == False):
            for train_sample_idx, train_sample in enumerate(
                self.data.get_train_batches(
                    total_batches=int(self.args.total_iter_per_epoch * self.args.total_epochs) - self.state["current_iter"],
                    augment_images=self.augment_flag,
                )
            ):
                # print(self.state['current_iter'], (self.args.total_epochs * self.args.total_iter_per_epoch))
                if (self.state["current_iter"] % self.args.total_iter_per_epoch == 0):
                    pbar_train = tqdm.tqdm(total=self.args.total_iter_per_epoch)

                train_losses, total_losses, = self.train_iteration(
                    train_sample=train_sample,
                    total_losses=self.total_losses,
                    epoch_idx=self.epoch,
                    sample_idx=train_sample_idx,
                    pbar_train = pbar_train
                )

                self.state["current_iter"] += 1  # update current iter

                if (self.state["current_iter"] % self.args.total_iter_per_epoch == 0):

                    pbar_train.close()
                    # at the end iter of this epoch
                    if not self.args.retrain: # search process
                        self.model.classifier.edge_decision(
                            self.epoch,
                            self.logging,
                        )
                        # self.check_operations(train_sample)

                        self.state['alphas'] = self.model.classifier._alphas()
                        # if self.model.classifier.isSearchFinished():
                        #     self.logging.info("finish searching process at epoch {}".format(self.epoch))
                    val_losses = dict()
                    val_losses['val_accuracy_mean'] = 0.0
                    val_losses['val_accuracy_std'] = 0.0
                    val_losses['val_loss_mean'] = 0.0
                    val_losses['val_loss_std'] = 0.0

                    if self.args.retrain or self.epoch == 0 or torch.sum(self.model.classifier.reduce_candidate_flags.int()) == 0: # at the beginning of search or after search
                        val_losses = self.validation()

                        # add scalars
                        self.writer.add_scalars('accuracy', {
                            "train_accuracy": train_losses['train_accuracy_mean'],
                            "test_accuracy": val_losses['val_accuracy_mean']
                        }, self.epoch)

                        self.writer.add_scalars('loss', {
                            "train_loss": train_losses['train_loss_mean'],
                            "test_loss": val_losses['val_loss_mean']
                        }, self.epoch)

                    self.state = self.merge_two_dicts(
                        first_dict=self.merge_two_dicts(
                            first_dict=self.state, second_dict=train_losses
                        ),
                        second_dict=val_losses,
                    )

                    self.state = self.pack_and_save_metrics(
                        create_summary_csv=self.create_summary_csv,
                        train_losses=train_losses,
                        val_losses=val_losses,
                        state=self.state,
                    )
                    self.total_losses = dict()
                    save_to_json(
                        filename=os.path.join(self.logs_filepath, "summary_statistics.json"),
                        dict_to_store=self.state["per_epoch_statistics"],
                    )

                    self.save_models(
                        model=self.model, epoch=self.epoch, state=self.state
                    )

                    self.epochs_done_in_this_run += 1
                    self.epoch += 1

                    if self.epochs_done_in_this_run >= self.total_epochs_before_pause:
                        self.logging.info(
                            "train_seed {}, val_seed: {}, at pause time".format(
                                self.data.dataset.seed["train"],
                                self.data.dataset.seed["val"],
                            )
                        )
                        sys.exit()

        if self.args.retrain or self.args.evaluate_on_test_set_only:
            self.logging.info("Best test acc = {}, at epoch {}".format(self.state["best_val_acc"], self.state["best_epoch"]))
            if self.args.use_ensemble and self.args.retrain:
                self.logging.info("use ensemble method proposed by MAML++. More checkpoint are needed. Recommand to rerun the traning!")
                self.evaluated_test_set_using_the_best_models(top_n_models=self.args.top_n_model_for_test)
