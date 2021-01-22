from data import MetaLearningSystemDataLoader
# from data_meta import FewShotLearningDataset
# from experiment_builder import ExperimentBuilder
from experiment_builder_pruning import PruningExperimentBuilder
# from few_shot_learning_system import MAMLFewShotClassifier
from few_shot_learning_system_pruning import MAMLFewShotClassifier
from utils.parser_utils import get_args
from utils.dataset_tools import maybe_unzip_dataset

# Combines the arguments, model, data and experiment builders to run an experiment
args, device = get_args()
model = MAMLFewShotClassifier(args=args, device=device,
                              im_shape=(2, args.image_channels,
                                        args.image_height, args.image_width))

maybe_unzip_dataset(args=args)
data = MetaLearningSystemDataLoader

# data = FewShotLearningDataset

# maml_system = ExperimentBuilder(model=model, data=data, args=args, device=device)
maml_system = PruningExperimentBuilder(model=model, data=data, args=args, device=device)
maml_system.run_experiment()
