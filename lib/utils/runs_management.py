import os
import yaml
import numpy as np
import pandas as pd
import tensorflow as tf
from colorama import Fore, Style


class RunManager:
    """Utility class to manage saved runs.
    The project directory is expected to be organized as follows:
    project_root/
    ├── src/
    └── runs/
        ├── default.yaml
        ├── DNS-GT/
        |   ├── default.yaml
        │   ├── <run_name_1>/
        │   │   ├── weights.h5
        │   │   ├── embeddings.npy
        │   │   ├── conf.yaml
        |   |   └── predictions.csv
        │   └── <run_name_2>/
        │       ├── weights.h5
        │       ├── embeddings.npy
        │       ├── conf.yaml
        |       └── predictions.csv
        ├── W2V-CBOW/
        |   ├── default.yaml
        │   ├── <run_name_1>/
        │   │   ├── weights.h5
        │   │   ├── embeddings.npy
        │   │   ├── conf.yaml
        │   └── <run_name_2>/
        │       ├── weights.h5
        │       ├── embeddings.npy
        │       ├── conf.yaml
        |       └── predictions.csv
        └── W2V-SkipGram/
            ├── default.yaml
            └── <run_name>/
                ├── weights.h5
                ├── embeddings.npy
                ├── conf.yaml
                └── predictions.csv
    """

    # Static constants
    _PROJECT_ROOT = os.path.join(os.path.dirname(__file__), "..", "..")  # ".."
    _WEIGHTS_FILE_NAME = "weights.h5"
    _WEIGHTS_FINETUNING_FILE_NAME = "weights.finetuning.h5"
    _EMBEDDINGS_FILE_NAME = "embeddings.npy"
    _CONF_FILE_NAME = "conf.yaml"
    _PREDICTIONS_FOLDER_NAME = "predictions"
    _DEFAULT_CONF_FILE_NAME = "default.yaml"

    # These fields will not be saved
    _CONF_WHITELIST = {
        "demo",
        "eager",
        "finetune",
        "from_pretrained",
        "start_from",
        "skip_predictions",
        "gpu",
        "test_seq",
    }

    def __init__(
        self,
        model_object: tf.keras.Model,
        model_name: str,
        run_name: str = None,
        start_from: str = None,
        last: bool = False,  # deprecated
        verbose: bool = False,
    ):
        # Initialize parameters
        self.model_object = model_object
        self.model_name = model_name
        self.run_name = run_name
        self.start_from = start_from
        self.last = last
        self.verbose = verbose

        # Create path if it doesn't exist
        if not os.path.exists(
            os.path.join(self._PROJECT_ROOT, "runs", self.model_name, self.run_name)
        ):
            os.makedirs(
                os.path.join(self._PROJECT_ROOT, "runs", self.model_name, self.run_name)
            )

        # Store correct paths for weights, embeddings and conf
        (
            self.run_path,
            self.weights_path,
            self.embeddings_path,
            self.conf_path,
            self.predictions_path,
        ) = self._get_paths(model_name, run_name)

    def exist_weights(self):
        """Returns True if weights exist for this run.

        Returns:
            exists (bool): True if weights exist, False otherwise.
        """
        return os.path.exists(self.weights_path)

    def _assert_and_get_model(self, model_object=None):
        if model_object is None and self.model_object is None:
            raise ValueError(
                f"{Fore.RED}[ERROR] At least one between model_object and self.model_object must not be None.{Fore.RESET}"
            )
        if model_object is None:
            return self.model_object
        else:
            return model_object

    def save_weights(self, model_object: tf.keras.Model = None) -> None:
        model_object = self._assert_and_get_model(model_object)
        try:
            model_object.save_weights(self.weights_path)
        except Exception as e:
            print(
                f"{Fore.RED}The following error occurred while saving "
                + f"model weights to {self.weights_path}: \n{e}{Fore.RESET}"
            )
            raise Exception(e)
        else:
            if self.verbose:
                print(
                    f"{Fore.GREEN}[INFO] Model weights successfully saved to {self.weights_path}.{Fore.RESET}"
                )

    def save_embeddings(self, model_object: tf.keras.Model = None) -> None:
        model_object = self._assert_and_get_model(model_object)
        try:
            # TODO may break if model class uses a different variable name; use a get_embeddings() method instead
            domain_embeddings = model_object.domain_embeddings.embeddings.numpy()
            np.save(self.embeddings_path, domain_embeddings)
        except Exception as e:
            print(
                f"{Fore.RED}The following error occurred while saving "
                + f"model embeddings to {self.weights_path}: \n{e}{Fore.RESET}"
            )
            raise Exception(e)
        else:
            if self.verbose:
                print(
                    f"{Fore.GREEN}[INFO] Model embeddings successfully saved to {self.embeddings_path}.{Fore.RESET}"
                )

    def save_conf(self, model_object: tf.keras.Model = None) -> None:
        model_object = self._assert_and_get_model(model_object)

        conf_to_save = dict(model_object.conf)

        # Remove whitelisted fields from dict (they won't be saved)
        for field in self._CONF_WHITELIST:
            if field in conf_to_save:
                conf_to_save.pop(field)

        try:
            with open(self.conf_path, "w") as f:
                yaml.safe_dump(
                    conf_to_save, f, default_flow_style=False, sort_keys=False
                )
        except Exception as e:
            print(
                f"{Fore.RED}The following error occurred while saving "
                + f"the conf file to {self.weights_path}: \n{e}{Fore.RESET}"
            )
            raise Exception(e)
        else:
            if self.verbose:
                print(
                    f"{Fore.GREEN}[INFO] Configuration file successfully saved to {self.conf_path}.{Fore.RESET}"
                )

    def save_predictions(self, predictions, test_partition, test_fold):
        # Create predictions folder if it doesn't exist
        if not os.path.exists(self.predictions_path):
            os.makedirs(self.predictions_path)

        save_path = os.path.join(
            self.predictions_path,
            f"predictions-partition{test_partition}-fold{test_fold}.csv",
        )
        try:
            predictions.to_csv(save_path)
        except Exception as e:
            print(
                f"{Fore.RED}The following error occurred while saving predictions to {save_path}: \n{e}{Fore.RESET}"
            )
            raise Exception(e)
        else:
            if self.verbose:
                print(
                    f"{Fore.GREEN}[INFO] Predictions successfully saved to {save_path}.{Fore.RESET}"
                )

    def load_weights(
        self, model_object: tf.keras.Model = None, override_run_name: str = None
    ):
        if model_object is None:
            model_object = self.model_object

        # if no weights have been saved yet, load from self.start_from
        weights_path = None
        if not self.exist_weights():
            _, start_from_weights_path, _, _, _ = self._get_paths(
                self.model_name, self.start_from
            )
            weights_path = start_from_weights_path
        else:
            weights_path = self.weights_path

        # try to load weights into the model
        try:
            model_object.load_weights(weights_path)
        except Exception as e:
            print(
                f"{Fore.YELLOW}[WARN] Could not load model weights from {weights_path}: \n{e}{Fore.RESET}"
            )
        else:
            if self.verbose:
                print(
                    f"{Fore.GREEN}[INFO] Model weights successfully loaded from {weights_path}.{Fore.RESET}"
                )
        return model_object

    def load_embeddings(self):
        # try to load embeddings as numpy array
        embeddings = None

        try:
            embeddings = np.load(self.embeddings_path)
        except Exception as e:
            print(
                f"{Fore.YELLOW}[WARN] Could not load model embeddings from {self.embeddings_path}: \n{e}{Fore.RESET}"
            )
            raise Exception(e)
        else:
            if self.verbose:
                print(
                    f"{Fore.GREEN}[INFO] Model embeddings successfully loaded from {self.embeddings_path}.{Fore.RESET}"
                )
        return embeddings

    def load_conf(self):
        conf = {}

        # Load default root conf as a base
        default_root_conf_path = os.path.join(
            self._PROJECT_ROOT, "runs", self._DEFAULT_CONF_FILE_NAME
        )
        try:
            with open(
                default_root_conf_path,
                "r",
            ) as f:
                conf = conf | yaml.safe_load(f)
        except:
            print(
                f"{Fore.YELLOW}[WARN] Could not load default conf file for {self.run_name} in {default_root_conf_path}.\n"
                + f"Make sure that there exists a default.yaml file in the model runs folder.{Fore.RESET}"
            )

        # Load default model conf and override the root one
        default_model_conf_path = os.path.join(
            self._PROJECT_ROOT,
            "runs",
            self.model_name,
            self._DEFAULT_CONF_FILE_NAME,
        )
        try:
            with open(
                default_model_conf_path,
                "r",
            ) as f:
                conf = conf | yaml.safe_load(f)
        except:
            print(
                f"{Fore.YELLOW}[WARN] Could not load default conf file for {self.run_name} in {default_model_conf_path}.\n"
                + f"Make sure that there exists a default.yaml file in the model runs folder.{Fore.RESET}"
            )

        # if conf has not been saved yet, load it from start_from
        conf_path = None
        if not os.path.exists(self.conf_path):
            _, _, _, start_from_conf_path, _ = self._get_paths(
                self.model_name, self.start_from
            )
            conf_path = start_from_conf_path
        else:
            conf_path = self.conf_path

        # Load run-specific conf and override the default model one
        try:
            with open(conf_path, "r") as f:
                conf = conf | yaml.safe_load(f)
            if self.verbose:
                print(
                    f"{Fore.GREEN}[INFO] Conf file successfully loaded from {conf_path}.{Fore.RESET}"
                )
        except:
            print(
                f"{Fore.YELLOW}[INFO] Conf file not found for {self.run_name} in {conf_path}. Starting from scratch.{Fore.RESET}"
            )

        return conf

    def load_predictions(self, test_partition, test_fold):
        # try to load predictions as a pandas DataFrame
        predictions = None

        predictions_path = os.path.join(
            self.predictions_path,
            f"predictions-partition{test_partition}-fold{test_fold}.csv",
        )
        try:
            predictions = pd.read_csv(predictions_path)
        except Exception as e:
            print(
                f"{Fore.YELLOW}[WARN] Could not load predictions from {predictions_path}: \n{e}{Fore.RESET}"
            )
            raise Exception(e)
        else:
            if self.verbose:
                print(
                    f"{Fore.GREEN}[INFO] Predictions successfully loaded from {predictions_path}.{Fore.RESET}"
                )
        return predictions

    def _get_paths(
        self,
        model_name: str = None,
        run_name: str = None,
        finetuning: bool = False,
    ):
        # check for ValueErrors
        if self.model_name is None and model_name is None:
            raise ValueError(f"{Fore.RED}model_name must be specified.{Fore.RESET}")
        if self.run_name is None and run_name is None and not self.last:
            raise ValueError(
                f"{Fore.RED}run_name must be specified if load is False."
                + f"{Fore.RESET}"
            )

        # if model_name is None, use the one provided in the constructor
        if model_name is None:
            model_name = self.model_name

        # if model_name is None, use the one provided in the constructor
        if run_name is None:
            run_name = self.run_name

        # set weights, embeddings, conf and predictions paths
        run_path = os.path.join(self._PROJECT_ROOT, "runs", model_name, run_name)
        weights_path = os.path.join(
            run_path,
            self._WEIGHTS_FILE_NAME
            if not finetuning
            else self._WEIGHTS_FINETUNING_FILE_NAME,
        )
        embeddings_path = os.path.join(run_path, self._EMBEDDINGS_FILE_NAME)
        conf_path = os.path.join(run_path, self._CONF_FILE_NAME)
        predictions_path = os.path.join(run_path, self._PREDICTIONS_FOLDER_NAME)

        return (
            run_path,
            weights_path,
            embeddings_path,
            conf_path,
            predictions_path,
        )

    @staticmethod
    def load_root_conf():
        conf = {}

        # Load default root conf as a base
        default_root_conf_path = os.path.join(
            __class__._PROJECT_ROOT, "runs", __class__._DEFAULT_CONF_FILE_NAME
        )
        try:
            with open(
                default_root_conf_path,
                "r",
            ) as f:
                conf = conf | yaml.safe_load(f)
        except:
            print(
                f"{Fore.RED}[ERROR] Could not load root conf file from {default_root_conf_path}.\n"
                + f"Make sure that there exists a default.yaml file in the runs/ folder.{Fore.RESET}"
            )

        return conf
