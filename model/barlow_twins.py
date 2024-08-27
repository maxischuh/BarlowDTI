import torch
torch.manual_seed(42)
torch.backends.cudnn.deterministic = True
from torch import nn
import numpy as np
from typing import *
from datetime import datetime
import os
import pickle
import inspect
from tqdm.auto import trange

from base_model import BaseModel


class BarlowTwins(BaseModel):
    def __init__(
            self,
            n_bits: int = 1024,
            aa_emb_size: int = 1024,
            enc_n_neurons: int = 512,
            enc_n_layers: int = 2,
            proj_n_neurons: int = 2048,
            proj_n_layers: int = 2,
            embedding_dim: int = 512,
            act_function: str = "relu",
            loss_weight: float = 0.005,
            batch_size: int = 512,
            optimizer: str = "adamw",
            momentum: float = 0.9,
            learning_rate: float = 0.0001,
            betas: tuple = (0.9, 0.999),
            weight_decay: float = 1e-3,
            step_size: int = 10,
            gamma: float = 0.1,
            verbose: bool = True,
    ):
        super().__init__()
        
        self.enc_aa = None
        self.enc_mol = None
        self.proj = None
        
        self.scheduler = None
        self.optimizer = None

        # store input in dict
        self.param_dict = {
            "act_function": self.activation_dict[
                act_function
            ],  # which activation function to use among dict options
            "loss_weight": loss_weight,  # off-diagonal cross correlation loss weight
            "batch_size": batch_size,  # samples per gradient step
            "learning_rate": learning_rate,  # update step magnitude when training
            "betas": betas,  # momentum hyperparameter for adam-like optimizers
            "step_size": step_size,  # decay period for the learning rate
            "gamma": gamma,  # decay coefficient for the learning rate
            "optimizer": self.optimizer_dict[
                optimizer
            ],  # which optimizer to use among dict options
            "momentum": momentum,  # momentum hyperparameter for SGD
            "enc_n_neurons": enc_n_neurons,  # neurons to use for the mlp encoder
            "enc_n_layers": enc_n_layers,  # number of hidden layers in the mlp encoder
            "proj_n_neurons": proj_n_neurons,  # neurons to use for the mlp projector
            "proj_n_layers": proj_n_layers,  # number of hidden layers in the mlp projector
            "embedding_dim": embedding_dim,  # latent space dim for downstream tasks
            "weight_decay": weight_decay,  # l2 regularization for linear layers
            "verbose": verbose,  # whether to print feedback
            "radius": "Not defined yet",  # fingerprint radius
            "n_bits": n_bits,  # fingerprint bit size
            "aa_emb_size": aa_emb_size,  # aa embedding size
        }

        # create history dictionary
        self.history = {
            "train_loss": [],
            "on_diag_loss": [],
            "off_diag_loss": [],
            "validation_loss": [],
        }

        # run NN architecture construction method
        self.construct_model()

        # run scheduler construction method
        self.construct_scheduler()

        # print if necessary
        if self.param_dict["verbose"] is True:
            self.print_config()

    @staticmethod
    def __validate_inputs(locals_dict) -> None:
        # get signature types from __init__
        init_signature = inspect.signature(BarlowTwins.__init__)

        # loop over all chosen arguments
        for param_name, param_value in locals_dict.items():
            # skip self
            if param_name != "self":
                # check that parameter exists
                if param_name in init_signature.parameters:
                    # check that param is correct type
                    expected_type = init_signature.parameters[param_name].annotation
                    assert isinstance(
                        param_value, expected_type
                    ), f"[BT]: Type mismatch for parameter '{param_name}'"
                else:
                    raise ValueError(f"[BT]: Unexpected parameter '{param_name}'")

    def construct_mlp(self, input_units, layer_units, n_layers, output_units) -> nn.Sequential:

        # make empty list to fill
        mlp_list = []

        # make lists defining layer sizes (input + n_neurons*n_layers + embedding_dim)
        units = [input_units] + [layer_units] * n_layers

        # add layer stack (linear -> batchnorm -> dropout -> activation)
        for i in range(len(units) - 1):
            mlp_list.append(nn.Linear(units[i], units[i + 1]))
            mlp_list.append(nn.BatchNorm1d(units[i + 1]))
            mlp_list.append(self.param_dict["act_function"]())

        # add final linear layer
        mlp_list.append(nn.Linear(units[-1], output_units))

        return nn.Sequential(*mlp_list)

    def construct_model(self) -> None:
        # create fingerprint transformer
        self.enc_mol = self.construct_mlp(
            self.param_dict["n_bits"],
            self.param_dict["enc_n_neurons"],
            self.param_dict["enc_n_layers"],
            self.param_dict["embedding_dim"],
        )

        # create aa transformer
        self.enc_aa = self.construct_mlp(
            self.param_dict["aa_emb_size"],
            self.param_dict["enc_n_neurons"],
            self.param_dict["enc_n_layers"],
            self.param_dict["embedding_dim"],
        )

        # create mlp projector
        self.proj = self.construct_mlp(
            self.param_dict["embedding_dim"],
            self.param_dict["proj_n_neurons"],
            self.param_dict["proj_n_layers"],
            self.param_dict["proj_n_neurons"],
        )

        # print if necessary
        if self.param_dict["verbose"] is True:
            print("[BT]: Model constructed successfully")

    def construct_scheduler(self):
        # make optimizer
        self.optimizer = self.param_dict["optimizer"](
            list(self.enc_mol.parameters())
            + list(self.enc_aa.parameters())
            + list(self.proj.parameters()),
            lr=self.param_dict["learning_rate"],
            betas=self.param_dict["betas"],
            # momentum=self.param_dict["momentum"],
            weight_decay=self.param_dict["weight_decay"],
        )

        # wrap optimizer in scheduler
        """
         self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer,
            T_max=self.param_dict["step_size"], # T_0
            # eta_min=1e-7,
            verbose=True
        ) 

        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer,
            patience=self.param_dict["step_size"],
            verbose=True
        )
        """
        self.scheduler = torch.optim.lr_scheduler.StepLR(
            self.optimizer,
            step_size=self.param_dict["step_size"],
            gamma=self.param_dict["gamma"],
        )

        # print if necessary
        if self.param_dict["verbose"] is True:
            print("[BT]: Optimizer constructed successfully")

    def switch_mode(self, is_training: bool):
        if is_training:
            self.enc_mol.train()
            self.enc_aa.train()
            self.proj.train()
        else:
            self.enc_mol.eval()
            self.enc_aa.eval()
            self.proj.eval()

    @staticmethod
    def normalize_projection(tensor: torch.tensor) -> torch.tensor:
        means = torch.mean(tensor, axis=0)
        std = torch.std(tensor, axis=0)
        centered = torch.add(tensor, -means)
        scaled = torch.div(centered, std)

        return scaled

    def compute_loss(
        self,
        mol_embedding: torch.tensor,
        aa_embedding: torch.tensor,
    ) -> torch.tensor:

        # empirical cross-correlation matrix
        mol_embedding = self.normalize_projection(mol_embedding).T
        aa_embedding = self.normalize_projection(aa_embedding)
        c = mol_embedding @ aa_embedding

        # normalize by number of samples
        c.div_(self.param_dict["batch_size"])

        # compute elements on diagonal
        on_diag = torch.diagonal(c).add_(-1).pow_(2).sum()

        # compute elements off diagonal
        n, m = c.shape
        off_diag = c.flatten()[:-1].view(n - 1, n + 1)[:, 1:].flatten()
        off_diag = off_diag.pow_(2).sum() * self.param_dict["loss_weight"]

        return on_diag, off_diag

    def forward(
        self, mol_data: torch.tensor, aa_data: torch.tensor, is_training: bool = True
    ) -> torch.tensor:

        # switch according to input
        self.switch_mode(is_training)

        # get embeddings
        mol_embeddings = self.enc_mol(mol_data)
        aa_embeddings = self.enc_aa(aa_data)

        # get projections
        mol_proj = self.proj(mol_embeddings)
        aa_proj = self.proj(aa_embeddings)

        # compute loss
        on_diag, off_diag = self.compute_loss(mol_proj, aa_proj)

        return on_diag, off_diag

    def train(
        self,
        train_data: torch.utils.data.DataLoader,
        val_data: torch.utils.data.DataLoader = None,
        num_epochs: int = 20,
        patience: int = None,
    ):
        if self.param_dict["verbose"] is True:
            print("[BT]: Training started")

        if patience is None:
            patience = 2 * self.param_dict["step_size"]

        pbar = trange(num_epochs, desc="[BT]: Epochs", leave=False, colour="blue")

        for epoch in pbar:
            # initialize loss containers
            train_loss = 0.0
            on_diag_loss = 0.0
            off_diag_loss = 0.0
            val_loss = 0.0

            # loop over training set
            for _, (mol_data, aa_data) in enumerate(train_data):
                # reset grad
                self.optimizer.zero_grad()

                # compute train loss for batch
                on_diag, off_diag = self.forward(mol_data, aa_data, is_training=True)
                t_loss = on_diag + off_diag

                # backpropagation and optimization
                t_loss.backward()
                """
                nn.utils.clip_grad_norm_(
                    list(self.enc_mol.parameters()) +
                    list(self.enc_aa.parameters()) +
                    list(self.proj.parameters()),
                    1
                )
                """
                self.optimizer.step()

                # add i-th loss to training container
                train_loss += t_loss.item()
                on_diag_loss += on_diag.item()
                off_diag_loss += off_diag.item()

            # add mean epoch loss for train data to history dictionary
            self.history["train_loss"].append(train_loss / len(train_data))
            self.history["on_diag_loss"].append(on_diag_loss / len(train_data))
            self.history["off_diag_loss"].append(off_diag_loss / len(train_data))

            # define msg to be printed
            msg = (
                f"[BT]: Epoch [{epoch + 1}/{num_epochs}], "
                f"Train loss: {train_loss / len(train_data):.3f}, "
                f"On diagonal: {on_diag_loss / len(train_data):.3f}, "
                f"Off diagonal: {off_diag_loss / len(train_data):.3f} "
            )

            # loop over validation set (if present)
            if val_data is not None:

                for _, (mol_data, aa_data) in enumerate(val_data):
                    # compute val loss for batch
                    on_diag_v_loss, off_diag_v_loss = self.forward(
                        mol_data, aa_data, is_training=False
                    )

                    # add i-th loss to val container
                    v_loss = on_diag_v_loss + off_diag_v_loss
                    val_loss += v_loss.item()

                # add mean epoc loss for val data to history dictionary
                self.history["validation_loss"].append(val_loss / len(val_data))

                # add val loss to msg
                msg += f", Val loss: {val_loss / len(val_data):.3f}"

                # early stopping
                if self.early_stopping(patience=patience):
                    break

                pbar.set_postfix(
                    {
                        "train loss": train_loss / len(train_data),
                        "val loss": val_loss / len(val_data),
                    }
                )

            else:
                pbar.set_postfix({"train loss": train_loss / len(train_data)})

            # update scheduler
            self.scheduler.step()  # val_loss / len(val_data)

            if self.param_dict["verbose"] is True:
                print(msg)

        if self.param_dict["verbose"] is True:
            print("[BT]: Training finished")

    def encode(
        self, vector: np.ndarray, mode: str = "embedding", normalize: bool = True, encoder: str = "mol"
    ) -> np.ndarray:
        """
        Encodes a given vector using the Barlow Twins model.

        Args:
        - vector (np.ndarray): the input vector to encode
        - mode (str): the mode to use for encoding, either "embedding" or "projection"
        - normalize (bool): whether to L2 normalize the output vector

        Returns:
        - np.ndarray: the encoded vector
        """

        # set mol encoder to eval mode
        self.switch_mode(is_training=False)

        # convert from numpy to tensor
        if type(vector) is not torch.Tensor:
            vector = torch.from_numpy(vector)

        # if oly one molecule pair is passed, add a batch dimension
        if len(vector.shape) == 1:
            vector = vector.unsqueeze(0)

        # get representation
        if encoder == "mol":
            embedding = self.enc_mol(vector)
            if mode == "projection":
                embedding = self.proj(embedding)
        elif encoder == "aa":
            embedding = self.enc_aa(vector)
            if mode == "projection":
                embedding = self.proj(embedding)
        else:
            raise ValueError("[BT]: Encoder not recognized")

        # L2 normalize (optional)
        if normalize:
            embedding = torch.nn.functional.normalize(embedding)

        # convert back to numpy
        return embedding.cpu().detach().numpy()

    def zero_shot(
        self, mol_vector: np.ndarray, aa_vector: np.ndarray, l2_norm: bool = True, device: str = "cpu"
    ) -> np.ndarray:

        # disable training
        self.switch_mode(is_training=False)

        # cast aa vectors (pos and neg) to correct size, force single precision
        # to both
        mol_vector = np.array(mol_vector, dtype=np.float32)
        aa_vector = np.array(aa_vector, dtype=np.float32)

        # convert to tensors
        mol_vector = torch.from_numpy(mol_vector).to(device)
        aa_vector = torch.from_numpy(aa_vector).to(device)

        # get embeddings
        mol_embedding = self.encode(mol_vector, normalize=l2_norm, encoder="mol")
        aa_embedding = self.encode(aa_vector, normalize=l2_norm, encoder="aa")

        # concat mol and aa embeddings
        concat = np.concatenate((mol_embedding, aa_embedding), axis=1)
        return concat
    
    def zero_shot_explain(
        self, mol_vector, aa_vector, l2_norm: bool = True, device: str = "cpu"
    ):
        self.switch_mode(is_training=False)

        mol_embedding = self.encode(mol_vector, normalize=l2_norm, encoder="mol")
        aa_embedding = self.encode(aa_vector, normalize=l2_norm, encoder="aa")

        return torch.cat((mol_embedding, aa_embedding), dim=1)

    def consume_preprocessor(self, preprocessor) -> None:
        # save attributes related to fingerprint generation from
        # preprocessor object
        self.param_dict["radius"] = preprocessor.radius
        self.param_dict["n_bits"] = preprocessor.n_bits

    def save_model(self, path: str) -> None:
        # get current date and time for the filename
        now = datetime.now()
        formatted_date = now.strftime("%d%m%Y")
        formatted_time = now.strftime("%H%M")
        folder_name = f"{formatted_date}_{formatted_time}"

        # make full path string and folder
        folder_path = path + "/" + folder_name
        os.makedirs(folder_path)

        # make paths for weights, config and history
        weight_path = folder_path + "/weights.pt"
        param_path = folder_path + "/params.pkl"
        history_path = folder_path + "/history.json"

        # save each Sequential state dict in one object to the path
        torch.save(
            {
                "enc_mol": self.enc_mol.state_dict(),
                "enc_aa": self.enc_aa.state_dict(),
                "proj": self.proj.state_dict(),
            },
            weight_path,
        )

        # dump params in pkl
        with open(param_path, "wb") as file:
            pickle.dump(self.param_dict, file)

        # dump history in json
        with open(history_path, "wb") as file:
            pickle.dump(self.history, file)

        # print if verbose is True
        if self.param_dict["verbose"] is True:
            print(f"[BT]: Model saved at {folder_path}")

    def load_model(self, path: str) -> None:
        # make weights, config and history paths
        weights_path = path + "/weights.pt"
        param_path = path + "/params.pkl"
        history_path = path + "/history.json"

        # load weights, history and params
        checkpoint = torch.load(weights_path, map_location=self.device)
        with open(param_path, "rb") as file:
            param_dict = pickle.load(file)
        with open(history_path, "rb") as file:
            history = pickle.load(file)

        # construct model again, overriding old verbose key with new instance
        verbose = self.param_dict["verbose"]
        self.param_dict = param_dict
        self.param_dict["verbose"] = verbose
        self.history = history
        self.construct_model()

        # set weights in Sequential models
        self.enc_mol.load_state_dict(checkpoint["enc_mol"])
        self.enc_aa.load_state_dict(checkpoint["enc_aa"])
        self.proj.load_state_dict(checkpoint["proj"])

        # recreate scheduler and optimizer in order to add new weights
        # to graph
        self.construct_scheduler()

        # print if verbose is True
        if self.param_dict["verbose"] is True:
            print(f"[BT]: Model loaded from {path}")
            print("[BT]: Loaded parameters:")
            print(self.param_dict)

    def move_to_device(self, device) -> None:
        # move each Sequential model to device
        self.enc_mol.to(device)
        self.enc_aa.to(device)
        self.proj.to(device)
        self.device = device
