import csv
import json
from ast import literal_eval
from contextlib import contextmanager
from os import PathLike
from pathlib import Path
from typing import Annotated

import joblib
import numpy as np
import sklearn
import typer
from CDPL.Chem import (
    Atom,  # pyright:ignore[reportAttributeAccessIssue]
    BasicMolecule,  # pyright:ignore[reportAttributeAccessIssue]
    FileSDFMoleculeReader,  # pyright:ignore[reportAttributeAccessIssue]
    MolecularGraph,  # pyright:ignore[reportAttributeAccessIssue]
    generateSMILES,  # pyright:ignore[reportAttributeAccessIssue]
    getStructureData,  # pyright:ignore[reportAttributeAccessIssue]
)
from rich.console import Console
from rich.json import JSON
from rich.progress import Progress, SpinnerColumn, TextColumn, TimeElapsedColumn
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    average_precision_score,
    f1_score,
    make_scorer,
    matthews_corrcoef,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import (
    GridSearchCV,
    GroupKFold,
    TunedThresholdClassifierCV,
)
from sklearn.pipeline import make_pipeline

from fame3r import FAME3RScoreEstimator, FAME3RVectorizer

# Unfortunately, this monkey-patching is required to get CDPKit
# objects like atoms and molecules into NumPy arrays...
del Atom.__getitem__


def extract_som_labels(mol: MolecularGraph) -> list[tuple[Atom, bool]]:
    structure_data = {
        entry.header.split("<")[1].split(">")[0]: entry.data
        for entry in getStructureData(mol)
    }
    som_indices = (
        literal_eval(structure_data["soms"]) if "soms" in structure_data else []
    )

    return [(atom, atom.index in som_indices) for atom in mol.getAtoms()]


def read_labeled_atoms_from_sdf(path: PathLike) -> list[Atom]:
    results = []

    reader = FileSDFMoleculeReader(str(path))
    mol = BasicMolecule()
    while reader.read(mol):
        results.extend(extract_som_labels(mol))
        mol = BasicMolecule()

    return results


def top2_rate_score(y_true, y_prob, groups):
    unique_groups, _ = np.unique(groups, return_index=True)
    top2_sucesses = 0

    for current_group in unique_groups:
        mask = groups == current_group

        # Sort by predicted probability (descending) and take the top 2
        top_2_indices = np.argsort(y_prob[mask])[-2:]
        if y_true[mask][top_2_indices].sum() > 0:
            top2_sucesses += 1

    return top2_sucesses / len(unique_groups)


stdout = Console()
stderr = Console(stderr=True)


@contextmanager
def Spinner(*, title: str):
    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}[/]"),
        TimeElapsedColumn(),
        transient=True,
        console=stderr,
    ) as progress:
        progress.add_task(description=title, total=None)
        yield
        time_elapsed = TimeElapsedColumn().render(progress.tasks[-1])

    stderr.print(
        f"[progress.spinner]⠿[/] {title} [progress.elapsed]{time_elapsed}[/]",
        highlight=False,
    )


app = typer.Typer(add_completion=False)


@app.command(
    name="train",
    help="Train a FAME3R model and/or scorer on SOM data.",
)
def train(
    input_path: Annotated[
        Path,
        typer.Option(
            "--input",
            "-i",
            help="Path to input SOM training data (SDF).",
        ),
    ],
    models_path: Annotated[
        Path,
        typer.Option(
            "--output",
            "-o",
            help="Path to model output directory.",
        ),
    ],
    radius: Annotated[
        int,
        typer.Option(
            "--radius",
            help="Atom environment radius in bonds.",
        ),
    ] = 5,
    model_kinds: Annotated[
        list[str],
        typer.Option(
            "--kind",
            help="Models kinds to train.",
        ),
    ] = ["random-forest", "fame-scorer"],
    hyperparameter_path: Annotated[
        Path | None,
        typer.Option(
            "--hyperparameters",
            help="Path to JSON file containing model hyperparameters.",
        ),
    ] = None,
    n_neighbors: Annotated[
        int, typer.Option(help="Number of neighbors for FAME score estimator.")
    ] = 3,
):
    som_atoms_labeled = read_labeled_atoms_from_sdf(input_path)

    atom_count = len(som_atoms_labeled)
    mol_count = len({atom.molecule.getObjectID() for atom, _ in som_atoms_labeled})

    if "random-forest" in model_kinds:
        hyperparameters = (
            json.loads(hyperparameter_path.read_text())
            if hyperparameter_path
            else {
                "n_estimators": 250,
                "max_depth": None,
                "min_samples_split": 2,
                "min_samples_leaf": 1,
                "max_features": "sqrt",
                "class_weight": "balanced_subsample",
            }
        )

        with Spinner(
            title=f"Training random forest on {atom_count} atoms ({mol_count} molecules)"
        ):
            score_pipeline = make_pipeline(
                FAME3RVectorizer(radius=radius, input="cdpkit"),
                RandomForestClassifier(random_state=42, **hyperparameters),
            ).fit(
                [[som_atom] for som_atom, _ in som_atoms_labeled],
                [label for _, label in som_atoms_labeled],
            )

        models_path.mkdir(exist_ok=True, parents=True)
        joblib.dump(
            score_pipeline.named_steps["randomforestclassifier"],
            models_path / "random_forest_classifier.joblib",
        )

    if "fame-scorer" in model_kinds:
        with Spinner(
            title=f"Training FAME score estimator on {atom_count} atoms ({mol_count} molecules)"
        ):
            score_pipeline = make_pipeline(
                FAME3RVectorizer(radius=radius, input="cdpkit", output=["fingerprint"]),
                FAME3RScoreEstimator(n_neighbors=n_neighbors),
            ).fit(
                [[som_atom] for som_atom, _ in som_atoms_labeled],
                [label for _, label in som_atoms_labeled],
            )

        models_path.mkdir(exist_ok=True, parents=True)
        joblib.dump(
            score_pipeline.named_steps["fame3rscoreestimator"],
            models_path / "fame3r_score_estimator.joblib",
        )


@app.command(
    name="infer",
    help="Predict SOMs using an existing FAME3R model.",
)
def infer(
    input_path: Annotated[
        Path,
        typer.Option(
            "--input",
            "-i",
            help="Path to input data for which to predict SOMs (SDF).",
        ),
    ],
    models_path: Annotated[
        Path,
        typer.Option(
            "--models",
            "-m",
            help="Path to existing model directory.",
        ),
    ],
    output_path: Annotated[
        Path,
        typer.Option(
            "--output",
            "-o",
            help="Path to output prediction CSV file.",
        ),
    ],
    radius: Annotated[
        int,
        typer.Option(
            "--radius",
            help="Atom environment radius in bonds.",
        ),
    ] = 5,
    threshold: Annotated[
        float,
        typer.Option(
            "--threshold",
            help="Prediction probability threshold",
        ),
    ] = 0.3,
    compute_fame_score: Annotated[
        bool,
        typer.Option(
            "--fame-score", help="Also compute FAME score for applicability."
        ),
    ] = False,
):
    som_atoms = read_labeled_atoms_from_sdf(input_path)

    atom_count = len(som_atoms)
    mol_count = len({atom.molecule.getObjectID() for atom, _ in som_atoms})

    classifier = joblib.load(models_path / "random_forest_classifier.joblib")
    clf_pipeline = make_pipeline(
        FAME3RVectorizer(radius=radius, input="cdpkit").fit(), classifier
    )

    with Spinner(
        title=f"Predicting SOM probabilities for {atom_count} atoms ({mol_count} molecules)"
    ):
        prediction_probabilities = clf_pipeline.predict_proba(
            [[som_atom] for som_atom, _ in som_atoms]
        )[:, 1]
        predictions_binary = prediction_probabilities > threshold

    score_estimator = joblib.load(models_path / "fame3r_score_estimator.joblib")
    score_pipeline = make_pipeline(
        FAME3RVectorizer(radius=radius, input="cdpkit", output=["fingerprint"]).fit(),
        score_estimator,
    )

    if compute_fame_score:
        with Spinner(
            title=f"Predicting FAME scores for {atom_count} atoms ({mol_count} molecules)"
        ):
            fame_scores = score_pipeline.predict(
                [[som_atom] for som_atom, _ in som_atoms]
            )
    else:
        fame_scores = np.full_like(prediction_probabilities, np.nan)

    with output_path.open("w", encoding="UTF-8", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "smiles",
                "atom_id",
                "y_true",
                "y_pred",
                "y_prob",
                "fame_score",
            ],
        )
        writer.writeheader()

        for i in range(len(som_atoms)):
            writer.writerow(
                {
                    "smiles": generateSMILES(som_atoms[i][0].molecule),
                    "atom_id": som_atoms[i][0].index,
                    "y_true": int(som_atoms[i][1]),
                    "y_pred": int(predictions_binary[i]),
                    "y_prob": np.round(prediction_probabilities[i], 2),
                    "fame_score": np.round(fame_scores[i], 2),
                }
            )


@app.command(
    name="metrics",
    help="Calculate metrics for existing SOM predictions.",
)
def metrics(
    input_path: Annotated[
        Path,
        typer.Option(
            "--input",
            "-i",
            help="Path to input prediction CSV file.",
        ),
    ],
    output_path: Annotated[
        Path | None,
        typer.Option(
            "--output",
            "-o",
            help="Path to output metrics JSON file.",
        ),
    ] = None,
):
    rows = [row for row in csv.DictReader(input_path.open())]

    smiles = np.array([row["smiles"] for row in rows], dtype=str)
    y_true = np.array([int(row["y_true"]) for row in rows], dtype=bool)
    y_pred = np.array([int(row["y_pred"]) for row in rows], dtype=bool)
    y_prob = np.array([row["y_prob"] for row in rows], dtype=float)

    computed_metrics = {
        "roc_auc": np.round(roc_auc_score(y_true, y_prob), 4),
        "average_precision": np.round(average_precision_score(y_true, y_prob), 4),
        "f1": np.round(f1_score(y_true, y_pred), 4),
        "matthews_corrcoef": np.round(matthews_corrcoef(y_true, y_pred), 4),
        "precision": np.round(precision_score(y_true, y_pred), 4),
        "recall": np.round(recall_score(y_true, y_pred), 4),
        "top2_rate": np.round(top2_rate_score(y_true, y_prob, smiles), 4),
    }

    if output_path:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps(computed_metrics, indent=4))
    else:
        stderr.print(JSON.from_data(computed_metrics))


@app.command(
    name="hyperparameters",
    help="Perform CV hyperparameter search for a FAME3R model.",
)
def hyperparameters(
    input_path: Annotated[
        Path,
        typer.Option(
            "--input",
            "-i",
            help="Path to SOM training/cross-validation data.",
        ),
    ],
    output_path: Annotated[
        Path | None,
        typer.Option(
            "--output",
            "-o",
            help="Path to output hyperparameter JSON file.",
        ),
    ] = None,
    radius: Annotated[
        int,
        typer.Option(
            "--radius",
            help="Atom environment radius in bonds.",
        ),
    ] = 5,
    num_folds: Annotated[
        int,
        typer.Option(
            "--num-folds",
            help="Number of cross-validation folds to perform.",
        ),
    ] = 10,
):
    # Required for passing KFold groups to cross-validation
    sklearn.set_config(enable_metadata_routing=True)

    som_atoms_labeled = read_labeled_atoms_from_sdf(input_path)

    labels = [label for _, label in som_atoms_labeled]
    containing_mol_ids = [atom.molecule.getObjectID() for atom, _ in som_atoms_labeled]

    atom_count = len(som_atoms_labeled)
    mol_count = len(set(containing_mol_ids))

    with Spinner(
        title=f"Computing descriptors for {atom_count} atoms ({mol_count} molecules)"
    ):
        descriptors = FAME3RVectorizer(radius=radius, input="cdpkit").fit_transform(
            [[som_atom] for som_atom, _ in som_atoms_labeled]
        )

    k_fold = GroupKFold(n_splits=num_folds, random_state=42, shuffle=True)

    with Spinner(title=f"Performing CV on {atom_count} atoms ({mol_count} molecules)"):
        grid_search = GridSearchCV(
            RandomForestClassifier(random_state=42),
            {
                "n_estimators": [100, 250, 500, 750],
                "max_depth": [None, 10, 20, 30],
                "min_samples_split": [2, 5, 10],
                "min_samples_leaf": [1, 2, 4],
                "max_features": ["sqrt", "log2"],
                "class_weight": ["balanced", "balanced_subsample"],
            },
            cv=k_fold,
            scoring=make_scorer(average_precision_score, greater_is_better=True),
            n_jobs=-1,
            verbose=3,
        ).fit(descriptors, labels, groups=containing_mol_ids)

    if output_path:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps(grid_search.best_params_, indent=4))
    else:
        stdout.print(JSON.from_data(grid_search.best_params_))


@app.command(
    name="threshold",
    help="Perform CV threshold post-tuning for a FAME3R model.",
)
def threshold(
    input_path: Annotated[
        Path,
        typer.Option(
            "--input",
            "-i",
            help="Path to SOM testing/cross-validation data.",
        ),
    ],
    model_path: Annotated[
        Path,
        typer.Option(
            "--models",
            "-m",
            help="Path to existing FAME3R classifier model file.",
        ),
    ],
    output_path: Annotated[
        Path | None,
        typer.Option(
            "--output",
            "-o",
            help="Path to output threshold JSON file.",
        ),
    ] = None,
    radius: Annotated[
        int,
        typer.Option(
            "--radius",
            help="Atom environment radius in bonds.",
        ),
    ] = 5,
    num_folds: Annotated[
        int,
        typer.Option(
            "--num-folds",
            help="Number of cross-validation folds to perform.",
        ),
    ] = 10,
):
    # Required for passing KFold groups to cross-validation
    sklearn.set_config(enable_metadata_routing=True)

    som_atoms_labeled = read_labeled_atoms_from_sdf(input_path)

    labels = [label for _, label in som_atoms_labeled]
    containing_mol_ids = [atom.molecule.getObjectID() for atom, _ in som_atoms_labeled]

    atom_count = len(som_atoms_labeled)
    mol_count = len(set(containing_mol_ids))

    with Spinner(
        title=f"Computing descriptors for {atom_count} atoms ({mol_count} molecules)"
    ):
        descriptors = FAME3RVectorizer(radius=radius, input="cdpkit").fit_transform(
            [[som_atom] for som_atom, _ in som_atoms_labeled]
        )

    k_fold = GroupKFold(n_splits=num_folds, random_state=42, shuffle=True)

    classifier = joblib.load(model_path)

    with Spinner(
        title=f"Tuning threshold on {atom_count} atoms ({mol_count} molecules)"
    ):
        tuner = TunedThresholdClassifierCV(
            classifier,
            scoring="matthews_corrcoef",
            cv=k_fold,
            # Multiple jobs with TunedThresholdClassifierCV are buggy
            n_jobs=1,
            random_state=42,
        ).fit(descriptors, labels, groups=containing_mol_ids)

    if output_path:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps(tuner.best_threshold_, indent=4))
    else:
        stdout.print(JSON.from_data({"best_threshold": tuner.best_threshold_}))


@app.command(
    name="descriptors",
    help="Compute FAME3R descriptors for external use.",
)
def descriptors(
    input_path: Annotated[
        Path,
        typer.Option(
            "--input",
            "-i",
            help="Path to SOM input data.",
        ),
    ],
    output_path: Annotated[
        Path,
        typer.Option(
            "--output",
            "-o",
            help="Path to output descriptors CSV file.",
        ),
    ],
    radius: Annotated[
        int,
        typer.Option(
            "--radius",
            help="Atom environment radius in bonds.",
        ),
    ] = 5,
    included_descriptors: Annotated[
        list[str],
        typer.Option(
            "--subset",
            help="Subsets of FAME3R descriptors to generate.",
        ),
    ] = ["fingerprint", "physicochemical", "topological"],
):
    som_atoms = read_labeled_atoms_from_sdf(input_path)
    containing_mol_ids = [atom.molecule.getObjectID() for atom, _ in som_atoms]

    atom_count = len(som_atoms)
    mol_count = len(set(containing_mol_ids))

    vectorizer = FAME3RVectorizer(
        radius=radius,
        input="cdpkit",
        output=included_descriptors,  # pyright:ignore[reportArgumentType]
    )

    with Spinner(
        title=f"Computing descriptors for {atom_count} atoms ({mol_count} molecules)"
    ):
        descriptors = vectorizer.fit_transform(
            [[som_atom] for som_atom, _ in som_atoms]
        )

    with output_path.open("w", encoding="UTF-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["smiles", "atom_id"] + vectorizer.get_feature_names_out())

        for i in range(len(som_atoms)):
            writer.writerow(
                [
                    generateSMILES(som_atoms[i][0].molecule),
                    som_atoms[i][0].index,
                ]
                + list(descriptors[i])
            )


if __name__ == "__main__":
    app()
