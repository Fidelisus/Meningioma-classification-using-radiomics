from pathlib import Path

from meningioma_random_forest.config import PLOTS_PATH, BASE_DIR, TEST_SIZE, RANDOM_SEED
from meningioma_random_forest.data_loading.data_loader import load_train_test_splits
from meningioma_random_forest.evaluation.evaluation import evaluate_best_model
from meningioma_random_forest.training.training_loop import training_loop


def run_random_forest(
    radiomics_file_name: str,
    base_dir: Path,
    random_seed: int = 123,
    redo_split: bool = False,
    test_size: float = 0.2,
    drop_correlated_features: bool = False,
    only_unfiltered_radiomics: bool = False,
):
    test_dataframe, train_dataframe = load_train_test_splits(
        base_dir, radiomics_file_name, redo_split, test_size
    )

    rf, x_dataframe_uncorrelated = training_loop(
        train_dataframe,
        random_seed=random_seed,
        drop_correlated_features=drop_correlated_features,
        only_unfiltered_radiomics=only_unfiltered_radiomics,
    )

    evaluate_best_model(rf, test_dataframe, x_dataframe_uncorrelated)


if __name__ == "__main__":
    radiomics_file_name = "radiomics_features_2023-01-30.csv"
    only_unfiltered_radiomics = False
    PLOTS_PATH.mkdir(exist_ok=True)
    BASE_DIR.mkdir(exist_ok=True)

    run_random_forest(
        radiomics_file_name=radiomics_file_name,
        base_dir=BASE_DIR,
        redo_split=True,
        test_size=TEST_SIZE,
        only_unfiltered_radiomics=only_unfiltered_radiomics,
        random_seed=RANDOM_SEED,
    )
