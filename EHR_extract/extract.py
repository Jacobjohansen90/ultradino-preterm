import hydra
import json
import polars as pl
#from dotenv import load_dotenv
from omegaconf import DictConfig, OmegaConf

from EHR_extract.custom_find_functions import (find_close_births,
                                               find_scantime_ga,
                                               find_images_and_timedeltas,
                                               find_multiple_pregnancies,
                                               find_images_with_predicted_classes)

from EHR_extract.paths import get_config_path

from EHR_extract.utils import (filter_numeric_rows,
                               get_python_operator,
                               load_table,
                               update_population,
                               write_imaging_metadata_to_formats,
                               merge_population_tables)

#load_dotenv()

custom_functions = {"find_close_births": find_close_births,
                    "find_images_and_timedeltas": find_images_and_timedeltas,
                    "find_scantime_ga": find_scantime_ga,
                    "find_multiple_pregnancies": find_multiple_pregnancies,
                    "find_images_with_predicted_classes": find_images_with_predicted_classes}


def merge_tables(cfg):
    return merge_population_tables(cfg.population.tables)

def inclusion_exclusion(cfg, population, logger):
    all_discards = []
    population_with_img_data = None

    if 'conditional_criteria' in cfg.keys():
        for criterion in cfg.conditional_criteria:
            criterion_population = set()
            for condition in criterion.conditions:
                table = load_table(condition.table, strict=cfg.strict)
                logger.info(f"Table rows / unique IDs total: {len(table)} / "\
                            "{table[condition.match_on].n_unique()} for table: {condition.table}")
    
                table = table.filter(pl.col(condition.match_on).is_in(population[cfg.population.population_key]))
                logger.info(f"Table rows / unique IDs matching population IDs: {len(table)} / "\
                            "{table[condition.match_on].n_unique()} after filtering on {condition.match_on}")
    
                py_operator = get_python_operator(condition.operator)
                if condition.operator in [">", "<", ">=", "<="]:
                    table = filter_numeric_rows(table, condition.column)
                table = table.filter(py_operator(pl.col(condition.column), condition.value))
                logger.info(f"Table rows / unique IDs matching population IDs: {len(table)} / "\
                            "{table[condition.match_on].n_unique()} after filtering on "\
                            "{condition.column} {condition.operator} {condition.value}")
    
                if condition.condition is None:
                    last_condition_population = set(table[condition.match_on])
                elif condition.condition == "and":
                    last_condition_population = last_condition_population.intersection(set(table[condition.match_on]))
                elif condition.condition == "or":
                    criterion_population = last_condition_population
                    last_condition_population = set(table[condition.match_on])
                else:
                    logger.debug(f"{condition.condition} not implemented")
    
            criterion_population = criterion_population.union(last_condition_population)
            population, discards, n_discards, n_population_before_discard = update_population(
                population=population,
                key=cfg.population.population_key,
                subset=set(criterion_population),
                action=criterion.action,
            )
            logger.info(f"Population size: {len(population)} after filtering on criteria {criterion}")
            all_discards.append([OmegaConf.to_container(criterion), list(discards), n_discards, n_population_before_discard])
    
    logger.info("\n ### Applying custom criteria ### \n")
    if 'custom_criteria' in cfg.keys():
        for custom_cfg in cfg.get("custom_criteria", {}):
            fn = custom_functions[custom_cfg.function]
            args = custom_cfg.args
            set_of_matches = fn(**args, population=set(population.get_column(cfg.population.population_key)))
            population, discards, n_discards, n_population_before_discard = update_population(
                population=population,
                key=cfg.population.population_key,
                subset=set_of_matches,
                action=custom_cfg.action,
            )
            all_discards.append([OmegaConf.to_container(custom_cfg), list(discards), n_discards, n_population_before_discard])
    
            logger.info(f"Population size: {len(population)} after filtering on custom criteria {custom_cfg.function}")
            logger.info("---")
    logger.info("\n ### Applying imaging matching criteria ### \n")
    if 'imaging_matching_criteria' in cfg.keys():
        for custom_cfg in cfg.get("imaging_matching_criteria", {}):
            fn = custom_functions[custom_cfg.function]
            args = custom_cfg.args
            set_of_ID_matches, population_with_img_data = fn(
                **args,
                population=population,
                population_with_img_data=population_with_img_data,
                population_key_column=cfg.population.population_key)
            population, discards, n_discards, n_population_before_discard = update_population(
                population=population,
                key=cfg.population.population_key,
                subset=set_of_ID_matches,
                action=custom_cfg.action,
            )
            all_discards.append([OmegaConf.to_container(custom_cfg), list(discards), n_discards, n_population_before_discard])
    
            logger.info(f"Population size: {len(population)} after filtering on custom criteria {custom_cfg.function}")
            logger.info("---")
    return population_with_img_data, all_discards

def make_train_test_split(holdout_csv_path, population, file_path_key, prefix, has_header=False):
    holdout = load_table(holdout_csv_path, has_header=has_header)
    holdout = holdout.with_columns(pl.col('column_1').str.replace_all(prefix, ''))
    train = population.filter(pl.col(file_path_key).is_in(holdout['column_1']).not_())
    test = population.filter(pl.col(file_path_key).is_in(holdout['column_1']))
    return train, test
    

@hydra.main(
    config_path=get_config_path(),
    config_name="Preterm",
    version_base="1.2",
)
def main(cfg: DictConfig) -> None:
    population = merge_population_tables(cfg)
    population, discards, imaging_metadata = inclusion_exclusion(cfg, population)

    d = {}
    for i in range(len(discards)):
        d[i] = {
            "criteria": discards[i][0],
            "n_discards": discards[i][2],
            "n_population_pre_discard": discards[i][3],
            "n_population_post_discard": discards[i][3] - discards[i][2],
            "discards": discards[i][1],
        }

    with open(cfg.paths.discards_save_path, "w") as fp:
        json.dump(d, fp, indent=4)

    population.write_csv(cfg.paths.population_save_path)
    if imaging_metadata is not None:
        write_imaging_metadata_to_formats(
            imaging_metadata, cfg.paths.imaging_metadata_output_formats, cfg.paths.imaging_metadata_save_path
        )


if __name__ == "__main__":
    main()
