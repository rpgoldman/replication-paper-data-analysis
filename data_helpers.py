import os
import sys
import json
import numpy as np
import pandas as pd
import xarray as xr
import math
import logging
import re

from typing import List

__all__ = ['all_of_strain', 'canonical_data_frame', 'files_with_strain', 'gate_dataframe', 'compute_gated_measures', 'df_create_well_column', 'well_rewrite', 'NEW_GFP_COLUMNS']

NEW_GFP_COLUMNS = ['mean_log_gfp_gated',
                   'std_log_gfp_gated',
                   'gated_count',
                   'mean_log_gfp_gated_live',
                   'std_log_gfp_gated_live',
                   'gated_live_count']

def df_add_new_gfp_columns(df: pd.DataFrame) -> None:
    """
    Add new columns -- filled with NaN -- to the input DataFrame.
    
    Column names come from ``NEW_GFP_COLUMNS``.
    """
    for x in NEW_GFP_COLUMNS:
        df.loc[:, x] = np.nan


def all_of_strain(strain_name: str, files: List[str]) -> pd.DataFrame:
    """
    Return a DataFrame with all of the Flow Cytometry data points for ``strain_name``,
    taken from ``files``.  ``files`` should be a list computed by ``files_with_strain()``.
    """
    df: pd.DataFrame = pd.read_csv(files.pop(), index_col=0)
    df = df[df['strain_name'] == strain_name]

    for x in files:
        new_df = pd.read_csv(x, index_col=0)
        new_df = new_df[new_df['strain_name'] == strain_name]
        df = pd.concat([df, new_df])
        new_df = None
    return df

def canonical_data_frame(df: pd.DataFrame) -> pd.DataFrame:
    """
    Put DataFrame of Flow Cytometry data into canonical form.
    
    Returns
    ~~~~~~~
    New pd.DataFrame in canonical form.
    
    Canonical form is indexed by:
    * `strain_name`
    * `inc_temp_degrees`
    * `inc_time_2_hrs`
    * `od`
    * `lab_id`
    * `plate_id`
    * `well`
    * `replicate` and
    * `event`
    
    Additionally, `growth_media_1` is replaced by a pandas `CategoricalDtype`, and we add
    `inc_temp_degrees`, `inc_time_1_hrs` and `inc_time_2_hrs` columns, which are easier 
    to deal with than the columns with units.
    """
    df = df.copy()
    df.loc[:, 'media'] = \
        df['growth_media_1'].astype(pd.CategoricalDtype(sorted(df['growth_media_1'].unique())))
    df.loc[:, 'inc_temp_degrees'] = np.vectorize(lambda x: int(x.split("_")[1]))(df['inc_temp'])
    df.loc[:, 'inc_time_1_hrs'] = np.vectorize(lambda x: int(x.split(":")[0]))(df['inc_time_1'])
    df.loc[:, 'inc_time_2_hrs'] = np.vectorize(lambda x: int(x.split(":")[0]))(df['inc_time_2'])
    df_create_well_column(df)
    df.loc[:, 'replicate'] = df.groupby(['lab_id', 'plate_id', 'well']).ngroup()
    df.loc[:, 'event'] = df.groupby(['lab_id', 'plate_id', 'well']).cumcount()
    if 'lab' in df.columns or 'plan' in df.columns:
        df.drop(columns=['lab', 'plan'], inplace=True)
    df.set_index(['strain_name', 'inc_temp_degrees', 'inc_time_2_hrs', 'od', 'lab_id', 'plate_id', 'well', 'replicate', 'event'], inplace=True)
    return df

def files_with_strain(strain_name: str, filename: str = 'accuracy_set.csv') -> List[str]:
    table = pd.read_csv(filename)
    files = (x + '.csv' for x in table.query(f'strain_name == "{strain_name}"')['lab_id'].unique())
    return list(files)

WELL_SPLIT_RE = re.compile(r"([A-H])([1-9][0-9]?)")

def well_rewrite(well: str) -> str:
    match = re.match(WELL_SPLIT_RE, well.upper())
    assert match is not None
    col, row = match.groups()
    row = int(row)
    return f"{col}{row:02d}"


def df_create_well_column(df: pd.DataFrame) -> None:
    raw_wells = np.vectorize(lambda x: x.split("_")[-1])(df['id'])
    wells = np.vectorize(well_rewrite)(raw_wells)
    df.loc[:, 'well'] = wells


def gate_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    logger = logging.getLogger("data_helpers")
    query_string = '25_000 <= FSC_A <= 900_000 and 50_000 <= SSC_A <= 900_000 and 0 <= BL1_A <= 20_000'
    new_df = df.query(query_string)
    logging.info(f"After gating, have gone from {df.shape[0]} events to {new_df.shape[0]}.")
    dropped_replicates = set()
    def filter_small_replicate(df):
        tuple = df.iloc[0, :]['xx_lab_id'], df.iloc[0, :]['xx_well']
        dropped_replicates.add(tuple)
        return None if df.shape[0] < 10_000 else df
    new_df.loc[:, 'xx_lab_id'] = new_df.index.get_level_values('lab_id')
    new_df.loc[:, 'xx_well'] = new_df.index.get_level_values('well')
    
    new_df2 = new_df.groupby(['lab_id', 'well']).apply(lambda df: None if df.shape[0] < 10_000 else df)
    new_df2.drop(columns=['xx_lab_id', 'xx_well'], inplace=True)
    
    if dropped_replicates:
        logging.info("Dropped replicates with low counts.")
        for x in dropped_replicates: logging.info("Dropped replicate ", x)
    else:
        logging.debug("All replicates have acceptable counts.")
    return new_df2

def compute_gated_scores(data_df: pd.DataFrame) -> pd.DataFrame:
    df = data_df
    df.loc[:, 'logGFP'] = np.log10(np.maximum(df['BL1_A'], 1))
    groups: pd.GroupBy = df.groupby(level=['lab_id', 'strain_name', 'plate_id', 'well'])
    gated = groups.agg(mean_log_gfp_gated=('logGFP', 'mean'), 
                       std_log_gfp_gated=('logGFP', 'std'), 
                       gated_count=('logGFP', 'count'))
    groups = df[df['live'] == 1].groupby(level=['lab_id', 'strain_name', 'plate_id', 'well'])
    gated_live = groups.agg(mean_log_gfp_gated_live=('logGFP', 'mean'), 
                            std_log_gfp_gated_live=('logGFP', 'std'), 
                            gated_live_count=('logGFP', 'count'))
    return gated.join(gated_live, how='outer')
    
    