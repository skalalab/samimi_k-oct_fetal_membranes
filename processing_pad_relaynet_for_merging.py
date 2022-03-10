import pandas as pd
import numpy as np

def pad_relaynet_for_merging(df_apex_rise_pressure: pd.DataFrame, 
                              df_thickness: pd.DataFrame) -> pd.DataFrame:
    """
    Helper function to pad relaynet output to match number of rows in apex_rise
    vs pressure file. This happens because ReLayNet errors out near the end of 
    the sample when the fetal membrane busts and there is nothing to segment
    in the frame

    Parameters
    ----------
    df_apex_rise_pressure : pd.DataFrame
        dataframe containig apex_rise vs pressure registration
    df_thickness : pd.DataFrame
        dataframe containing thickness calculations 

    Returns
    -------
    dataframe with merged apex_rise_vs_pressure and thickness measurements

    """
    ### check that they are the same length, extend if necessary
    if not len(df_apex_rise_pressure) == len(df_thickness):
        
            ## fill thickness df to match apex rise
            num_missing_rows= len(df_apex_rise_pressure) - len(df_thickness)
            df_copy = df_thickness.copy()
            # create list of indices to match apex rise df
            list_new_indices = np.arange(num_missing_rows) + len(df_thickness)
            # print(df_copy.loc[len(df_copy.index)-1])
            filler_row = ["NaN"] * len(df_thickness.columns) # create filler row matching number of cols 
            for idx in list_new_indices: # fill indicex with filler row
                df_copy.loc[idx] = filler_row
            df_thickness = df_copy # replace df
            
    return df_thickness