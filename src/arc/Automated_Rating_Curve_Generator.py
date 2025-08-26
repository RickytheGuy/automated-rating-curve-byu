
"""
Written initially by Mike Follum with Follum Hydrologic Solutions, LLC.
Program simply creates depth, velocity, and top-width information for each stream cell in a domain.

"""

import sys
import os
import math
import warnings
from multiprocessing import Pool

import tqdm
import numpy as np
import pandas as pd
from datetime import datetime
from scipy.optimize import OptimizeWarning
from osgeo import gdal

from arc import LOG
import arc.functions as f

warnings.filterwarnings("ignore", category=OptimizeWarning)
gdal.UseExceptions()

def main(MIF_Name: str, args: dict, quiet: bool):
    starttime = datetime.now()  
    ### Read Main Input File ###
    conf = f.read_main_input_file(MIF_Name, args)
    
    ### Read the Flow Information ###
    COMID, QBaseFlow, QMax = f.read_flow_file(conf)

    ### Read Raster Data ###
    ### Imbed the Stream and DEM data within a larger Raster to help with the boundary issues. ###
    i_boundary_number = max(1, conf['Gen_Dir_Dist'], conf['Gen_Slope_Dist'])

    # Load DEM
    dem_ds: gdal.Dataset = gdal.Open(conf['DEM_File'])
    ncols = dem_ds.RasterXSize
    nrows = dem_ds.RasterYSize
    dcellsize = dem_ds.GetGeoTransform()[1]
    dyll = dem_ds.GetGeoTransform()[3] - nrows * np.fabs(dem_ds.GetGeoTransform()[5])
    dyur = dem_ds.GetGeoTransform()[3]
    dem_geotransform = dem_ds.GetGeoTransform()
    dem_projection = dem_ds.GetProjection()
    array_shape = (ncols + i_boundary_number * 2, nrows + i_boundary_number * 2)
    dem_ds = None

    # Check that land use and streams have same shape
    stream_ds: gdal.Dataset = gdal.Open(conf['Stream_File'])
    if stream_ds.RasterXSize != ncols or stream_ds.RasterYSize != nrows:
        LOG.error('Stream file does not match DEM dimensions!')
        return
    stream_ds = None

    land_ds: gdal.Dataset = gdal.Open(conf['LU_Raster_SameRes'])
    if land_ds.RasterXSize != ncols or land_ds.RasterYSize != nrows:
        LOG.error('Land use file does not match DEM dimensions!')
        return
    land_ds = None

    # if the DEM contains negative values, add 100 m to the height to get rid of the negatives, we'll subtract it back out later
    b_modified_dem = False
    dem_shmem = f.create_shared_memory(conf['DEM_File'], 'dem', array_shape, np.float32)
    dm_elevation = np.ndarray(shape=array_shape, dtype=np.float32, buffer=dem_shmem.buf)
    b_modified_dem = f.modify_dem(dm_elevation, b_modified_dem)

    # Load stream
    stream_shmem = f.create_shared_memory(conf['Stream_File'], 'stream', array_shape, np.int64)
    dm_stream = np.ndarray(shape=array_shape, dtype=np.int64, buffer=stream_shmem.buf)

    # Load land use
    land_use_shmem = f.create_shared_memory(conf['LU_Raster_SameRes'], 'land_use', array_shape, np.float64)
    dm_land_use = np.ndarray(shape=array_shape, dtype=np.float64, buffer=land_use_shmem.buf)    

    ##### Begin Calculations #####
    i_number_of_increments = conf['VDT_Database_NumIterations']

    if conf['bathymetry_file']:
        bathy_shmem = f.create_shared_memory(
            np.full(
                (nrows + i_boundary_number * 2, ncols + i_boundary_number * 2), 
                np.nan, 
                dtype=np.float32  # Optional: Specify dtype if needed
            ), 'bathymetry', array_shape, np.float32)
        dm_output_bathymetry = np.ndarray(shape=array_shape, dtype=np.float32, buffer=bathy_shmem.buf)

    # This is used for debugging purposes with stream and cross-section angles.
    #dm_output_streamangles = np.zeros((nrows + i_boundary_number * 2, ncols + i_boundary_number * 2))
    
    if conf['AROutFLOOD']:
        dm_out_flood = np.zeros((nrows + i_boundary_number * 2, ncols + i_boundary_number * 2)).astype(int)
        dm_out_flood[ia_valued_row_indices, ia_valued_column_indices] = 3

    # Get the list of stream locations
    ia_valued_row_indices, ia_valued_column_indices = np.where(np.isin(dm_stream, COMID, kind='table'))

    i_number_of_stream_cells = len(ia_valued_row_indices)

    # Get the original landcover value before we start changing it
    og_lu_shmem = f.create_shared_memory(dm_land_use.copy(), 'land_use_before_streams', array_shape, np.float64)
    
    # Make all Land Cover that is a stream look like water
    dm_land_use[ia_valued_row_indices, ia_valued_column_indices] = conf['LC_Water_Value']
    
    #Assing Manning n Values
    ### Read in the Manning Table ###
    manning_n_shmem = f.create_shared_memory(f.read_manning_table(conf['LU_Manning_n'], dm_land_use), 'manning_n', array_shape, np.float64)
    dm_manning_n_raster = np.ndarray(shape=array_shape, dtype=np.float64, buffer=manning_n_shmem.buf)

    # Correct the mannings values here
    # dm_manning_n_raster[dm_manning_n_raster > 0.3] = 0.035
    # dm_manning_n_raster[dm_manning_n_raster < 0.005] = 0.005
    

    # Get the cell dx and dy coordinates
    dx, dy, dproject = f.convert_cell_size(dcellsize, dyll, dyur)
    LOG.info('Cellsize X = ' + str(dx))
    LOG.info('Cellsize Y = ' + str(dy))

    i_center_point = int((conf['X_Section_Dist'] / (sum([dx, dy]) * 0.5)) / 2.0) + 1

    # Find all the different angle increments to test
    l_angles_to_test = [0.0]
    d_increments = 0

    d_degree_manipulation = conf['Degree_Manip']
    d_degree_interval = conf['Degree_Interval']
    if d_degree_manipulation > 0.0 and d_degree_interval > 0.0:
        # Calculate the increment
        d_increments = int(d_degree_manipulation / (2.0 * d_degree_interval))

        # Test if the increment should be considered
        if d_increments > 0:
            for d in range(1, d_increments + 1):
                for s in range(-1, 2, 2):
                    l_angles_to_test.append(s * d * d_degree_interval)

    # We now precompute the cross-section ordinates
    i_precompute_angles = 30
    d_precompute_angles = np.pi / i_precompute_angles

    # Create shared memory for index arrays
    index_shmems = [] # This list keeps the shared memory objects around. Otherwise Python will gc them and they will be lost
    index_shape = (i_precompute_angles + 1, i_center_point + 1)

    ia_xc_dr_index_main = f.create_index_shared_memory(index_shmems, 'ia_xc_dr_index_main', np.int64, index_shape)  # Only need to go to center point, because the other side of xs we can just use *-1
    ia_xc_dc_index_main = f.create_index_shared_memory(index_shmems, 'ia_xc_dc_index_main', np.int64, index_shape)  # Only need to go to center point, because the other side of xs we can just use *-1
    ia_xc_dr_index_second = f.create_index_shared_memory(index_shmems, 'ia_xc_dr_index_second', np.int64, index_shape)  # Only need to go to center point, because the other side of xs we can just use *-1
    ia_xc_dc_index_second = f.create_index_shared_memory(index_shmems, 'ia_xc_dc_index_second', np.int64, index_shape)  # Only need to go to center point, because the other side of xs we can just use *-1
    d_distance_z = f.create_index_shared_memory(index_shmems, 'd_distance_z', np.float64, (index_shape[0],))
    da_xc_main_fract = f.create_index_shared_memory(index_shmems, 'da_xc_main_fract', np.float64, index_shape)
    da_xc_second_fract = f.create_index_shared_memory(index_shmems, 'da_xc_second_fract', np.float64, index_shape)
    da_xc_main_fract_int = f.create_index_shared_memory(index_shmems, 'da_xc_main_fract_int', np.int64, index_shape)
    da_xc_second_fract_int = f.create_index_shared_memory(index_shmems, 'da_xc_second_fract_int', np.int64, index_shape)

    for i in range(i_precompute_angles+1):
        d_xs_direction = d_precompute_angles * i
        # Get the Cross-Section Ordinates
        d_distance_z[i], da_xc_main_fract_int[i], da_xc_second_fract_int[i] = f.get_xs_index_values_precalculated(ia_xc_dr_index_main[i], ia_xc_dc_index_main[i], ia_xc_dr_index_second[i], ia_xc_dc_index_second[i], da_xc_main_fract[i], da_xc_second_fract[i], d_xs_direction,
                                                                                           i_center_point, dx, dy)

    LOG.info('With Degree_Manip=' + str(d_degree_manipulation) + '  and  Degree_Interval=' + str(d_degree_interval) + '\n  Angles to evaluate= ' + str(l_angles_to_test))
    l_angles_to_test = np.multiply(l_angles_to_test, math.pi / 180.0)
    LOG.info('  Angles (radians) to evaluate= ' + str(l_angles_to_test))

    # Here we will capture a list of all stream cell values that will be used if we build a reach average curve file
    if conf['Print_Curve_File'] and conf['Reach_Average_Curve_File']:
        All_COMID_curve_list = []
        All_Row_curve_list = []
        All_Col_curve_list = []
        All_BaseElev_curve_list = []
        All_DEM_Elev_curve_list = []
        All_QMax_curve_list = []
    
    # instantiate the lists we will use to create the XS File
    if conf['XS_Out_File']:
        XS_COMID_List = []
        XS_Row_List = []
        XS_Col_List = []
        # da_xs_profile1_str
        XS_da_xs_profile1_str = []
        XS_da_xs_profile2_str = []
        # dm_manning_n_raster1_str
        XS_dm_manning_n_raster1_str = []
        XS_dm_manning_n_raster2_str = []
        # d_ordinate_dist
        XS_d_ordinate_dist = []
        # r1, c1, r2, c2
        XS_r1 = []
        XS_c1 = []
        XS_r2 = []
        XS_c2 = []

    # Create the dictionary and lists that will be used to create our VDT database
    vdt_list: list = []
    q_list = []
    v_list = []
    t_list = []
    wse_list = []

    # Create the dictionary and lists that will be used to create our ATW database
    if conf['Print_AP_Database']:
        comid_ap_dict_list = []
        row_ap_dict_list = []
        col_ap_dict_list = []
        ap_q_list = []
        ap_a_list = []
        ap_p_list = []
    
    #Create the list that we will use to generate the output Curve file
    if conf['Print_Curve_File']:
        COMID_curve_list = []
        Row_curve_list = []
        Col_curve_list = []
        BaseElev_curve_list = []
        DEM_Elev_curve_list = []
        QMax_curve_list = []
        depth_a_curve_list = []
        depth_b_curve_list = []
        tw_a_curve_list = []
        tw_b_curve_list = []
        vel_a_curve_list = []
        vel_b_curve_list = []

    # Write the percentiles into the files
    LOG.info('Looking at ' + str(i_number_of_stream_cells) + ' stream cells')

    # Some stuff to send to processes
    conf['b_modified_dem'] = b_modified_dem
    conf['array_shape'] = array_shape
    conf['dx'] = dx
    conf['dy'] = dy
    conf['i_center_point'] = i_center_point
    conf['d_increments'] = d_increments
    conf['l_angles_to_test'] = l_angles_to_test
    conf['i_boundary_number'] = i_boundary_number
    conf['nrows'] = nrows
    conf['ncols'] = ncols
    conf['d_precompute_angles'] = d_precompute_angles
    conf['index_shape'] = index_shape

    args = []
    for i in range(i_number_of_stream_cells):
        row, col = ia_valued_row_indices[i], ia_valued_column_indices[i]
        i_cell_comid = dm_stream[row, col]
        try:
            im_flow_index = np.where(COMID == i_cell_comid)[0][0]
            d_q_baseflow = QBaseFlow[im_flow_index]
            d_q_maximum = QMax[im_flow_index]
        except:
            # print("I cant find the flow index")
            continue

        args.append((conf, i, row, col, d_q_baseflow, d_q_maximum))

    ### Begin the stream cell solution loop ###
    # Global DEMS: fab, alos, tilezen, majority vote
    
    # Single: 51-60
    # 1-58, 2-35, 4-22, 6-20, 8-17, 16-18, 28-21 # cs=1
    # 1-50, 2-26, 4-18, 6-14, 8-13, 16-13, 28-14 # cs=10
    # 1-49, 2-29, 4-18, 6-16, 8-14, 16-13, 28-16 # cs=100
    # 1-48, 2-28, 4-17, 6-15, 8-15, 16-13, 28-16 # cs=1000
    if conf['parallel']:
        pool = Pool(12)
        iterator = pool.imap(f.process_streamcell_star, args, chunksize=10)
    else:
        iterator = map(f.process_streamcell_star, args)
        f.init_global_shmems() # When not in multiprocessing context, we can init the shmems here

    with tqdm.tqdm(range(i_number_of_stream_cells), total=i_number_of_stream_cells, disable=quiet) as pbar:
        for result in iterator:
            pbar.update(1)
            if result is None:
                continue
            if len(result) == 5:
                result = result[0]
                All_COMID_curve_list.append(result[0])
                All_Row_curve_list.append(result[1])
                All_Col_curve_list.append(result[2])
                All_BaseElev_curve_list.append(result[3])
                All_DEM_Elev_curve_list.append(result[3])
                All_QMax_curve_list.append(result[4])
            else:
                ap_row, ap_data, average_curve_data, vdt_row, vdt_data, curve_data, xs_data = result

                # AP comid, row, and col
                if ap_row:
                    comid_ap_dict_list.append(ap_row[0])
                    row_ap_dict_list.append(ap_row[1])
                    col_ap_dict_list.append(ap_row[2])

                # AP discharge, area, and perimeter
                if ap_data:
                    ap_q_list.append(ap_data[0])
                    ap_a_list.append(ap_data[1])
                    ap_p_list.append(ap_data[2])

                # Gather up all the values for the stream cell if we are going to build a reach average curve file
                if average_curve_data:
                    All_COMID_curve_list.append(average_curve_data[0])
                    All_Row_curve_list.append(average_curve_data[1])
                    All_Col_curve_list.append(average_curve_data[2])
                    All_BaseElev_curve_list.append(average_curve_data[3])
                    All_DEM_Elev_curve_list.append(average_curve_data[3])
                    All_QMax_curve_list.append(average_curve_data[4])

                if vdt_row:
                    vdt_list.append(vdt_row)

                # VDT stuff
                if vdt_data:
                    q_list.append(vdt_data[0])
                    v_list.append(vdt_data[1])
                    t_list.append(vdt_data[2])
                    wse_list.append(vdt_data[3])

                if curve_data:
                    COMID_curve_list.append(curve_data[0])
                    Row_curve_list.append(curve_data[1])
                    Col_curve_list.append(curve_data[2])
                    BaseElev_curve_list.append(curve_data[3])
                    DEM_Elev_curve_list.append(curve_data[4])
                    QMax_curve_list.append(curve_data[5])
                    depth_a_curve_list.append(curve_data[6])
                    depth_b_curve_list.append(curve_data[7])
                    tw_a_curve_list.append(curve_data[8])
                    tw_b_curve_list.append(curve_data[9])
                    vel_a_curve_list.append(curve_data[10])
                    vel_b_curve_list.append(curve_data[11])

                if xs_data:
                    XS_COMID_List.append(xs_data[0])
                    XS_Row_List.append(xs_data[1])
                    XS_Col_List.append(xs_data[2])
                    XS_da_xs_profile1_str.append(xs_data[3])
                    XS_da_xs_profile2_str.append(xs_data[4])
                    XS_dm_manning_n_raster1_str.append(xs_data[5])
                    XS_dm_manning_n_raster2_str.append(xs_data[6])
                    XS_d_ordinate_dist.append(xs_data[7])
                    XS_r1.append(xs_data[8])
                    XS_c1.append(xs_data[9])
                    XS_r2.append(xs_data[10])
                    XS_c2.append(xs_data[11])

   
    # Create the output VDT Database file - datatypes are figured out automatically
    colorder = ['COMID', 'Row', 'Col', 'Elev', 'QBaseflow'] + [f"{prefix}_{i}" for i in range(1, i_number_of_increments + 1) for prefix in ['q', 'v', 't', 'wse']]
    vdt_df = (pd.concat([pd.DataFrame(vdt_list, columns=['COMID', 'Row', 'Col', 'Elev', 'QBaseflow']), 
                        pd.DataFrame(q_list, columns=[f'q_{i}' for i in range(1, len(q_list[0]) + 1)]),
                        pd.DataFrame(v_list, columns=[f'v_{i}' for i in range(1, len(v_list[0]) + 1)]),
                        pd.DataFrame(t_list, columns=[f't_{i}' for i in range(1, len(t_list[0]) + 1)]),
                        pd.DataFrame(wse_list, columns=[f'wse_{i}' for i in range(1, len(wse_list[0]) + 1)])], axis=1)
                        .round(3)
                        [colorder] # Reorder columns to match the way Mike had it
                        )
    
    # Remove rows with NaN values
    vdt_df = vdt_df.dropna()
    # # Remove rows where any column has a negative value except wse or elevation
    # Select columns NOT starting with 'wse' or 'Elev'
    cols_to_check = [col for col in vdt_df.columns if (col.startswith('q') or col.startswith('t') or col.startswith('v'))]
    # Remove rows where any of the selected columns have a negative value
    vdt_df = vdt_df.loc[~(vdt_df[cols_to_check] < 0).any(axis=1)]
    if conf['Print_VDT_Database'].endswith('.parquet'):
        vdt_df.to_parquet(conf['Print_VDT_Database'], compression='brotli', index=False) # Brotli does very well with VDT data
    else:
        vdt_df.to_csv(conf['Print_VDT_Database'], index=False)    
    LOG.info(f'Finished writing {conf["Print_VDT_Database"]}')

    # Output the area and wetted perimeter database file if requested
    if conf['Print_AP_Database']:
        # Write the output AP Database file
        colorder = ['COMID', 'Row', 'Col'] + [f"{prefix}_{i}" for i in range(1, i_number_of_increments + 1) for prefix in ['q', 'a', 'p']]
        o_ap_file_df = (pd.concat([pd.DataFrame({'COMID': comid_ap_dict_list, 'Row': row_ap_dict_list, 'Col': col_ap_dict_list}),
                                   pd.DataFrame(ap_q_list, columns=[f'q_{i}' for i in range(1, len(ap_q_list[0]) + 1)]),
                                   pd.DataFrame(ap_a_list, columns=[f'a_{i}' for i in range(1, len(ap_a_list[0]) + 1)]),
                                   pd.DataFrame(ap_p_list, columns=[f'p_{i}' for i in range(1, len(ap_p_list[0]) + 1)])], axis=1)
                                   .round(3)
                                   .dropna()
                                   [colorder]
                                   )
                                
        # # Remove rows where any column has a negative value except wse or elevation
        # Select columns NOT starting with 'wse' or 'Elev'
        cols_to_check = [col for col in o_ap_file_df.columns if (col.startswith('q') or col.startswith('a') or col.startswith('p'))]
        # Remove rows where any of the selected columns have a negative value
        o_ap_file_df = o_ap_file_df.loc[~(o_ap_file_df[cols_to_check] < 0).any(axis=1)]
        if conf['Print_AP_Database'].endswith('.parquet'):
            o_ap_file_df.to_parquet(conf['Print_AP_Database'], compression='brotli', index=False)
        else:
            o_ap_file_df.to_csv(conf['Print_AP_Database'], index=False)
        LOG.info(f'Finished writing {conf["Print_AP_Database"]}')

    # Here we'll generate reach-based coefficients for all stream cells, if the flag is triggered
    if conf['Reach_Average_Curve_File']:
        # Creating a dictionary to map column names to the lists
        data = {
            "COMID": All_COMID_curve_list,
            "Row": All_Row_curve_list,
            "Col": All_Col_curve_list,
            "BaseElev":  All_BaseElev_curve_list,
            "DEM_Elev": All_DEM_Elev_curve_list,
            "QMax": All_QMax_curve_list,
        }

        # Creating the DataFrame
        reach_average_curvefile_df = pd.DataFrame(data)

        # Round
        reach_average_curvefile_df = reach_average_curvefile_df.round(3)

        # Dynamically select columns, starting with prefixes
        q_prefixes = [col for col in vdt_df.columns if col.startswith("q_")]
        t_prefixes = [col for col in vdt_df.columns if col.startswith("t_")]
        v_prefixes = [col for col in vdt_df.columns if col.startswith("v_")]
        wse_prefixes = [col for col in vdt_df.columns if col.startswith("wse_")]

        # Initialize lists to store regression coefficients
        comid_list = []
        d_t_a_list, d_t_b_list = [], []
        d_v_a_list, d_v_b_list = [], []
        d_d_a_list, d_d_b_list = [], []

        # Extract all unique COMID values
        unique_comids = vdt_df["COMID"].unique()
        # unique_comids = o_out_file_df["COMID"].unique()


        # Process each unique COMID
        for comid in unique_comids:
            group: pd.DataFrame = vdt_df[vdt_df["COMID"] == comid]
            # group = o_out_file_df[o_out_file_df["COMID"] == comid]

            
            # Create a MultiIndex from the current group's Row and Col for precise matching
            group_index = pd.MultiIndex.from_arrays([group["Row"].values, group["Col"].values], names=["Row", "Col"])

            # Filter reach_average_curvefile_df using COMID and matching Row-Col pairs
            matching_reach: pd.DataFrame = reach_average_curvefile_df[
                (reach_average_curvefile_df["COMID"] == comid) &
                (pd.MultiIndex.from_frame(reach_average_curvefile_df[["Row", "Col"]]).isin(group_index))
            ]

            matching_reach = matching_reach.drop_duplicates(subset=["Row", "Col", "COMID"])

            if matching_reach.empty:
                LOG.warning(f"No matching BaseElev values found for COMID {comid}. Skipping...")
                continue

            # Get the BaseElev values for subtraction
            base_elev_values = matching_reach.set_index(["Row", "Col"])["BaseElev"]

            # Combine WSE_ values and subtract BaseElev
            depth_combined_values_list = []
            for prefix in wse_prefixes:
                # Match rows using Row and Col from the group
                wse_values = group.set_index(["Row", "Col"])[prefix]
                depth_values = wse_values - base_elev_values
                depth_combined_values_list.extend(depth_values.values)
            d_combined_values = np.array(depth_combined_values_list)

            # Combine Q_ values
            q_combined_values_list = []
            for prefix in q_prefixes:
                q_combined_values_list.extend(group[prefix].values)
            q_combined_values = np.array(q_combined_values_list)

            # Combine T_ values
            t_combined_values_list = []
            for prefix in t_prefixes:
                t_combined_values_list.extend(group[prefix].values)
            t_combined_values = np.array(t_combined_values_list)

            # Combine V_ values
            v_combined_values_list = []
            for prefix in v_prefixes:
                v_combined_values_list.extend(group[prefix].values)
            v_combined_values = np.array(v_combined_values_list)

            # Calculate regression coefficients
            try:
                (d_t_a, d_t_b, d_t_R2) = f.linear_regression_power_function(q_combined_values, t_combined_values, [12, 0.3])
                (d_v_a, d_v_b, d_v_R2) = f.linear_regression_power_function(q_combined_values, v_combined_values, [1, 0.3])
                (d_d_a, d_d_b, d_d_R2) = f.linear_regression_power_function(q_combined_values, d_combined_values, [0.2, 0.5])
            except Exception as e:
                # Handle cases where regression fails (e.g., insufficient data)
                LOG.warning(f"Regression failed for COMID {comid}: {e}")
                d_t_a, d_t_b, d_v_a, d_v_b, d_d_a, d_d_b = np.nan, np.nan, np.nan, np.nan, np.nan, np.nan

            # Append results to lists
            comid_list.append(comid)
            d_t_a_list.append(d_t_a if not np.isnan(d_t_a) else np.nan)
            d_t_b_list.append(d_t_b if not np.isnan(d_t_b) else np.nan)
            d_v_a_list.append(d_v_a if not np.isnan(d_v_a) else np.nan)
            d_v_b_list.append(d_v_b if not np.isnan(d_v_b) else np.nan)
            d_d_a_list.append(d_d_a if not np.isnan(d_d_a) else np.nan)
            d_d_b_list.append(d_d_b if not np.isnan(d_d_b) else np.nan)

        # Create a DataFrame with regression coefficients
        regression_df = pd.DataFrame({
            "COMID": comid_list,
            "depth_a": d_d_a_list,
            "depth_b": d_d_b_list,
            "tw_a": d_t_a_list,
            "tw_b": d_t_b_list,
            "vel_a": d_v_a_list,
            "vel_b": d_v_b_list,
        })
        regression_df = regression_df.round(3)

        # Merge the regression_df into reach_average_curvefile_df based on COMID
        reach_average_curvefile_df = reach_average_curvefile_df.merge(regression_df, on="COMID", how="left")

        # Drop all rows with any NaN values
        reach_average_curvefile_df = reach_average_curvefile_df.dropna()

        # Write the output file
        if conf['Print_Curve_File'].endswith('.parquet'):
            reach_average_curvefile_df.to_parquet(conf['Print_Curve_File'], compression='brotli', index=False)
        else:
            reach_average_curvefile_df.to_csv(conf['Print_Curve_File'], index=False)        
        LOG.info('Finished writing ' + str(conf['Print_Curve_File']))

    elif conf['Print_Curve_File']:
        # Write the output Curve file
        o_curve_file_dict = {'COMID': COMID_curve_list,
                            'Row': Row_curve_list,
                            'Col': Col_curve_list,
                            'BaseElev': BaseElev_curve_list,
                            'DEM_Elev': DEM_Elev_curve_list,
                            'QMax': QMax_curve_list,
                            'depth_a': depth_a_curve_list,
                            'depth_b': depth_b_curve_list,
                            'tw_a': tw_a_curve_list,
                            'tw_b': tw_b_curve_list,
                            'vel_a': vel_a_curve_list,
                            'vel_b': vel_b_curve_list,}
        o_curve_file_df = pd.DataFrame(o_curve_file_dict)
        # Round to 3 decimal places
        o_curve_file_df = o_curve_file_df.round(3)
        # Remove rows with NaN values
        o_curve_file_df = o_curve_file_df.dropna()
        # # Remove rows where any column has negative a coefficient value
        o_curve_file_df = o_curve_file_df.loc[(o_curve_file_df['depth_a'] > 0) & (o_curve_file_df['tw_a'] > 0) & (o_curve_file_df['vel_a'] > 0)]
        if conf['Print_Curve_File'].endswith('.parquet'):
            o_curve_file_df.to_parquet(conf['Print_Curve_File'], compression='brotli', index=False)
        else:
            o_curve_file_df.to_csv(conf['Print_Curve_File'], index=False)            
        LOG.info('Finished writing ' + str(conf['Print_Curve_File']))
    
    # Use the lists to write the output cross-section file if requested
    if conf['XS_Out_File']:
        with open(conf['XS_Out_File'], 'w') as o_xs_file:
            o_xs_file.write('COMID\tRow\tCol\tXS1_Profile\tOrdinate_Dist\tManning_N_Raster1\tXS2_Profile\tOrdinate_Dist\tManning_N_Raster2\tr1\tc1\tr2\tc2\n')
            for i in range(len(XS_COMID_List)):
                o_xs_file.write(f"{XS_COMID_List[i]}\t{XS_Row_List[i]}\t{XS_Col_List[i]}\t{XS_da_xs_profile1_str[i]}\t{XS_d_ordinate_dist[i]}\t{XS_dm_manning_n_raster1_str[i]}\t{XS_da_xs_profile2_str[i]}\t{XS_d_ordinate_dist[i]}\t{XS_dm_manning_n_raster2_str[i]}\t{XS_r1[i]}\t{XS_c1[i]}\t{XS_r2[i]}\t{XS_c2[i]}\n")
        LOG.info('Finished writing ' + str(conf['XS_Out_File']))
    
    
    #write_output_raster('StreamAngles.tif', dm_output_streamangles[i_boundary_number:nrows + i_boundary_number, i_boundary_number:ncols + i_boundary_number], ncols, nrows, dem_geotransform, dem_projection, "GTiff", gdal.GDT_Float32)
    

    # Write the output rasters
    if conf['bathymetry_file']:
        #Make sure all the bathymetry points are above the DEM elevation
        if not conf['Bathy_Use_Banks']:
            dm_output_bathymetry = np.where(dm_output_bathymetry>dm_elevation, np.nan, dm_output_bathymetry)
        # remove the increase in elevation, if negative elevations were present
        if b_modified_dem:
            # Subtract 100 only for cells that are not NaN
            dm_output_bathymetry[~np.isnan(dm_output_bathymetry)] -= 100
        # # Joseph was testing a simple smoothing algorithm here to attempt to reduce variation in the bank based bathmetry (functions but doesn't provide better results)
        # if conf['Bathy_Use_Banks']:
        #     dm_output_bathymetry = smooth_bathymetry_gaussian_numba(dm_output_bathymetry)
        f.write_output_raster(conf['bathymetry_file'], dm_output_bathymetry[i_boundary_number:nrows + i_boundary_number, i_boundary_number:ncols + i_boundary_number], ncols, nrows, dem_geotransform, dem_projection, "GTiff", gdal.GDT_Float32)

    if conf['AROutFLOOD']:
        f.write_output_raster(conf['AROutFLOOD'], dm_out_flood[i_boundary_number:nrows + i_boundary_number, i_boundary_number:ncols + i_boundary_number], ncols, nrows, dem_geotransform, dem_projection, "GTiff", gdal.GDT_Int32)

    for name in ('dem', 'stream', 'land_use', 'land_use_before_streams', 
                 'manning_n', "ia_xc_dr_index_main", "ia_xc_dc_index_main", 
                 "ia_xc_dr_index_second", "ia_xc_dc_index_second", "d_distance_z", 
                 "da_xc_main_fract", "da_xc_second_fract", "da_xc_main_fract_int", 
                 "da_xc_second_fract_int", "bathymetry"):
        f.release_shared_memory(name)

    # Log the compute time
    d_sim_time = datetime.now() - starttime
    i_sim_time_s = int(d_sim_time.seconds)

    if i_sim_time_s < 60:
        LOG.info('Simulation Took ' + str(i_sim_time_s) + ' seconds')
    else:
        LOG.info('Simulation Took ' + str(int(i_sim_time_s / 60)) + ' minutes and ' + str(i_sim_time_s - (int(i_sim_time_s / 60) * 60)) + ' seconds')
