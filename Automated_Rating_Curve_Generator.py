
#Program simply cleans-up rasterized stream networks.

import sys
import os

import gdal

import numpy as np

import math




def Write_Output_Raster(s_output_filename, raster_data, ncols, nrows, dem_geotransform, dem_projection, s_file_format, s_output_type):   
    o_driver = gdal.GetDriverByName(s_file_format)  #Typically will be a GeoTIFF "GTiff"
    #o_metadata = o_driver.GetMetadata()
    
    # Construct the file with the appropriate data shape
    o_output_file = o_driver.Create(s_output_filename, xsize=ncols, ysize=nrows, bands=1, eType=s_output_type)
    
    # Set the geotransform
    o_output_file.SetGeoTransform(dem_geotransform)
    
    # Set the spatial reference
    o_output_file.SetProjection(dem_projection)
    
    # Write the data to the file
    o_output_file.GetRasterBand(1).WriteArray(raster_data)
    
    # Once we're done, close properly the dataset
    o_output_file = None



def Read_Raster_GDAL(InRAST_Name):
    if os.path.isfile(InRAST_Name)==False:
        print('Cannot Find Raster ' + InRAST_Name)
    
    try:
        dataset = gdal.Open(InRAST_Name, gdal.GA_ReadOnly)     
    except RuntimeError:
        sys.exit(" ERROR: Field Raster File cannot be read!")
    # Retrieve dimensions of cell size and cell count then close DEM dataset
    geotransform = dataset.GetGeoTransform()
    # Continue grabbing geospatial information for this use...
    band = dataset.GetRasterBand(1)
    RastArray = band.ReadAsArray()
    #global ncols, nrows, cellsize, yll, yur, xll, xur
    ncols=band.XSize
    nrows=band.YSize
    band = None
    cellsize = geotransform[1]
    yll = geotransform[3] - nrows * np.fabs(geotransform[5])
    yur = geotransform[3]
    xll = geotransform[0];
    xur = xll + (ncols)*geotransform[1]
    lat = np.fabs((yll+yur)/2.0)
    Rast_Projection = dataset.GetProjectionRef()
    dataset = None
    print('Spatial Data for Raster File:')
    print('   ncols = ' + str(ncols))
    print('   nrows = ' + str(nrows))
    print('   cellsize = ' + str(cellsize))
    print('   yll = ' + str(yll))
    print('   yur = ' + str(yur))
    print('   xll = ' + str(xll))
    print('   xur = ' + str(xur))
    return RastArray, ncols, nrows, cellsize, yll, yur, xll, xur, lat, geotransform, Rast_Projection

def GetParamName(lines, numlist, str_find):
    for i in range(numlist):
        ls = lines[i].strip().split('\t')
        if ls[0]==str_find:
            if len(ls[1])>0:
                print('  ' + str_find + ' is set to ' + ls[1])
                return ls[1]
            else:
                print('  ' + str_find + ' is set to 1')
                return 1
    print('  Could not find ' + str_find)
    return ''

def Read_MainInputFile(MIF_Name):
    infile = open(MIF_Name,'r')
    lines = infile.readlines()
    infile.close()
    numlist = len(lines)
    global DEM_Name 
    DEM_Name = GetParamName(lines, numlist, 'DEM_File')
    global STRM_Name 
    STRM_Name = GetParamName(lines, numlist, 'Stream_File')
    global LC_Name 
    LC_Name = GetParamName(lines, numlist, 'LU_Raster_SameRes')
    global Manning_Name 
    Manning_Name = GetParamName(lines, numlist, 'LU_Manning_n')
    global Flow_File_Name
    Flow_File_Name = GetParamName(lines, numlist, 'Flow_File')
    global Flow_File_ID
    Flow_File_ID = GetParamName(lines, numlist, 'Flow_File_ID')
    global Flow_File_BF
    Flow_File_BF = GetParamName(lines, numlist, 'Flow_File_BF')
    global Flow_File_QMax
    Flow_File_QMax = GetParamName(lines, numlist, 'Flow_File_QMax')
    global Spatial_Units
    Spatial_Units = GetParamName(lines, numlist, 'Spatial_Units')
    if Spatial_Units=='':
        Spatial_Units = 'deg'
    global X_Section_Dist
    X_Section_Dist = GetParamName(lines, numlist, 'X_Section_Dist')
    if X_Section_Dist=='':
        X_Section_Dist = 5000.0
    X_Section_Dist = float(X_Section_Dist)
    global Print_VDT_Database
    Print_VDT_Database = GetParamName(lines, numlist, 'Print_VDT_Database')
    global Meta_File
    Meta_File = GetParamName(lines, numlist, 'Meta_File')
    global Degree_Manip
    Degree_Manip = GetParamName(lines, numlist, 'Degree_Manip')
    if Degree_Manip=='':
        Degree_Manip = 1.1
    Degree_Manip = float(Degree_Manip )
    global Degree_Interval
    Degree_Interval = GetParamName(lines, numlist, 'Degree_Interval')
    if Degree_Interval=='':
        Degree_Interval = 1.0
    Degree_Interval = float(Degree_Interval)
    global Low_Spot_Range
    Low_Spot_Range = GetParamName(lines, numlist, 'Low_Spot_Range')
    if Low_Spot_Range=='':
        Low_Spot_Range = 0
    Low_Spot_Range = int(Low_Spot_Range)
    global Gen_Dir_Dist
    Gen_Dir_Dist= GetParamName(lines, numlist, 'Gen_Dir_Dist')
    if Gen_Dir_Dist=='':
        Gen_Dir_Dist = 10
    Gen_Dir_Dist = int(Gen_Dir_Dist)
    global Gen_Slope_Dist
    Gen_Slope_Dist = GetParamName(lines, numlist, 'Gen_Slope_Dist')
    if Gen_Slope_Dist=='':
        Gen_Slope_Dist = 0
    Gen_Slope_Dist = int(Gen_Slope_Dist)
    global Bathy_Trap_H 
    Bathy_Trap_H = GetParamName(lines, numlist, 'Bathy_Trap_H')
    if Bathy_Trap_H=='':
        Bathy_Trap_H = 0.2
    Bathy_Trap_H = float(Bathy_Trap_H)
    global AROutBATHY
    AROutBATHY = GetParamName(lines, numlist, 'AROutBATHY')
    global AROutDEPTH
    AROutDEPTH = GetParamName(lines, numlist, 'AROutDEPTH')
    global AROutFLOOD
    AROutFLOOD = GetParamName(lines, numlist, 'AROutFLOOD')
    
    return


def convert_cell_size(dem_cell_size, dem_lower_left, dem_upper_right):
    """
    Determines the x and y cell sizes based on the geographic location

    Parameters
    ----------
    None. All input data is available in the parent object

    Returns
    -------
    None. All output data is set into the object

    """
    x_cell_size = dem_cell_size
    y_cell_size = dem_cell_size
    projection_conversion_factor = 1

    ### Get the cell size ###
    d_lat = np.fabs((dem_lower_left + dem_upper_right) / 2)

    ### Determine if conversion is needed
    if dem_cell_size > 0.5:
        # This indicates that the DEM is projected, so no need to convert from geographic into projected.
        x_cell_size = dem_cell_size
        y_cell_size = dem_cell_size
        projection_conversion_factor = 1

    else:
        # Reprojection from geographic coordinates is needed
        assert d_lat > 1e-16, "Please use lat and long values greater than or equal to 0."

        # Determine the latitude range for the model
        if d_lat >= 0 and d_lat <= 10:
            d_lat_up = 110.61
            d_lat_down = 110.57
            d_lon_up = 109.64
            d_lon_down = 111.32
            d_lat_base = 0.0

        elif d_lat > 10 and d_lat <= 20:
            d_lat_up = 110.7
            d_lat_down = 110.61
            d_lon_up = 104.64
            d_lon_down = 109.64
            d_lat_base = 10.0

        elif d_lat > 20 and d_lat <= 30:
            d_lat_up = 110.85
            d_lat_down = 110.7
            d_lon_up = 96.49
            d_lon_down = 104.65
            d_lat_base = 20.0

        elif d_lat > 30 and d_lat <= 40:
            d_lat_up = 111.03
            d_lat_down = 110.85
            d_lon_up = 85.39
            d_lon_down = 96.49
            d_lat_base = 30.0

        elif d_lat > 40 and d_lat <= 50:
            d_lat_up = 111.23
            d_lat_down = 111.03
            d_lon_up = 71.70
            d_lon_down = 85.39
            d_lat_base = 40.0

        elif d_lat > 50 and d_lat <= 60:
            d_lat_up = 111.41
            d_lat_down = 111.23
            d_lon_up = 55.80
            d_lon_down = 71.70
            d_lat_base = 50.0

        elif d_lat > 60 and d_lat <= 70:
            d_lat_up = 111.56
            d_lat_down = 111.41
            d_lon_up = 38.19
            d_lon_down = 55.80
            d_lat_base = 60.0

        elif d_lat > 70 and d_lat <= 80:
            d_lat_up = 111.66
            d_lat_down = 111.56
            d_lon_up = 19.39
            d_lon_down = 38.19
            d_lat_base = 70.0

        elif d_lat > 80 and d_lat <= 90:
            d_lat_up = 111.69
            d_lat_down = 111.66
            d_lon_up = 0.0
            d_lon_down = 19.39
            d_lat_base = 80.0

        else:
            raise AttributeError('Please use legitimate (0-90) lat and long values.')

        ## Convert the latitude ##
        d_lat_conv = d_lat_down + (d_lat_up - d_lat_down) * (d_lat - d_lat_base) / 10
        y_cell_size = dem_cell_size * d_lat_conv * 1000.0  # Converts from degrees to m

        ## Longitude Conversion ##
        d_lon_conv = d_lon_down + (d_lon_up - d_lon_down) * (d_lat - d_lat_base) / 10
        x_cell_size = dem_cell_size * d_lon_conv * 1000.0  # Converts from degrees to m

        ## Make sure the values are in bounds ##
        if d_lat_conv < d_lat_down or d_lat_conv > d_lat_up or d_lon_conv < d_lon_up or d_lon_conv > d_lon_down:
            raise ArithmeticError("Problem in conversion from geographic to projected coordinates")

        ## Calculate the conversion factor ##
        projection_conversion_factor = 1000.0 * (d_lat_conv + d_lon_conv) / 2.0
    return x_cell_size, y_cell_size, projection_conversion_factor

def ReadFlowFile(Flow_File_Name, Flow_File_ID, Flow_File_BF, Flow_File_QMax):
    infile = open(Flow_File_Name,'r')
    lines = infile.readlines()
    infile.close()
    numrecords=len(lines)
    COMID = np.zeros(numrecords-1,dtype=int)
    QBaseFlow = np.zeros(numrecords-1,dtype=float)
    QMax = np.zeros(numrecords-1,dtype=float)
    ls = lines[0].strip().split(',')
    fid=0
    bf=0 
    qm=0 
    for i in range(len(ls)):
        if ls[i]==Flow_File_ID:
            fid=i 
        if ls[i]==Flow_File_BF:
            bf=i 
        if ls[i]==Flow_File_QMax:
            qm=i
    
    for i in range(1,numrecords):
        ls = lines[i].strip().split(',')
        COMID[i-1]=int(ls[fid])
        QBaseFlow[i-1]=float(ls[bf])
        QMax[i-1]=float(ls[qm])
    return COMID,QBaseFlow,QMax

def Get_Stream_Slope_Information(RR,CC,E,S,num_strm_cells,dx,dy):
    '''
        1.) Find all stream cells within the Gen_Slope_Dist that have the same stream id value
        2.) Look at the slope of each of the stream cells.
        3.) Average the slopes to get the overall slope we use in the model.
    '''
    strm_slope = np.zeros(num_strm_cells)
    for i in range(num_strm_cells):
        r = RR[i]
        c = CC[i]
        e_cell_of_interest = E[r,c]
        cellvalue = S[r,c]
        S_Box = S[r-Gen_Slope_Dist:r+Gen_Slope_Dist,c-Gen_Slope_Dist:c+Gen_Slope_Dist]
        E_Box = E[r-Gen_Slope_Dist:r+Gen_Slope_Dist,c-Gen_Slope_Dist:c+Gen_Slope_Dist]
        (r_list,c_list) = np.where(S_Box==cellvalue)
        if len(r_list)>0:
            e_list = E_Box[r_list,c_list]
            #The Gen_Slope_Dist is the row/col for the cell of interest within the subsample box
            dz_list = np.sqrt(np.square((r_list - Gen_Slope_Dist)*dy) + np.square((c_list - Gen_Slope_Dist)*dx))  #Distance between the cell of interest and every cell with a similar stream id
            for x in range(len(r_list)):
                if dz_list[x]>0.0:
                    strm_slope[i] = strm_slope[i] + abs(e_cell_of_interest-e_list[x]) / dz_list[x]
            strm_slope[i] = strm_slope[i] / (len(r_list)-1)  #Add the minus one because the cell of interest was in the list
    return strm_slope

def Get_Stream_Direction_Information(RR,CC,S,num_strm_cells,dx,dy):
    '''
        1.) Find all stream cells within the Gen_Dir_Dist that have the same stream id value
        2.) Assume there are 4 quadrants:
                Q3 | Q4      r<0 c<0  |  r<0 c>0
                Q2 | Q1      r>0 c<0  |  r>0 c>0
        3.) Calculate the distance from the cell of interest to each of the stream cells idendified.
        4.) Create a weight that provides a higher weight to the cells that are farther away
        5.) Calculate the Stream Direction based on the Unit Circle inverted around the x axis (this is done because rows increase downward)
        6.) The stream direction needs to be betweeen 0 and pi, so adjust directions between pi and 2pi to be between 0 and pi
    '''
    strm_direction = np.zeros(num_strm_cells)
    xs_direction = np.zeros(num_strm_cells)
    for i in range(num_strm_cells):
        r = RR[i]
        c = CC[i]
        cellvalue = S[r,c]
        S_Box = S[r-Gen_Dir_Dist:r+Gen_Dir_Dist,c-Gen_Dir_Dist:c+Gen_Dir_Dist]
        
        #The Gen_Dir_Dist is the row/col for the cell of interest within the subsample box
        (r_list,c_list) = np.where(S_Box==cellvalue)
        if len(r_list)>0:
            r_list = r_list - Gen_Dir_Dist
            c_list = c_list - Gen_Dir_Dist
            dz_list = np.sqrt(np.square((r_list)*dy) + np.square((c_list)*dx))  #Distance between the cell of interest and every cell with a similar stream id
            dz_list = dz_list / max(dz_list)
            atanvals = np.arctan2(r_list,c_list)
            for x in range(len(r_list)):
                if dz_list[x]>0.0:
                    if atanvals[x]>math.pi:
                        strm_direction[i] = strm_direction[i] + (atanvals[x]-math.pi)*dz_list[x]
                    elif atanvals[x]<0.0:
                        strm_direction[i] = strm_direction[i] + (atanvals[x]+math.pi)*dz_list[x]
                    else:
                        strm_direction[i] = strm_direction[i] + atanvals[x]*dz_list[x]
            strm_direction[i] = strm_direction[i] / sum(dz_list)
            
            #Cross-Section Direction is just perpendicular to the Stream Direction
            xs_direction[i] = strm_direction[i] - math.pi / 2.0
            if xs_direction[i]<0.0:
                xs_direction[i] = xs_direction[i] + math.pi
        #print(str(strm_direction[i]*180/math.pi) + '  ' + str(xs_direction[i]*180/math.pi))
    return strm_direction, xs_direction

def Get_XS_Index_Values(xc_dr_index_main, xc_dc_index_main, xc_dr_index_second, xc_dc_index_second, xc_main_fract, xc_second_fract, xs_direction, r_start, c_start, centerpoint, dx, dy):
    
    #Row Dominant
    if xs_direction>=(math.pi/4) and xs_direction<=(3*math.pi/4):
        for i in range(centerpoint):
            dist_x = i * dy * math.cos(xs_direction)
            x_int = int(dist_x/dx)
            xc_dr_index_main[i] = i 
            xc_dc_index_main[i] = x_int
            sign = 1
            if dist_x < 0:
                sign = -1
            x_int = round((dist_x/dx) + 0.5*sign)
            xc_dr_index_second[i] = i 
            xc_dc_index_second[i] = x_int
            
            ddx = abs(float(dist_x/dx) - x_int)
            xc_main_fract[i] = 1.0 - ddx   #ddx is the distance from the main cell to the location where the line passes through.  Do 1-ddx to get the weight
            xc_second_fract[i] = ddx
        #Distance between each increment
        dist_Z = math.sqrt( (dy * math.cos(xs_direction))*(dy * math.cos(xs_direction)) + dy*dy )
    #Col Dominant
    else:
        for i in range(centerpoint):
            dist_y = i * dx * math.sin(xs_direction)
            y_int = int(dist_y/dy)
            
            xc_dr_index_main[i] = y_int
            xc_dc_index_main[i] = i
            sign = 1
            if dist_y < 0:
                sign = -1
            y_int = round((dist_y/dy) + 0.5*sign)
            xc_dr_index_second[i] = y_int
            xc_dc_index_second[i] = i
            
            ddy = abs(float(dist_y/dy) - y_int)
            xc_main_fract[i] = 1.0 - ddy   #ddy is the distance from the main cell to the location where the line passes through.  Do 1-ddx to get the weight
            xc_second_fract[i] = ddy
        #Distance between each increment
        dist_Z = math.sqrt( (dx * math.sin(xs_direction))*(dx * math.sin(xs_direction)) + dx*dx )
    return dist_Z

def Sample_Cross_Section_From_DEM(XS_Profile, r,c,E,centerpoint, xc_r_index_main, xc_c_index_main, xc_r_index_second, xc_c_index_second, xc_main_fract, xc_second_fract, r_bot, r_top, c_bot, c_top):
    xs_n = centerpoint
    
    #Get the limits of the Cross-section Index
    a = np.where(xc_r_index_main<=r_bot)
    if len(a[0])>0 and int(a[0][0])<xs_n:
        xs_n = int(a[0][0])
    a = np.where(xc_r_index_second<=r_bot)
    if len(a[0])>0 and int(a[0][0])<xs_n:
        xs_n = int(a[0][0])
    a = np.where(xc_r_index_main>=r_top)
    if len(a[0])>0 and int(a[0][0])<xs_n:
        xs_n = int(a[0][0])
    a = np.where(xc_r_index_second>=r_top)
    if len(a[0])>0 and int(a[0][0])<xs_n:
        xs_n = int(a[0][0])
    a = np.where(xc_c_index_main<=c_bot)
    if len(a[0])>0 and int(a[0][0])<xs_n:
        xs_n = int(a[0][0])
    a = np.where(xc_c_index_second>=c_bot)
    if len(a[0])>0 and int(a[0][0])<xs_n:
        xs_n = int(a[0][0])
    a = np.where(xc_c_index_main<=c_top)
    if len(a[0])>0 and int(a[0][0])<xs_n:
        xs_n = int(a[0][0])
    a = np.where(xc_c_index_second>=c_top)
    if len(a[0])>0 and int(a[0][0])<xs_n:
        xs_n = int(a[0][0])
    
    XS_Profile[xs_n] = 99999.9
    XS_Profile[0:xs_n] = E[xc_r_index_main[0:xs_n],xc_c_index_main[0:xs_n]]*xc_main_fract[0:xs_n]  +  E[xc_r_index_second[0:xs_n],xc_c_index_second[0:xs_n]]*xc_second_fract[0:xs_n] 
    return xs_n

def FindBank(XS_Profile, xs_n, y_target):
    for i in range(xs_n):
        if XS_Profile[i]>=y_target:
            return i-1
    return xs_n

def FindDepthOfBathymetry(Q_bf, B, TW, slope, n):
    H = (TW - B) * 0.5
    if slope<0.0002:
        slope = 0.0002
    y_start = 0.0
    dy_list = [1.0,0.5,0.1,0.01]
    
    for dy in dy_list:
        Qcalc = 0.0
        y = y_start
        while(Qcalc<=Q_bf):
            y = y + dy
            A = y * (B + TW) / 2.0
            P = B + 2.0*math.sqrt(H*H + y*y)
            R = A / P
            Qcalc = (1.0/n)*A*math.pow(R,(2/3)) * pow(slope,-0.5)
        y_start = y - dy
    y = y - dy
    A = y * (B + TW) / 2.0
    P = B + 2.0*math.sqrt(H*H + y*y)
    R = A / P
    Qcalc = (1.0/n)*A*math.pow(R,(2/3)) * pow(slope,-0.5)
    #print(str(y) + '  ' + str(Qcalc) + '  ' + str(Q_bf))
    return y

def Adjust_Profile_for_Bathymetry(XS_Profile, bank_index, total_bank_dist, trap_base, dist_Z, H_Dist, y_bathy, y_depth):
    if bank_index>0:
        for x in range(bank_index+1):
            dist_cell_to_bank = (bank_index-x)*dist_Z
            if dist_cell_to_bank<0 or dist_cell_to_bank>total_bank_dist:
                XS_Profile[x] = XS_Profile[x]
            elif dist_cell_to_bank>=H_Dist and dist_cell_to_bank<=(trap_base + H_Dist):
                XS_Profile[x] = y_bathy
            elif dist_cell_to_bank<=H_Dist:
                XS_Profile[x] = y_bathy + y_depth*(1.0-(dist_cell_to_bank / H_Dist))
            elif dist_cell_to_bank>=(trap_base + H_Dist):
                XS_Profile[x] = y_bathy + y_depth * (dist_cell_to_bank-(trap_base + H_Dist)) / H_Dist
    return

def Hypotnuse(x,y):
    return math.sqrt(x*x+y*y)

def Calculate_A_P_R_n_T(XS_Profile, wse, dist_Z, N_Profile):
    #Composite Manning N based on https://www.hec.usace.army.mil/confluence/rasdocs/ras1dtechref/6.5/theoretical-basis-for-one-dimensional-and-two-dimensional-hydrodynamic-calculations/1d-steady-flow-water-surface-profiles/composite-manning-s-n-for-the-main-channel
    A = 0.0
    P = 0.0
    R = 0.0
    T = 0.0
    np = 0.0
    y_dep = wse - XS_Profile
    if y_dep[0]>1e-16:
        for x in range(1,len(y_dep)):
            if y_dep[x]<=0.0:
                dist_use = dist_Z * y_dep[x-1] / (abs(y_dep[x-1]) + abs(y_dep[x]))
                A = A + 0.5*dist_use * y_dep[x-1]
                P = P + Hypotnuse(dist_use,y_dep[x-1])
                R = A / P
                np = np + P * math.pow(N_Profile[x-1],1.5)
                T = T + dist_use
                return A, P, R, np, T
            else:
                A = A + dist_Z * 0.5*( y_dep[x] + y_dep[x-1] )
                P = P + Hypotnuse(dist_Z,(y_dep[x] - y_dep[x-1]))
                R = A / P
                np = np + P * math.pow(N_Profile[x],1.5)
                T = T + dist_Z
    return A, P, R, np, T

def ReadManningTable(Manning_Name, LC):
    infile = open(Manning_Name,'r')
    lines = infile.readlines()
    infile.close()
    num_n = len(lines)
    for n in range(1,num_n):
        ls = lines[n].strip().split('\t')
        LC[LC==int(ls[0])] = float(ls[2])
    return LC

if __name__ == "__main__":
    
    
    #Read Main Input File
    MIF_Name = 'C:/PROGRAMS_Git/automated_rating_curve_generator/TestModel/AutoRoute_InputFiles/AUTOROUTE_INPUT_CO_06759500_Bathymetry.txt'
    Read_MainInputFile(MIF_Name)
    
    #Read the Flow Information
    (COMID,QBaseFlow,QMax) = ReadFlowFile(Flow_File_Name, Flow_File_ID, Flow_File_BF, Flow_File_QMax)
    
    
    #Read Raster Data
    (DEM, dncols, dnrows, dcellsize, dyll, dyur, dxll, dxur, dlat, dem_geotransform, dem_projection) = Read_Raster_GDAL(DEM_Name)
    (STRM, sncols, snrows, scellsize, syll, syur, sxll, sxur, slat, strm_geotransform, strm_projection) = Read_Raster_GDAL(STRM_Name)
    (LC, lncols, lnrows, lcellsize, lyll, lyur, lxll, lxur, llat, land_geotransform, land_projection) = Read_Raster_GDAL(LC_Name )
    if dnrows!=snrows or dnrows!=lnrows:
        print('Rows do not Match!')
    else:
        nrows = dnrows
    if dncols!=sncols or dncols!=lncols:
        print('Cols do not Match!')
    else:
        ncols = dncols
    
    #Imped the Stream and DEM data within a larger Raster to help with the boundary issues.
    boundary_num = max(1,Gen_Slope_Dist,Gen_Dir_Dist)
    S = np.zeros((nrows+boundary_num*2,ncols+boundary_num*2))
    S[boundary_num:(nrows+boundary_num), boundary_num:(ncols+boundary_num)] = STRM
    E = np.zeros((nrows+boundary_num*2,ncols+boundary_num*2))
    E[boundary_num:(nrows+boundary_num), boundary_num:(ncols+boundary_num)] = DEM
    L = np.zeros((nrows+boundary_num*2,ncols+boundary_num*2))
    L[boundary_num:(nrows+boundary_num), boundary_num:(ncols+boundary_num)] = LC
    
    #Read in the Manning Table
    ManningNRaster = ReadManningTable(Manning_Name, L)
    #print(np.unique(ManningNRaster))
    
    ep = 1000
    # A1 = np.zeros(ep, dtype=float)
    # P1 = np.zeros(ep, dtype=float)
    # R1 = np.zeros(ep, dtype=float)
    # np1 = np.zeros(ep, dtype=float)
    # T1 = np.zeros(ep, dtype=float)
    # A2 = np.zeros(ep, dtype=float)
    # P2 = np.zeros(ep, dtype=float)
    # R2 = np.zeros(ep, dtype=float)
    # np2 = np.zeros(ep, dtype=float)
    # T2 = np.zeros(ep, dtype=float)
    
    TotalT = np.zeros(ep, dtype=float)
    TotalA = np.zeros(ep, dtype=float)
    TotalP = np.zeros(ep, dtype=float)
    TotalV = np.zeros(ep, dtype=float)
    TotalQ = np.zeros(ep, dtype=float)
    
    
    
    
    #Get the list of stream locations
    (RR,CC) = S.nonzero()
    num_strm_cells = len(RR)
    
    
    #Get the cell dx and dy coordinates
    (dx,dy, dproject) = convert_cell_size(dcellsize, dyll, dyur)
    print('Cellsize X = ' + str(dx))
    print('Cellsize Y = ' + str(dy))
    
    #Get the Slope of each Stream Cell.  Slope should be in m/m
    strm_slope = Get_Stream_Slope_Information(RR,CC,E,S,num_strm_cells,dx,dy)
    
    #Get the Stream Direction of each Stream Cell.  Direction is between 0 and pi.  Also get the cross-section direction (also between 0 and pi)
    (strm_direction, xs_direction) = Get_Stream_Direction_Information(RR,CC,S,num_strm_cells,dx,dy)
    
    #Pull Cross-Sections
    centerpoint = int((X_Section_Dist/(sum([dx,dy])*0.5))/2.0)+1
    XS_Profile1 = np.zeros(centerpoint+1)
    XS_Profile2 = np.zeros(centerpoint+1)
    xc_dr_index_main = np.zeros(centerpoint+1, dtype=int)  #Only need to go to centerpoint, because the other side of xs we can just use *-1
    xc_dc_index_main = np.zeros(centerpoint+1, dtype=int)  #Only need to go to centerpoint, because the other side of xs we can just use *-1
    xc_dr_index_second = np.zeros(centerpoint+1, dtype=int)  #Only need to go to centerpoint, because the other side of xs we can just use *-1
    xc_dc_index_second = np.zeros(centerpoint+1, dtype=int)  #Only need to go to centerpoint, because the other side of xs we can just use *-1
    xc_main_fract = np.zeros(centerpoint+1)
    xc_second_fract = np.zeros(centerpoint+1)
    
    r_bot = centerpoint
    r_top = nrows + centerpoint
    c_bot = centerpoint
    c_top = ncols + centerpoint
    
    for i in range(num_strm_cells):
        r=RR[i]
        c=CC[i]
        
        slope_use = strm_slope[i]
        if slope_use<0.0002:
            slope_use = 0.0002
        
        #Get the Flow Rates Associated with the Stream Cell
        flow_index = np.where(COMID==int(S[r,c]))
        flow_index = int(flow_index[0][0])
        Q_bf = QBaseFlow[flow_index] 
        Q_max = QMax[flow_index] 
    
        
        dist_Z = Get_XS_Index_Values(xc_dr_index_main, xc_dc_index_main, xc_dr_index_second, xc_dc_index_second, xc_main_fract, xc_second_fract, xs_direction[i], r, c, centerpoint, dx, dy)
        
        xc_r1_index_main = r+xc_dr_index_main
        xc_r2_index_main = r+xc_dr_index_main*-1
        xc_c1_index_main = c+xc_dc_index_main
        xc_c2_index_main = c+xc_dc_index_main*-1
        (xs1_n) = Sample_Cross_Section_From_DEM(XS_Profile1, r,c,E,centerpoint, xc_r1_index_main, xc_c1_index_main, r+xc_dr_index_second, c+xc_dc_index_second, xc_main_fract, xc_second_fract, r_bot, r_top, c_bot, c_top)
        (xs2_n) = Sample_Cross_Section_From_DEM(XS_Profile2, r,c,E,centerpoint, xc_r2_index_main, xc_c2_index_main, r+xc_dr_index_second*-1, c+xc_dc_index_second*-1, xc_main_fract, xc_second_fract, r_bot, r_top, c_bot, c_top)
        
        
        #Adjust to the lowest-point in the Cross-Section
        lpi=0
        low_point_elev = XS_Profile1[0]
        if Low_Spot_Range>0:
            for x in range(Low_Spot_Range):
                if XS_Profile1[x]<low_point_elev:
                    low_point_elev = XS_Profile1[x]
                    lpi = x
                if XS_Profile2[x]<low_point_elev:
                    low_point_elev = XS_Profile2[x]
                    lpi = -1*x
            #print('LowPoint= ' + str(lpi) + '  ' + str(low_point_elev))
            if lpi>0:
                XS_Profile2[lpi:centerpoint] = XS_Profile2[0:centerpoint-lpi]
                XS_Profile2[0:lpi+1] = np.flip(XS_Profile1[0:lpi+1])
                XS_Profile1[0:centerpoint-lpi] = XS_Profile1[lpi:centerpoint]
                xs1_n = xs1_n - lpi
                xs2_n = xs2_n + lpi
                XS_Profile1[xs1_n]=99999.9
                
                xc_r2_index_main[lpi:centerpoint] = xc_r2_index_main[0:centerpoint-lpi]
                xc_r2_index_main[0:lpi+1] = np.flip(xc_r1_index_main[0:lpi+1])
                xc_r1_index_main[0:centerpoint-lpi] = xc_r1_index_main[lpi:centerpoint]
                
                xc_c2_index_main[lpi:centerpoint] = xc_c2_index_main[0:centerpoint-lpi]
                xc_c2_index_main[0:lpi+1] = np.flip(xc_c1_index_main[0:lpi+1])
                xc_c1_index_main[0:centerpoint-lpi] = xc_c1_index_main[lpi:centerpoint]
                
            elif lpi<0:
                lpi = lpi *-1
                XS_Profile1[lpi:centerpoint] = XS_Profile1[0:centerpoint-lpi]
                XS_Profile1[0:lpi+1] = np.flip(XS_Profile2[0:lpi+1])
                XS_Profile2[0:centerpoint-lpi] = XS_Profile2[lpi:centerpoint]
                xs2_n = xs2_n - lpi
                xs1_n = xs1_n + lpi
                XS_Profile2[xs2_n]=99999.9
                
                xc_r1_index_main[lpi:centerpoint] = xc_r1_index_main[0:centerpoint-lpi]
                xc_r1_index_main[0:lpi+1] = np.flip(xc_r2_index_main[0:lpi+1])
                xc_r2_index_main[0:centerpoint-lpi] = xc_r2_index_main[lpi:centerpoint]
                
                xc_c1_index_main[lpi:centerpoint] = xc_c1_index_main[0:centerpoint-lpi]
                xc_c1_index_main[0:lpi+1] = np.flip(xc_c2_index_main[0:lpi+1])
                xc_c2_index_main[0:centerpoint-lpi] = xc_c2_index_main[lpi:centerpoint]
        #print(XS_Profile1)
        #print(XS_Profile2)
        #print(XS_Profile1[xs1_n])
        #print(XS_Profile2[xs2_n])
        
        bank_1_index = FindBank(XS_Profile1, xs1_n, low_point_elev+0.1)
        bank_2_index = FindBank(XS_Profile2, xs2_n, low_point_elev+0.1)
        total_bank_cells = bank_1_index + bank_2_index - 1
        if total_bank_cells<1:
            total_bank_cells = 1
        total_bank_dist = total_bank_cells * dist_Z
        H_Dist = Bathy_Trap_H * total_bank_dist
        trap_base = total_bank_dist - 2.0 * H_Dist
        y_depth = FindDepthOfBathymetry(Q_bf, trap_base, total_bank_dist, strm_slope[i], 0.03)
        y_bathy = XS_Profile1[0] - y_depth
        XS_Profile1[0] = y_bathy
        XS_Profile2[0] = y_bathy
        if total_bank_cells>1:
            Adjust_Profile_for_Bathymetry(XS_Profile1, bank_1_index, total_bank_dist, trap_base, dist_Z, H_Dist, y_bathy, y_depth)
            Adjust_Profile_for_Bathymetry(XS_Profile2, bank_2_index, total_bank_dist, trap_base, dist_Z, H_Dist, y_bathy, y_depth)
        
        
        #Get a list of Elevations that we need to evaluate
        ElevList_mm = np.unique(np.concatenate((XS_Profile1[0:xs1_n]*1000,XS_Profile1[0:xs2_n]*1000)).astype(int))
        ElevList_mm = ElevList_mm[np.logical_and(ElevList_mm[:]>0,ElevList_mm[:]<99999900)]
        num_elevs = len(ElevList_mm)
        if num_elevs>=ep:
            print('ERROR, HAVE TOO MANY ELEVATIONS TO EVALUATE')
        
        
        for e in range(1,num_elevs):
            wse = ElevList_mm[e] / 1000.0
            (A1, P1, R1, np1, T1) = Calculate_A_P_R_n_T(XS_Profile1[0:xs1_n], wse, dist_Z, ManningNRaster[xc_r1_index_main[0:xs1_n],xc_c1_index_main[0:xs1_n]])
            #print(str(A1[e]) + '  ' +  str(P1[e]) + '  ' +  str(R1[e]) + '  ' +  str(n1[e]) + '  ' +  str(T1[e]))
            (A2, P2, R2, np2, T2) = Calculate_A_P_R_n_T(XS_Profile2[0:xs2_n], wse, dist_Z, ManningNRaster[xc_r2_index_main[0:xs2_n],xc_c2_index_main[0:xs2_n]])
            #print(str(A2[e]) + '  ' +  str(P2[e]) + '  ' +  str(R2[e]) + '  ' +  str(n2[e]) + '  ' +  str(T2[e]))
            
            TotalT[e] = T1 + T2
            TotalA[e] = A1 + A2
            TotalP[e] = P1 + P2
            if TotalT[e]<=0.0 or TotalA[e]<=0.0 or TotalP[e]<=0.0:
                TotalT[e] = 0.0
                TotalA[e] = 0.0
                TotalP[e] = 0.0
                TotalQ[e] = 0.0
                TotalV[e] = 0.0
            else:
                composite_n = math.pow(((np1+np2)/TotalP[e]),(2/3))
                if composite_n<0.0001:
                    composite_n = 0.035
                TotalQ[e] = (1/composite_n) * TotalA[e] * math.pow((TotalA[e]/TotalP[e]),(2/3)) * math.pow(slope_use,-0.5)
                TotalV[e] = TotalQ[e] / TotalA[e]
            #print(str(i) + '  ' + str(e) + '  ' + str(TotalQ[e]))
    
    #Write Output Data
    #Write_Output_Raster(STRM_File_Clean, B[1:nrows+1,1:ncols+1], ncols, nrows, dem_geotransform, dem_projection, "GTiff", gdal.GDT_Int32)
    