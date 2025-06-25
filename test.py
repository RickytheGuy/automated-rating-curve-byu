import os
# os.environ['NUMBA_DISABLE_JIT'] = '1'  # Disable JIT compilation for debugging purposes


from src.arc.Automated_Rating_Curve_Generator import main

# main(r"C:\Users\lrr43\Documents\alos_test_area\alos_-100_32\arc_inputs\main_input.txt", None, False)
main("", args={
"DEM_File":	r"C:\Users\lrr43\Downloads\USGS_13_n39w078_20220524.tif",
"Stream_File":	r"C:\Users\lrr43\Downloads\152_STRM_Raster_Clean.tif",
"LU_Raster_SameRes":	r"C:\Users\lrr43\Downloads\152_LAND_Raster.tif",
"LU_Manning_n":	r"C:\Users\lrr43\floodmap-evaluation\ESA_land_use.txt",
"Flow_File":	r"C:\Users\lrr43\Downloads\152_Reanalysis.csv",
"Flow_File_ID":	"COMID",
"Flow_File_BF":	"known_baseflow",
"Flow_File_QMax":	"rp100_premium",
"Spatial_Units":	"deg",
"X_Section_Dist":	6000,
"Degree_Manip":	6.1,
"Degree_Interval":	1.5,
"Low_Spot_Range":	2,
"Str_Limit_Val":	1,
"Gen_Dir_Dist":	10,
"Gen_Slope_Dist":	10,

#VDT_Output_File_and_CurveFile
"VDT_Database_NumIterations":	30,
"VDT_Database_File":	r"C:\Users\lrr43\Downloads\152_VDT_Database.txt",
"Print_VDT_Database":	r"C:\Users\lrr43\Downloads\152_VDT_Database.txt",
"Print_Curve_File":	r"C:\Users\lrr43\Downloads\152_CurveFile.csv",
"Reach_Average_Curve_File":	False,

#Bathymetry_Information
"Bathy_Trap_H":	0.20,
"Bathy_Use_Banks":	False,
"FindBanksBasedOnLandCover":	True,
"AROutBATHY":	r"C:\Users\lrr43\Downloads\152_ARC_Bathy.tif",
"BATHY_Out_File":	r"C:\Users\lrr43\Downloads\152_ARC_Bathy.tif",
"XS_Out_File":	r"C:\Users\lrr43\Downloads\152_XS_Out.txt",
}, quiet=False)