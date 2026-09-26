"""ARC's output files, written as legacy ARC wrote them:

- the VDT and AP databases, from each stream cell's rating curve (arc.outputs.rating_curves),
- the curve file and the reach-average curve file (arc.outputs.curve_files),
- the cross-section file (arc.outputs.cross_sections),
- the representative cross sections (arc.outputs.representative).

The bathymetry and flood rasters are written with arc.io.Raster.write_array.
"""
from arc.outputs.cross_sections import (XS_EXPORT_COLUMNS, CrossSectionRecord, cross_section_dataframe, format_array,
                                        write_cross_sections)
from arc.outputs.curve_files import (curve_file_dataframe, fit_power_law, reach_average_curve_file_dataframe,
                                     write_curve_file, write_reach_average_curve_file)
from arc.outputs.rating_curves import (INCREMENT_FIELDS, METADATA_COLUMNS, RatingCurves, ap_dataframe, vdt_dataframe,
                                       write_ap, write_vdt)
from arc.outputs.representative import (REPRESENTATIVE_CROSS_SECTION_COLUMNS, RepresentativeSample,
                                        representative_cross_section_dataframe, write_representative_cross_sections)

__all__ = [
    "CrossSectionRecord", "INCREMENT_FIELDS", "METADATA_COLUMNS", "REPRESENTATIVE_CROSS_SECTION_COLUMNS",
    "RatingCurves", "RepresentativeSample", "XS_EXPORT_COLUMNS", "ap_dataframe", "cross_section_dataframe",
    "curve_file_dataframe", "fit_power_law", "format_array", "reach_average_curve_file_dataframe",
    "representative_cross_section_dataframe", "vdt_dataframe", "write_ap", "write_cross_sections", "write_curve_file",
    "write_reach_average_curve_file", "write_representative_cross_sections", "write_vdt",
]
