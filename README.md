# ESSD_2025
Codes used in the study "A harmonized 2000–2024 dataset of daily river ice concentration and annual phenology for major Arctic rivers"

The script **DailyMODISRiverIce_CloudRemoved.js** can be executed in Google Earth Engine. Please prepare the shapefile of your river of interest in advance.

After obtaining the daily mapped river ice and exporting/downloading it locally, run **Gridded_RIC_construction.py** to generate the daily 3 km gridded RIC (river ice concentration) product.

Based on the published river ice dataset (including daily RIC, annual freeze-up date, annual breakup date, and annual river ice duration), run **Trend_analysis.py** to calculate and visualize the interannual trends. Here, only the **ice duration trend analysis** example is provided; the same approach applies to other phenological indicators (freeze-up date and breakup date).
The river-ice dataset is available at https://doi.org/10.5281/zenodo.17054619.

