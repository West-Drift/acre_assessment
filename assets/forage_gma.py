# --- Raw Cell [0] ---
# **Automated Edible Vegetation Masking and UAI Derivation**
# ----------------------------------------------------------
# 
# This notebook implements a reproducible geospatial workflow as part of the GIS/Remote Sensing practical assignment.
# 
# **Objective:** To derive an edible vegetation mask and segment the landscape into Unit Areas of Insurance (UAIs) using NDVI and supporting datasets. This supports forage indexing in Game Management Areas (GMAs) from 2020 to 2024.
# 
# **Core Tasks:**
# - Preprocess MODIS NDVI (2020–2024) and ESA WorldCover to mask non-vegetated and non-edible zones,
# - Classify vegetation into four NDVI-based classes (e.g. cropland, shrubland, wetland, evergreen forest),
# - Segment the AOI into homogeneous NDVI zones (intervals of 0.2) for initial UAI definition,
# - Visualize monthly NDVI time series for UAIs,
# - Extend the workflow to integrate Leaf Area Index (LAI), soil moisture, and precipitation using Multi-Criteria Decision Analysis (MCDA),
# - Highlight spatial discrepancies between NDVI-only and multi-variable UAI zones.
# 
# **Note**
# Each section provides step-by-step documentation of the method, dataset choices, parameter thresholds, and processing logic.
# The workflow is fully automated and designed to run entirely within the coding environment, with minimal to no reliance on external GIS software.

# --- Code Cell [1] ---
# Install necessary libraries
!pip install geopandas rasterio gdal earthengine-api geemap matplotlib numpy folium scikit-learn requests twilio bs4 zipfile36 shapely ipywidgets ipyleaflet


# --- Raw Cell [2] ---
# # ============================================================
# # 1.1 Load Area of Interest (AOI): Game Management Areas (GMAs)
# # ============================================================
# 
# In this step, we load the AOI shapefile (Game Management Areas) from Google Earth Engine.  
# Each GMA will be processed separately to allow site-specific edible vegetation and forage scarcity analysis.
# 
# **Approach:**
# - Load GMA boundaries as a `FeatureCollection`.
# - Filter the AOI by name or index to analyze one GMA at a time.
# - Export the AOI geometry for shapefile output and downstream spatial joins.
# 
# **Note:** This modular loading method supports batch processing of multiple conservancies in future use cases.

# --- Code Cell [3] ---
# Import necessary libraries
import ee
import geemap
import os

# Authenticate Earth Engine access
try:
    ee.Initialize()
except Exception as e:
        print(f"Earth Engine initialization failed: {e}")
        print("Attempting to authenticate...")
        ee.Authenticate()
        ee.Initialize()

# Create interactive map object
Map = geemap.Map()

# Load GMA FeatureCollection
gmas = ee.FeatureCollection("projects/ee-ongetewesly1/assets/bangweulu_gma")

# Inspect attribute table to get correct field for geometries within the collection
# Get the first feature in the collection
first_feature = gmas.first()

# Get the list of property keys
property_keys = first_feature.propertyNames().getInfo()

print("Available attribute fields:", property_keys)

# Get all unique 'Name' attributes
aoi_names = gmas.aggregate_array('NAME').getInfo() # use correct attribute field to fetch geometries

# Use an index to select an AOI (change `index` to process different AOIs)
index = 0  # Adjust this index for the AOI you want to process
gma_name = aoi_names[index]
gma = gmas.filter(ee.Filter.eq('NAME', gma_name))

# Visualize AOI on the map
Map.centerObject(gma, 9)
Map.addLayer(gma.style(**{'color': 'black', 'fillColor': '00000000'}), {}, gma_name)

print(f"AOI '{gma_name}' loaded and added to the map.")

# Export AOI as a shapefile to Google Drive
task = ee.batch.Export.table.toDrive(
    collection=gma,
    description=f"{gma_name}",
    fileFormat='SHP',
    folder='AOI'
)
task.start()

print(f"Export task for AOI '{gma_name}' started.")
Map


# --- Raw Cell [4] ---
# # ====================================
# # 1.2 Load MODIS NDVI Data (2020–2024)
# # ====================================
# 
# This section loads the **MODIS MOD13Q1** NDVI dataset from Google Earth Engine, filtering it for the AOI and the required period (2020–2024).
# 
# Approach:
# - NDVI (Normalized Difference Vegetation Index) measures vegetation health and availability.
# - Seasonal and long-term trends in NDVI allow us to monitor pasture conditions.
# - We prepare the data by scaling NDVI values (originally stored as integers ×10,000) into real units (0 to 1).
# 
# Coding Note:
# - `.multiply(0.0001)` rescales NDVI.
# - `.filterDate()` and `.filterBounds()` ensure that only the data over our AOI and time range is used.

# --- Code Cell [5] ---
# Define a function to scale NDVI values
def scale_ndvi(img):
    return img.multiply(0.0001).rename('NDVI').copyProperties(img, ['system:time_start'])

# Load, filter, and scale MODIS NDVI ImageCollection
modisNDVI = ee.ImageCollection('MODIS/006/MOD13Q1') \
    .select('NDVI') \
    .filterDate('2020-01-01', '2024-12-31') \
    .filterBounds(gma) \
    .map(scale_ndvi)

# Display a sample NDVI image clipped to the AOI
firstNDVI = modisNDVI.first()
Map.addLayer(firstNDVI.clip(gma), {'min': 0, 'max': 1, 'palette': ['white', 'green']}, 'Sample NDVI (First Image)')
print("MODIS NDVI loaded and sample image added to the map.")
# Map

# --- Raw Cell [6] ---
# ============================================================================
# # 1.3 Load Sentinel-2 Mosaic and ESA WorldCover 2020/2021 for Land Cover Data
# ============================================================================
# 
# In this step, we load Sentinel-2 imagery and ESA WorldCover land cover products to help mask out non-edible vegetation areas like built-up zones, water, and evergreen forests.
# 
# Approach:
# - Sentinel-2 provides detailed reflectance imagery used for land cover classification.
# - ESA WorldCover (2020 and 2021) helps identify **stable areas** where land cover hasn't changed, increasing reliability of classification.
# - Only areas with consistent land cover are used for training the classifier.
# 
# Coding Note:
# - `.eq()` compares ESA 2020 and 2021 maps pixel-by-pixel to find unchanged areas.
# - `.updateMask()` keeps only stable pixels.

# --- Code Cell [7] ---
# Load Sentinel-2 Harmonized Collection
sentinelCollection = ee.ImageCollection('COPERNICUS/S2_SR_HARMONIZED') \
    .filterBounds(gma) \
    .filterDate('2024-08-01', '2025-04-25') \
    .filter(ee.Filter.lte('CLOUDY_PIXEL_PERCENTAGE', 1))

# Mosaic the Sentinel-2 images
sentinelMosaic = sentinelCollection.median().clip(gma)

# Load ESA WorldCover 2020 and 2021
esa2020 = ee.Image('ESA/WorldCover/v100/2020').select('Map').clip(gma)
esa2021 = ee.Image('ESA/WorldCover/v200/2021').select('Map').clip(gma)

# Identify stable land cover areas (unchanged between 2020 and 2021)
stableLC = esa2020.eq(esa2021)

# Mask ESA 2021 to keep only stable pixels
esa_mask = esa2021.updateMask(stableLC).rename('lc')

# Generate stable training points using stratified sampling
samplePoints = esa_mask.stratifiedSample(
  numPoints=500,
  classBand='lc',
  region=gma.geometry(),
  scale=10,
  tileScale=8,
  geometries=True
)

# Define bands for classification
bands = ['B2', 'B3', 'B4', 'B8', 'B11', 'B12']

# Sample Sentinel-2 reflectance values at training point locations
trainingData = sentinelMosaic.select(bands).sampleRegions(
  collection=samplePoints,
  properties=['lc'],
  scale=10,
  tileScale=8
)

# Train Classifier and Classify
classifier = ee.Classifier.smileRandomForest(50).train(
  features=trainingData,
  classProperty='lc',
  inputProperties=bands
)

land_cover = sentinelMosaic.select(bands).classify(classifier).rename('lc')

# Visualize Classified Image
landCoverLegend = [
  {'value': 10, 'name': 'Tree cover', 'color': '006400'},
  {'value': 20, 'name': 'Shrubland', 'color': 'ffbb22'},
  {'value': 30, 'name': 'Grassland', 'color': 'ffff4c'},
  {'value': 40, 'name': 'Cropland', 'color': 'f096ff'},
  {'value': 50, 'name': 'Built-up', 'color': 'fa0000'},
  {'value': 60, 'name': 'Bare / sparse vegetation', 'color': 'b4b4b4'},
  {'value': 70, 'name': 'Snow and ice', 'color': 'f0f0f0'},
  {'value': 80, 'name': 'Permanent water bodies', 'color': '003df5'},
  {'value': 90, 'name': 'Herbaceous wetland', 'color': '0096a0'},
  {'value': 95, 'name': 'Mangroves', 'color': '00cf75'},
  {'value': 100, 'name': 'Moss and lichen', 'color': 'fae6a0'}
]

palette = [d['color'] for d in landCoverLegend]

# Optional geemap visualization
Map.addLayer(land_cover, {'min': 10, 'max': 100, 'palette': palette}, 'Land Cover')
# Map

# --- Raw Cell [8] ---
# ==================================================
# # 1.4 Calculate NDVI Amplitude and Mean (2020–2024)
# ==================================================
# 
# This step computes the **NDVI amplitude** (difference between maximum and minimum NDVI) and the **NDVI mean** across 2020–2024 for the AOI.
# 
# Approach:
# - NDVI Amplitude shows how much vegetation "fluctuates" seasonally — high amplitude = seasonal plants (grazing).
# - NDVI Mean represents overall vegetation productivity — low mean = degraded or dry zones.
# 
# These two indicators are critical:
# - **High seasonal fluctuation + healthy average** suggests edible grazing zones.
# - **Low fluctuation** or **high mean without fluctuation** suggests permanent (non-edible) vegetation.
# 
# Coding Note:
# - `subtract()` calculates pixel-wise differences.
# - `mean()` computes the temporal average across years.

# --- Code Cell [9] ---
# Define the range of years
years = ee.List.sequence(2020, 2024)

# Function to calculate yearly NDVI amplitude
def calculate_yearly_amplitude(year):
    ndvi = modisNDVI.filter(ee.Filter.calendarRange(year, year, 'year'))
    max_ndvi = ndvi.max()
    min_ndvi = ndvi.min()
    return max_ndvi.subtract(min_ndvi).set('year', year)

# Apply the function across all years
ndviYearly = ee.ImageCollection(years.map(calculate_yearly_amplitude))

# Calculate Mean NDVI Amplitude over the entire period
ndviAmpMean = ndviYearly.mean().rename('NDVI_amp')

# Calculate Mean NDVI over the entire period
ndviMean = modisNDVI.mean().rename('NDVI_mean')


# --- Optional: Print summary stats (can be slow due to getInfo()) ---

print("Calculating NDVI Amplitude Min/Max...")
ndviAmpStats = ndviAmpMean.reduceRegion(
     reducer=ee.Reducer.minMax(),
     geometry=gma.geometry(),
     scale=250, # Use appropriate scale
     maxPixels=1e13
 ).getInfo()
print('NDVI Amplitude Min and Max:', ndviAmpStats)



print("\nCalculating NDVI Mean Min/Max...")
ndviMeanStats = ndviMean.reduceRegion(
     reducer=ee.Reducer.minMax(),
     geometry=gma.geometry(),
     scale=250, # Use appropriate scale
     maxPixels=1e13
 ).getInfo()
print('NDVI Mean Min and Max:', ndviMeanStats)
# --- End Optional Stats ---



# Visualize results on map, min and max based on computed stats 
Map.addLayer(ndviAmpMean.clip(gma), {
    'min': 0.13, 
    'max': 0.84, 
    'palette': ['white', 'yellow', 'orange', 'red']
},  'Mean NDVI Amplitude (2020–2024)')

Map.addLayer(ndviMean.clip(gma), { 
    'min': 0.15, 
    'max': 0.83, 
    'palette': ['brown', 'lightgreen', 'green']
},  'Mean NDVI (2020–2024)')

print("NDVI Amplitude and Mean calculated and added to the map.")

# --- Raw Cell [10] ---
# ============================
# # 1.5 Create Land Cover Masks
# ============================
# 
# In this step, we generate masks for major **non-edible vegetation types** using classified land cover and NDVI information:
# - Water bodies (e.g., rivers, lakes)
# - Human activity areas (cropland, built-up areas, bare land)
# - Evergreen forests (high NDVI mean, low amplitude)
# - Wetlands
# 
# Approach:
# - Masking these areas ensures that only **temporary, seasonally edible** vegetation remains for further analysis.
# 
# Coding Note:
# - `.eq()` creates pixel-wise binary masks.
# - `.Or()` combines multiple land cover classes into one mask (e.g., human activity).
# - `.And()` and `.lt()` logic defines evergreen vegetation areas.

# --- Code Cell [11] ---
# Water bodies mask (class 80 from ESA WorldCover)
water = land_cover.eq(80).rename('Water')

# Human activity mask (cropland, built-up, bare land)
human = land_cover.eq(40) \
    .Or(land_cover.eq(50)) \
    .Or(land_cover.eq(60)) \
    .rename('Human')

# Evergreen vegetation mask (low NDVI fluctuation + high NDVI mean)
evergreen = ndviAmpMean.lt(0.25) \
    .And(ndviMean.gt(0.5)) \
    .clip(gma) \
    .rename('Evergreen')

# Wetlands mask (class 90 from ESA WorldCover)
wetlands = land_cover.eq(90).rename('Wetlands')

# Visualize all masks
Map.addLayer(water.selfMask(), {'palette': 'blue'}, 'Water Bodies')
Map.addLayer(human.selfMask(), {'palette': 'red'}, 'Human Activity')
Map.addLayer(evergreen.selfMask(), {'palette': 'darkgreen'}, 'Evergreen Vegetation')
Map.addLayer(wetlands.selfMask(), {'palette': 'cyan'}, 'Wetlands')

print("Land cover masks created and visualized.")


# Optional: Print Pixel Counts for Verification (can be slow)

print("\nCalculating Pixel Counts...")
masks = ee.Image.cat([water, human, evergreen, wetlands])
maskStats = masks.reduceRegion(
    reducer=ee.Reducer.sum(),
    geometry=gma.geometry(),
    scale=10,
    maxPixels=1e13
).getInfo()

print('Pixel Counts for Each Land Cover Mask:', maskStats)
# --- End Optional Stats ---


# --- Raw Cell [12] ---
# ===================================================
# # 1.6 Create Edible and Non-Edible Vegetation Layers
# ===================================================
# 
# This step uses the previously created masks (water, human, evergreen, wetlands) to define **edible** and **non-edible** vegetation zones.
# 
# Approach:
# - **Exclusion Zones**: Areas unsuitable for grazing (like water, settlements, evergreen forests) are first combined into an 'exclusion' mask.
# - **Edible Vegetation**: Areas outside the exclusion zones, having seasonal NDVI fluctuations.
# - **Non-Edible Vegetation**: Areas inside exclusion zones.
# 
# Coding Note:
# - `.Or()` combines masks together.
# - `.Not()` reverses the exclusion to get edible areas.
# - `.updateMask()` applies the logical mask to the NDVI Amplitude layer.

# --- Code Cell [13] ---
# Combine exclusion masks: water, human activity, evergreen, wetlands
exclusion = water.Or(human).Or(evergreen).Or(wetlands)

# Non-edible vegetation: inside exclusion zones
non_edible = ndviAmpMean.updateMask(exclusion).clip(gma).rename('Non-Edible')

# Edible vegetation: outside exclusion zones
edibleMask = exclusion.Not()
edible_vegetation = ndviAmpMean.updateMask(edibleMask).clip(gma).rename('Edible')


# --- Optional: Print statistics to guide visualization (can be slow) ---
# Edible Vegetation Stats
print("Calculating Edible Vegetation Stats...")
edibleStats = edible_vegetation.reduceRegion(
     reducer=ee.Reducer.minMax(),
     geometry=gma.geometry(),
     scale=250, # Scale matching NDVI amplitude calculation
     maxPixels=1e13
 ).getInfo()
print('Edible Vegetation NDVI Amplitude Stats (min/max):', edibleStats)

# Non-Edible Vegetation Stats
print("Calculating Non-Edible Vegetation Stats...")
non_edibleStats = non_edible.reduceRegion(
     reducer=ee.Reducer.minMax(),
     geometry=gma.geometry(),
     scale=250, # Scale matching NDVI amplitude calculation
     maxPixels=1e13
 ).getInfo()
print('Non-Edible Vegetation NDVI Amplitude Stats (min/max):', non_edibleStats)
# --- End Optional Stats ---


# Visualize edible vegetation, min and max based on computed stats
Map.addLayer(edible_vegetation, {
    'min': 0.19,
    'max': 0.84,
    'palette': ['white', 'yellow', 'orange', 'red']
}, 'Edible Vegetation')

# Visualize non-edible vegetation
Map.addLayer(non_edible, {
    'min': 0.13,
    'max': 0.73,
    'palette': ['gray', 'brown', 'darkred']
}, 'Non-Edible Vegetation')

print("Edible and Non-Edible vegetation layers created and visualized.")

# --- Raw Cell [14] ---
# ============================================================
# # 1.7.0 Cluster Edible Vegetation Areas using SNIC Segmentation
# ============================================================
# 
# In this step, we apply **SNIC segmentation** to edible vegetation to form **homogeneous grazing zones** based on landscape structure and NDVI similarity.
# 
# Approach:
# - SNIC (Simple Non-Iterative Clustering) groups neighboring pixels into spatially coherent clusters.
# - This creates meaningful ecological regions that can be used for insurance claims processing.
# 
# Coding Note:
# - `ee.Algorithms.Image.Segmentation.SNIC()` performs clustering.
# - `reduceConnectedComponents()` calculates the mean NDVI per cluster.
# - Clusters are later vectorized (converted into shapefiles).
# - Positional arguements:
#   *size - Specifies the approximate size of clusters (in pixels) that SNIC will produce.
#   *compactness - Controls the balance between spatial and spectral clustering.
#   *connectivity - Refers to the pixel connectivity used for clustering
#   *neighborhoodsize - Defines the size of the neighborhood around each pixel for finding potential neighbors to include in clusters.
#   *seeds - Optionally specifies seed points to initialize clustering. These can be predefined locations or left as 'None', in which case SNIC will          automatically generate seeds.
# ---
# 
# **Important:**
# - This is another approach that can be explored as an alternative to the approach specified within the practical assessment. 
#   {Unit segmentation of NDVI. GEE Native Solution. SNIC however can result in some noise and not cover the entire region at larger scales.
# - For this workflow however, we will comment this out and proceed with Segmentation as specified in the Task. 

# --- Code Cell [15] ---
"""
# Perform SNIC segmentation on the edible vegetation
# Positional arguments: image, size, compactness, connectivity, neighborhoodSize, seeds.
snic = ee.Algorithms.Image.Segmentation.SNIC(edible_vegetation, 50, 5, 8, 256, None)

# Select 'clusters' band and mask
clusters = snic.select('clusters').rename('Cluster_ID')
clusters_masked = clusters.selfMask()

# Calculate per-cluster mean NDVI
clusterMeans = clusters_masked.addBands(edible_vegetation).reduceConnectedComponents(
    reducer=ee.Reducer.mean(),
    labelBand='Cluster_ID'
)

# Rename mean band for clarity
clusterNDVI = clusterMeans.select(['Edible'], ['mean_ndvi'])

# Merge cluster ID with mean NDVI
clustered = clusters_masked.addBands(clusterNDVI)

# Visualize random clusters
Map.addLayer(clustered.randomVisualizer(), {}, 'Edible Vegetation Clusters')

# Vectorize the clustered image
vector_clusters = clustered.reduceToVectors(
    reducer=ee.Reducer.first(),
    geometry=gma,
    scale=100,
    geometryType='polygon',
    labelProperty='Cluster_ID',
    maxPixels=1e13
)

# Export clusters as shapefile
task = ee.batch.Export.table.toDrive(
    collection=vector_clusters,
    description=f"{gma_name}_Edible_Vegetation_Clusters",
    folder="UAIs",
    fileFormat="SHP"
)
task.start()

print("SNIC segmentation applied and clustering export started.")
"""

# --- Raw Cell [16] ---
# =============================================================================
# # 1.7.1 Clustering Edible Vegetation Region into Unit Areas of Insurance (UAIs)
# =============================================================================
# 
# This segmentation builds on **Cell 6: Create Edible and Non-Edible Vegetation Layers**, where exclusion zones (water, human settlements, evergreen forest, wetlands) were used to derive an **edible vegetation mask**.
# 
# We now apply NDVI **amplitude-based classification** to segment this edible vegetation into 5 classes, referred to as **Unit Areas of Insurance (UAIs)**. The NDVI amplitude image (`edible_vegetation`) reflects how much vegetation greenness fluctuates seasonally between maximum and minimum states. It helps isolate areas with varying levels of **seasonality and forage potential**.
# 
# Classification Logic:
# - **[0.0 – 0.2)**: Very low fluctuation — either barren/degraded or permanently green (e.g., riparian zones or forest edge).
# - **[0.2 – 0.4)**: Low fluctuation — sparse vegetation with mild seasonality.
# - **[0.4 – 0.6)**: Moderate fluctuation — grasslands or savanna with typical seasonal cycles.
# - **[0.6 – 0.8)**: High fluctuation — productive and dynamic areas, likely ideal grazing zones.
# - **[0.8 – 1.0]**: Very high fluctuation — potentially highly seasonal wetlands, floodplains, or areas of intense biomass cycling.
# 
# Output:
# - A classified raster where each pixel belongs to a UAI zone.
# - This segmentation is later used to extract monthly NDVI trends from 2020 to 2024.
# 
# **Important:**
# Using amplitude instead of mean NDVI ensures we’re identifying zones **not just by how green they are**, but by **how much they change over the       seasons**, which is critical in monitoring forage reliability and variability.

# --- Code Cell [17] ---
# Define NDVI Amplitude thresholds for UAI segmentation
intervals = [(0.0, 0.2), (0.2, 0.4), (0.4, 0.6), (0.6, 0.8), (0.8, 1.0)]
uai_images = []
uai_labels = []

# Create UAI masks from edible vegetation layer
for i, (low, high) in enumerate(intervals):
    uai = edible_vegetation.gte(low).And(edible_vegetation.lt(high))
    uai_mask = uai.updateMask(uai).rename(f'UAI_{i+1}')
    uai_images.append(uai_mask)
    uai_labels.append(f'UAI {i+1}: [{low}, {high})')

# Combine all UAI masks into one labeled image
uai_combined = ee.Image(0)
for i, uai in enumerate(uai_images):
    uai_combined = uai_combined.where(uai, i + 1)

# Visualize segmented edible vegetation UAIs
uai_palette = ['gray', 'yellow', 'orange', 'green', 'darkgreen']
Map.addLayer(uai_combined, {'min': 1, 'max': 5, 'palette': uai_palette}, 'UAIs over Edible Vegetation')
print("Segmented UAIs over edible vegetation added to the map.")

# --- Raw Cell [18] ---
# ==============================================================
# # 1.8 Extracting Monthly NDVI Time Series for UAIs (2020–2024)
# ==============================================================
# 
# The edible UAIs were defined using the NDVI **amplitude** image, which reflects seasonal vegetation change. These zones are fixed spatial clusters.
# 
# Now, we use those UAI zones to extract the actual monthly **NDVI values** (greenness) from MODIS between January 2020 and December 2024. This helps us understand how vegetation behaves over time in each zone. We expect zones with lower fluctuations to depict the same (Low amplitudes) and zones with higher fluctuations to depict (High amplitudes) 
# 
# For each month and each UAI:
# - We apply the UAI mask (from amplitude segmentation).
# - Extract mean NDVI over the area.
# - Store results in a table and visualize with a line chart.
# 
# Output: 
# -As detailed, a couple of datasets are missing 2023-2024 dataset
# -MODIS Dataset maintained for streamlined workflow, VIIRS Satellite Imagery introduced to visualize complete Trend. VIIRS blends well with MODIS.

# --- Code Cell [19] ---
import pandas as pd
import matplotlib.pyplot as plt

# Define time range
months = pd.date_range('2020-01-01', '2024-12-31', freq='MS')
results = []

# Loop through each month and UAI zone
for date in months:
    start = date.strftime('%Y-%m-%d')
    end = (date + pd.DateOffset(months=1)).strftime('%Y-%m-%d')
    
    # Get monthly mean NDVI image
    monthly_coll = modisNDVI.filterDate(start, end)
    monthly_ndvi_img = monthly_coll.mean()

    # Graceful skip if no NDVI band is found
    try:
        band_names = monthly_ndvi_img.bandNames().getInfo()
        if 'NDVI' not in band_names:
            print(f"No NDVI band found for {start} to {end}. Skipping.")
            continue
    except Exception as e:
        print(f"Error checking NDVI band for {start}: {e}. Skipping.")
        continue

    # Clip and select NDVI
    monthly_ndvi = monthly_ndvi_img.select('NDVI').clip(gma)

    for i in range(5):
        zone_mask = uai_combined.eq(i + 1)
        masked_ndvi = monthly_ndvi.updateMask(zone_mask)
        
        # Reduce to mean NDVI value for the zone
        try:
            stats = masked_ndvi.reduceRegion(
                reducer=ee.Reducer.mean(),
                geometry=gma.geometry(),
                scale=250,
                maxPixels=1e13
            )
            ndvi_val = stats.get('NDVI').getInfo() if stats.get('NDVI') else None
        except Exception as e:
            print(f"Error retrieving NDVI for UAI {i+1} at {start}: {e}")
            ndvi_val = None

        results.append({'Date': start, 'UAI': f'UAI {i+1}', 'NDVI': ndvi_val})

# Convert to DataFrame
df = pd.DataFrame(results)
df['Date'] = pd.to_datetime(df['Date'])

# Plot NDVI time series for each UAI
plt.figure(figsize=(12, 6))
for uai in df['UAI'].unique():
    subset = df[df['UAI'] == uai]
    plt.plot(subset['Date'], subset['NDVI'], label=uai)

plt.title('Monthly NDVI Time Series by UAI (Edible Vegetation, 2020–2024)')
plt.xlabel('Date')
plt.ylabel('Mean NDVI')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

# Export time series data to CSV
csv_path = f"{gma_name.replace(' ', '_')}_MODIS_NDVI_TimeSeries_UAIs.csv"
df.to_csv(csv_path, index=False)
print(f"Time series CSV exported: {csv_path}")

# --- Raw Cell [20] ---
# =======================================================================
# # 1.8.1 Extracting Monthly NDVI Time Series using VIIRS NDVI (2020–2024)
# =======================================================================
# 
# This section complements Section 1.8 by using the VIIRS VNP13A1 NDVI product
# as an alternative to MODIS NDVI, which has several data gaps.
# 
# Goals:
# - Visualize the full NDVI trend across UAIs using VIIRS.
# - Compare VIIRS-derived NDVI patterns with those from MODIS (Section 1.8).
# - Support validation of time series robustness using two sensors.
# 
# Notes:
# - VIIRS NDVI has a similar dynamic range (0–1) but is provided at 500m resolution.
# - Scale factor (0.0001) is applied to the raw values.
# - Differences in smoothing and gap patterns may reflect sensor health, design or cloud filtering logic. A depiction of this is the missing UAI 5.
# -UAI 5 is missing in the VIIRS time series as the corresponding zones had no valid NDVI values during the monthly averages, highlighting sensor          differences in capturing high-fluctuation vegetation.
# 
# We will also create a **side-by-side line chart** comparing VIIRS and MODIS NDVI across UAIs.

# --- Code Cell [21] ---
viirs = ee.ImageCollection('NOAA/VIIRS/001/VNP13A1') \
    .filterDate('2020-01-01', '2024-12-31') \
    .filterBounds(gma) \
    .select('NDVI')

viirs_scale_factor = 0.0001

# Define time range
months = pd.date_range('2020-01-01', '2024-12-31', freq='MS')
results_viirs = []

for date in months:
    start = date.strftime('%Y-%m-%d')
    end = (date + pd.DateOffset(months=1)).strftime('%Y-%m-%d')
    
    monthly_coll = viirs.filterDate(start, end)
    monthly_ndvi_img = monthly_coll.mean()

    try:
        band_names = monthly_ndvi_img.bandNames().getInfo()
        if 'NDVI' not in band_names:
            print(f"[VIIRS] No NDVI band for {start} to {end}. Skipping.")
            continue
    except Exception as e:
        print(f"[VIIRS] Band check failed for {start}: {e}. Skipping.")
        continue

    monthly_ndvi = monthly_ndvi_img.select('NDVI').multiply(viirs_scale_factor).clip(gma)

    for i in range(5):
        zone_mask = uai_combined.eq(i + 1)
        masked_ndvi = monthly_ndvi.updateMask(zone_mask)
        
        try:
            stats = masked_ndvi.reduceRegion(
                reducer=ee.Reducer.mean(),
                geometry=gma.geometry(),
                scale=500,
                maxPixels=1e13
            )
            ndvi_val = stats.get('NDVI').getInfo() if stats.get('NDVI') else None
        except Exception as e:
            print(f"[VIIRS] Error for UAI {i+1} at {start}: {e}")
            ndvi_val = None

        results_viirs.append({'Date': start, 'UAI': f'UAI {i+1}', 'NDVI': ndvi_val})

# Convert to DataFrame
df_viirs = pd.DataFrame(results_viirs)
df_viirs['Date'] = pd.to_datetime(df_viirs['Date'])

# Side-by-side MODIS vs VIIRS plots
plt.figure(figsize=(14, 10))

# MODIS
plt.subplot(2, 1, 1)
for uai in df['UAI'].unique():
    subset = df[df['UAI'] == uai]
    plt.plot(subset['Date'], subset['NDVI'], label=uai)
plt.title('MODIS NDVI Time Series (2020–2024)')
plt.ylabel('Mean NDVI')
plt.grid(True)
plt.legend()

# VIIRS
plt.subplot(2, 1, 2)
for uai in df_viirs['UAI'].unique():
    subset = df_viirs[df_viirs['UAI'] == uai]
    plt.plot(subset['Date'], subset['NDVI'], label=uai)
plt.title('VIIRS NDVI Time Series (2020–2024)')
plt.xlabel('Date')
plt.ylabel('Mean NDVI')
plt.grid(True)
plt.legend()

plt.tight_layout()
plt.show()

# Export VIIRS time series to CSV
viirs_csv_path = f"{gma_name.replace(' ', '_')}_VIIRS_NDVI_TimeSeries_UAIs.csv"
df_viirs.to_csv(viirs_csv_path, index=False)
print(f"[VIIRS] Time series CSV exported: {viirs_csv_path}")

# --- Raw Cell [22] ---
# ===============================================================
# # 2.0 Task 2 Introduction: Multi-Index UAI Segmentation (MCDA)
# ===============================================================
# 
# In Task 1, Unit Areas of Insurance (UAIs) were derived using NDVI amplitude alone, focusing on seasonal vegetation fluctuation.
# 
# In this second task, we incorporate **additional ecological and climatic variables** to improve the accuracy and relevance of UAI zoning.
# These include:
# 
# - **Leaf Area Index (LAI)** – Proxy for canopy structure and vegetative biomass.
# - **Precipitation (CHIRPS)** – Indicates total annual moisture input.
# - **Soil Moisture (SMAP)** – Captures available root-zone water.
# 
# Objectives:
# - Load and preprocess each additional dataset for the year 2020.
# - Normalize all inputs and combine them using a **Multi-Criteria Decision Analysis (MCDA)** approach.
# - Segment the resulting composite into 5 UAI zones reflecting integrated ecological and climatic patterns.
# - Extract and analyze NDVI time series for these new MCDA-UAIs.
# - Compare their performance and interpretability against the NDVI-only UAIs.
# 
# This task ultimately aims to show how combining remote sensing indices leads to **more robust spatial zoning** for climate risk monitoring and insurance applications.

# --- Raw Cell [23] ---
# ====================================================================
# # 2.1 Load and Prepare Additional Datasets (LAI, Precip, Soil Moisture)
# ====================================================================
# 
# In this step, we bring in three additional datasets to enhance UAI segmentation:
# - **Leaf Area Index (LAI)**: Captures canopy structure and biomass potential.
# - **Precipitation (CHIRPS)**: Reflects seasonal moisture input patterns.
# - **Soil Moisture (SMAP)**: Indicates sub-surface water availability.
# 
# Each dataset is:
# - Filtered to 2020.
# - Aggregated to an annual **mean** or **sum** (as appropriate).
# - Scaled and clipped to match the edible vegetation region.
# 
# These processed layers will be used in a multi-criteria decision analysis (MCDA) to segment the landscape into new UAIs.

# --- Code Cell [24] ---
# Load and Aggregate External Layers

# Leaf Area Index (MODIS)
lai = ee.ImageCollection('MODIS/006/MCD15A3H') \
    .filterDate('2020-01-01', '2020-12-31') \
    .select('Lai') \
    .mean() \
    .clip(gma)

# CHIRPS Precipitation
precip = ee.ImageCollection('UCSB-CHG/CHIRPS/DAILY') \
    .filterDate('2020-01-01', '2020-12-31') \
    .select('precipitation') \
    .sum() \
    .clip(gma)

# SMAP Soil Moisture (surface)
soil_moisture = ee.ImageCollection('NASA_USDA/HSL/SMAP10KM_soil_moisture') \
    .filterDate('2020-01-01', '2020-12-31') \
    .select('ssm') \
    .mean() \
    .clip(gma)

Map.addLayer(lai, {'min': 0, 'max': 6, 'palette': ['white', 'green']}, 'LAI 2020')
Map.addLayer(precip, {'min': 0, 'max': 2000, 'palette': ['white', 'blue']}, 'Precipitation 2020')
Map.addLayer(soil_moisture, {'min': 0, 'max': 0.5, 'palette': ['white', 'brown']}, 'Soil Moisture 2020')

print("All additional datasets have been added.")

# --- Raw Cell [25] ---
# =====================================================================
# # 2.2 Multi-Criteria Decision Analysis (MCDA) and New UAI Segmentation
# =====================================================================
# 
# We now combine NDVI with Leaf Area Index (LAI), Precipitation (CHIRPS), and Soil Moisture (SMAP)
# to derive an improved UAI segmentation through **Multi-Criteria Decision Analysis (MCDA)**.
# 
# Method:
# 1. Each dataset is **aggregated to 2020** (mean/sum as appropriate).
# 2. Normalize each raster (0–1 scale) using min-max scaling based on regional stats.
# 3. Assign equal weights to all datasets and used a simple **weighted overlay**.
# 4. The composite MCDA score was then segmented into **5 UAI zones** using thresholds.
# 
# This approach creates more ecologically and climatically meaningful zones for forage monitoring and insurance modeling.

# --- Code Cell [26] ---
# Normalize all layers (0–1)
def normalize(image, min_val, max_val):
    return image.subtract(min_val).divide(max_val - min_val).clamp(0, 1)

# Use edible_vegetation (masked amplitude values) as NDVI layer
ndvi_amp_norm = normalize(edible_vegetation, 0.1, 0.9)

# Normalization Stats
# Compute LAI stats over edible vegetation zone
laiStats = lai.reduceRegion(
    reducer=ee.Reducer.minMax().combine(ee.Reducer.mean(), None, True),
    geometry=gma.geometry(),
    scale=250,
    maxPixels=1e13
)
print('LAI Stats:', laiStats.getInfo())

# Compute precipitation stats over edible vegetation zone
precipStats = precip.reduceRegion(
    reducer=ee.Reducer.minMax().combine(ee.Reducer.mean(), None, True),
    geometry=gma.geometry(),
    scale=250,
    maxPixels=1e13
)
print('Precipitation Stats (2020):', precipStats.getInfo())

# Compute soil moisture stats over edible vegetation zone
soilStats = soil_moisture.reduceRegion(
    reducer=ee.Reducer.minMax().combine(ee.Reducer.mean(), None, True),
    geometry=gma.geometry(),
    scale=250,
    maxPixels=1e13
)
print('Soil Moisture Stats:', soilStats.getInfo())

# Apply stats for normalization
lai_norm = normalize(lai, 2, 20)
precip_norm = normalize(precip, 900, 1400)
soil_norm = normalize(soil_moisture, 9, 15)

# MCDA Composite (equal weight)
mcda = ndvi_amp_norm.add(lai_norm).add(precip_norm).add(soil_norm).divide(4).rename('MCDA')

Map.addLayer(mcda, {'min': 0, 'max': 1, 'palette': ['white', 'yellow', 'green']}, 'MCDA Composite (Edible Vegetation Only)')

# Segment MCDA into 5 UAI Zones (like 1.7.1)
intervals = [(0.0, 0.2), (0.2, 0.4), (0.4, 0.6), (0.6, 0.8), (0.8, 1.0)]
mcda_images = []
mcda_labels = []

for i, (low, high) in enumerate(intervals):
    uai = mcda.gte(low).And(mcda.lt(high))
    uai_mask = uai.updateMask(uai).rename(f'MCDA_UAI_{i+1}')
    mcda_images.append(uai_mask)
    mcda_labels.append(f'MCDA UAI {i+1}: [{low}, {high})')

# Combine masks into a single MCDA UAI layer
mcda_uai = ee.Image(0)
for i, uai in enumerate(mcda_images):
    mcda_uai = mcda_uai.where(uai, i + 1)

Map.addLayer(mcda_uai, {'min': 1, 'max': 5, 'palette': ['gray', 'yellow', 'orange', 'green', 'darkgreen']}, 'MCDA UAIs')
print("MCDA-based UAIs over edible vegetation added to the map.")

# --- Raw Cell [27] ---
# =====================================================
# # 2.3 Monthly NDVI Time Series for MCDA-based UAIs
# =====================================================
# 
# In this step, we apply the new MCDA-based UAI segmentation to extract NDVI time series from MODIS between 2020 and 2024.
# 
# This helps evaluate whether the MCDA UAIs provide clearer vegetation dynamics and potentially better insurance zones than NDVI-only ones.

# --- Code Cell [28] ---
import pandas as pd
import matplotlib.pyplot as plt

# Define time range
months = pd.date_range('2020-01-01', '2024-12-31', freq='MS')
results_mcda = []

# Loop through each month and MCDA UAI zone
for date in months:
    start = date.strftime('%Y-%m-%d')
    end = (date + pd.DateOffset(months=1)).strftime('%Y-%m-%d')

    # Get monthly mean NDVI image
    monthly_coll = modisNDVI.filterDate(start, end)
    monthly_ndvi_img = monthly_coll.mean()

    try:
        band_names = monthly_ndvi_img.bandNames().getInfo()
        if 'NDVI' not in band_names:
            print(f"[MODIS] No NDVI band for {start} to {end}. Skipping.")
            continue
    except Exception as e:
        print(f"[MODIS] Band check failed for {start}: {e}. Skipping.")
        continue

    # Clip and select NDVI
    monthly_ndvi = monthly_ndvi_img.select('NDVI').clip(gma)

    for i in range(5):
        zone_mask = mcda_uai.eq(i + 1)
        masked_ndvi = monthly_ndvi.updateMask(zone_mask)

        try:
            stats = masked_ndvi.reduceRegion(
                reducer=ee.Reducer.mean(),
                geometry=gma.geometry(),
                scale=250,
                maxPixels=1e13
            )
            ndvi_val = stats.get('NDVI').getInfo() if stats.get('NDVI') else None
        except Exception as e:
            print(f"[MODIS] Error retrieving NDVI for MCDA UAI {i+1} at {start}: {e}")
            ndvi_val = None

        results_mcda.append({'Date': start, 'UAI': f'MCDA UAI {i+1}', 'NDVI': ndvi_val})

# Convert to DataFrame
df_mcda = pd.DataFrame(results_mcda)
df_mcda['Date'] = pd.to_datetime(df_mcda['Date'])

# Plot NDVI time series for each MCDA UAI
plt.figure(figsize=(12, 6))
for uai in df_mcda['UAI'].unique():
    subset = df_mcda[df_mcda['UAI'] == uai]
    plt.plot(subset['Date'], subset['NDVI'], label=uai)

plt.title('Monthly NDVI Time Series by MCDA UAI (MODIS, 2020–2024)')
plt.xlabel('Date')
plt.ylabel('Mean NDVI')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

# Export to CSV
csv_path_mcda = f"{gma_name.replace(' ', '_')}_MODIS_NDVI_TimeSeries_MCDA_UAIs.csv"
df_mcda.to_csv(csv_path_mcda, index=False)
print(f"[MODIS] MCDA time series CSV exported: {csv_path_mcda}")

# --- Raw Cell [29] ---
# ==================================================================
# # 2.3.1 VIIRS vs MODIS Time Series over MCDA-based UAI Zones
# ==================================================================
# 
# We compare MODIS and VIIRS NDVI time series across the MCDA UAIs to:
# - Evaluate sensor consistency
# - Confirm whether VIIRS provides reliable alternative for future scaling
# - Fill data gaps

# --- Code Cell [30] ---
viirs = ee.ImageCollection('NOAA/VIIRS/001/VNP13A1') \
    .filterDate('2020-01-01', '2024-12-31') \
    .filterBounds(gma) \
    .select('NDVI')

viirs_scale_factor = 0.0001
results_viirs_mcda = []

for date in months:
    start = date.strftime('%Y-%m-%d')
    end = (date + pd.DateOffset(months=1)).strftime('%Y-%m-%d')

    monthly_coll = viirs.filterDate(start, end)
    monthly_ndvi_img = monthly_coll.mean()

    try:
        band_names = monthly_ndvi_img.bandNames().getInfo()
        if 'NDVI' not in band_names:
            print(f"[VIIRS] No NDVI band for {start} to {end}. Skipping.")
            continue
    except Exception as e:
        print(f"[VIIRS] Band check failed for {start}: {e}. Skipping.")
        continue

    monthly_ndvi = monthly_ndvi_img.select('NDVI').multiply(viirs_scale_factor).clip(gma)

    for i in range(5):
        zone_mask = mcda_uai.eq(i + 1)
        masked_ndvi = monthly_ndvi.updateMask(zone_mask)

        try:
            stats = masked_ndvi.reduceRegion(
                reducer=ee.Reducer.mean(),
                geometry=gma.geometry(),
                scale=500,
                maxPixels=1e13
            )
            ndvi_val = stats.get('NDVI').getInfo() if stats.get('NDVI') else None
        except Exception as e:
            print(f"[VIIRS] Error for MCDA UAI {i+1} at {start}: {e}")
            ndvi_val = None

        results_viirs_mcda.append({'Date': start, 'UAI': f'MCDA UAI {i+1}', 'NDVI': ndvi_val})

# Convert to DataFrame
df_viirs_mcda = pd.DataFrame(results_viirs_mcda)
df_viirs_mcda['Date'] = pd.to_datetime(df_viirs_mcda['Date'])

# Plot MODIS vs VIIRS time series
plt.figure(figsize=(14, 10))

# MODIS
plt.subplot(2, 1, 1)
for uai in df_mcda['UAI'].unique():
    subset = df_mcda[df_mcda['UAI'] == uai]
    plt.plot(subset['Date'], subset['NDVI'], label=uai)
plt.title('MODIS NDVI Time Series (MCDA UAIs)')
plt.ylabel('Mean NDVI')
plt.grid(True)
plt.legend()

# VIIRS
plt.subplot(2, 1, 2)
for uai in df_viirs_mcda['UAI'].unique():
    subset = df_viirs_mcda[df_viirs_mcda['UAI'] == uai]
    plt.plot(subset['Date'], subset['NDVI'], label=uai)
plt.title('VIIRS NDVI Time Series (MCDA UAIs)')
plt.xlabel('Date')
plt.ylabel('Mean NDVI')
plt.grid(True)
plt.legend()

plt.tight_layout()
plt.show()

# Export VIIRS results to CSV
viirs_csv_path_mcda = f"{gma_name.replace(' ', '_')}_VIIRS_NDVI_TimeSeries_MCDA_UAIs.csv"
df_viirs_mcda.to_csv(viirs_csv_path_mcda, index=False)
print(f"[VIIRS] MCDA time series CSV exported: {viirs_csv_path_mcda}")

# --- Raw Cell [31] ---
# ===================================================================
# # 2.4 NDVI-only vs MCDA-UAIs: Time Series Comparison (MODIS Only)
# ===================================================================
# 
# Here we visually compare the NDVI trends extracted from:
# - NDVI-only UAIs (`uai_combined`)
# - MCDA-UAIs (`mcda_uai`)
# 
# This helps us understand how adding multiple datasets affects the zonation
# and what that means for forage dynamics and drought monitoring.

# --- Code Cell [32] ---
plt.figure(figsize=(14, 6))

for i in range(5):
    ndvi_only_label = f'UAI {i+1}'
    mcda_label = f'MCDA UAI {i+1}'

    # Extract MODIS NDVI-only UAI time series (from 1.8)
    subset_ndvi = df[df['UAI'] == ndvi_only_label]
    
    # Extract MODIS MCDA UAI time series (from 2.3)
    subset_mcda = df_mcda[df_mcda['UAI'] == mcda_label]

    # Plot both
    plt.plot(subset_ndvi['Date'], subset_ndvi['NDVI'], label=f'{ndvi_only_label} (NDVI-only)', linestyle='--')
    plt.plot(subset_mcda['Date'], subset_mcda['NDVI'], label=f'{mcda_label} (MCDA)', linestyle='-')

plt.title("NDVI-only vs MCDA UAIs (MODIS NDVI, 2020–2024)")
plt.xlabel("Date")
plt.ylabel("Mean NDVI")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

# Combine both dataframes for export
df_comparison = pd.concat([
    df.assign(Segment='NDVI-only'),
    df_mcda.assign(Segment='MCDA')
])

# Export combined comparison table
comparison_csv_path = f"{gma_name.replace(' ', '_')}_NDVI_Comparison_NDVI_vs_MCDA.csv"
df_comparison.to_csv(comparison_csv_path, index=False)
print(f"[Comparison] NDVI-only vs MCDA time series exported: {comparison_csv_path}")

# --- Raw Cell [33] ---
# =====================================================
# # 3.0 Maps, Raster Exports, and Dashboard Preparation
# =====================================================
# 
# With all Unit Areas of Insurance (UAIs) generated in Sections 1 and 2, this section focuses on **spatial outputs** and **exporting results** for visualization and integration into external platforms.
# 
# Objectives:
# - Visualize all key raster layers and vegetation classes on the map.
# - Export raster outputs (GeoTIFFs) for NDVI-only UAIs, MCDA UAIs, and key land cover masks (e.g., water, evergreen, edible vegetation).
# - Ensure all layers are **clipped**, properly named, and **ready for integration** into QGIS, ArcGIS, or an interactive dashboard (e.g., Streamlit or Leaflet).
# - Optionally prepare shapefiles or summary charts for overlay or web-based use.
# 
# This step ensures the workflow transitions from **analytical output to practical application**, supporting tools like parametric insurance dashboards or environmental monitoring portals.

# --- Raw Cell [34] ---
# =================================================
# # 3.1 Export Land Cover Masks as GeoTIFFs
# =================================================
# 
# In this step, all important thematic masks (water bodies, human activity, evergreen vegetation, wetlands, edible vegetation, and non-edible vegetation) are **converted to GeoTIFF raster images** and exported to Google Drive.
# 
# Approach:
# - **GeoTIFFs** are lightweight and faster for interactive mapping and visualization.
# - Raster-based layers are ideal for **web map overlays**, **risk analysis**, and **landscape monitoring**.
# - Each thematic mask highlights a different vegetation or land cover type important for hazard and risk assessments.
# 
# Coding Notes:
# - **Thresholding** is applied: continuous vegetation masks (edible and non-edible) are **binarized** (0 or 1) for export.
# - `selfMask()` ensures no-data areas are transparent (masked out).
# - `ee.batch.Export.image.toDrive` is used to export each thematic mask as an individual GeoTIFF.
# - TIFFs are exported at **10m resolution** (scale=10) and **EPSG:4326** projection.
# 
# ---
# 
# **Important:**
# - Exported filenames automatically include the `gma_name` for easier file organization.
# - Download the exported TIFFs from Google Drive before proceeding to mapping.

# --- Code Cell [35] ---
# For vectorization, convert the continuous edible and non_edible_vegetation to a binary image.
# Here, we threshold > 0; adjust this threshold if needed.
edible_veg = edible_vegetation.gt(0).selfMask()
non_edible_veg = non_edible.gt(0).selfMask()

# Export function for masks (TIFF export)
def export_mask_to_tiff(mask, name):
    task = ee.batch.Export.image.toDrive(
        image=mask,
        description=f"{name}",
        folder="GMAs",
        fileNamePrefix=name,
        region=gma.geometry(),  # Area of interest
        scale=10,
        crs="EPSG:4326",  # or match your preferred CRS
        maxPixels=1e13,
        fileFormat="GeoTIFF"
    )
    task.start()
    print(f"Export task started for {name}. Check Google Drive for TIFF file.")

# Export masks
export_mask_to_tiff(water, f"{gma_name}_water_bodies")
export_mask_to_tiff(human, f"{gma_name}_human_activity")
export_mask_to_tiff(evergreen, f"{gma_name}_evergreen_vegetation")
export_mask_to_tiff(wetlands, f"{gma_name}_wetlands")
export_mask_to_tiff(edible_veg, f"{gma_name}_edible_vegetation")
export_mask_to_tiff(non_edible_veg, f"{gma_name}_non_edible_vegetation")
export_mask_to_tiff(uai_combined, f"{gma_name}_uai_segmentation")
export_mask_to_tiff(mcda_uai, f"{gma_name}_mcda_uai_segmentation")


print("All specified land cover masks are being exported as TIFFs.")

# --- Raw Cell [36] ---
# ====================================================================
# # 3.1.1 Convert Exported TIFFs to Shapefiles (Offline Post-Processing)
# ====================================================================
# 
# After exporting all thematic land cover masks as **GeoTIFFs** in Step 1.9, this optional step converts those raster masks into **vector shapefiles (SHP)** using Python.
# 
# Importance:
# - **Shapefiles are better suited for post-analysis** in GIS software like QGIS or ArcGIS.
# - While Google Earth Engine struggled with direct shapefile exports due to memory and geometry-type limitations, TIFFs are computationally lightweight    and export reliably.
# - Converting locally ensures **full control**, avoids geometry-type errors (e.g., mixing `Polygon` and `LineString`), and supports **batch processing**.
# - This step only extracts valid data areas (non-zero, non-nodata pixels) as **Polygon** features — removing background and noise.
# 
# Approach:
# - Use Python's `rasterio` to read each exported TIFF mask.
# - Apply a binary mask to remove nodata regions.
# - Use `rasterio.features.shapes` and `shapely` to convert valid regions to polygons.
# - Output each shapefile with the **same name** as the input `.tif`, saved to a designated output directory.
# 
# Note:
# - ⚠️ This process runs for quite sometime due to the complexity and large number of polygons. Processing Speed also depends on PC capabilities and        spatial extent.
# - Therefore, if **Output** is not available within the repository at the submission deadline (4.00 PM EAT 5-12-2025), will be pushed to repository on     completion within the same date.
#  

# --- Code Cell [37] ---
import os
import rasterio
from rasterio.features import shapes
import geopandas as gpd
import numpy as np
from shapely.geometry import shape

# === CONFIGURE INPUT AND OUTPUT FOLDERS ===
# Define input and output folders
# Get the current working directory
input_folder = os.getcwd()  # Dynamically set to the notebook's directory
output_folder = os.path.join(input_folder, "layer_shapefiles")  # Output within working directory

# Ensure the output folder exists
if not os.path.exists(output_folder):
    os.makedirs(output_folder)

# === PROCESS ALL TIF FILES ===
for filename in os.listdir(input_folder):
    if filename.lower().endswith(".tif"):
        input_path = os.path.join(input_folder, filename)
        output_path = os.path.join(output_folder, filename.replace(".tif", ".shp"))

        print(f"Processing {filename}...")

        with rasterio.open(input_path) as src:
            image = src.read(1)
            mask = image != src.nodata  # Exclude nodata pixels
            transform = src.transform
            crs = src.crs

        polygons = []
        values = []

        for geom, value in shapes(image, mask=mask, transform=transform):
            if value != 0:  # Optional: skip background/zero if needed
                polygons.append(shape(geom))
                values.append(value)

        if polygons:
            gdf = gpd.GeoDataFrame({'value': values, 'geometry': polygons}, crs=crs)
            gdf = gdf.dissolve()  # Merge all polygons into one feature
            gdf.to_file(output_path)
            print(f"Saved: {output_path}")
        else:
            print(f"No valid polygons found in {filename}.")

print("All raster files have been processed.")

# --- Raw Cell [38] ---
# ================================================
# # 3.2 Build Interactive Map to Visualize Outputs
# ================================================
# 
# This step creates a **web-based interactive map** displaying both exported **shapefiles** and **GeoTIFF raster images**, with **color-coded layers** representing different landscape features.
# 
# Raster images (GeoTIFFs) allow for **faster and more lightweight rendering** compared to shapefiles, making them more suitable for efficient web map visualization.
# 
# Approach:
# - **Raster layers** (GeoTIFFs) are used for vegetation, water, human activities, and wetlands.
# - **Shapefile layers** are used for the GMA boundary and Edible Vegetation Clusters.
# - Each feature is **color-coded** for clear differentiation.
# - **Transparent backgrounds** are created for raster layers to avoid displaying no-data areas.
# - A **custom legend** and **layer control** allow users to easily navigate the map.
# 
# Purpose:
# - Supports **stakeholder communication** and **decision-making** with visual clarity.
# - Enables users to **toggle layers**, **zoom** into important zones, and **explore spatial patterns** interactively.
# 
# Coding Notes:
# - `folium` is used to create the interactive map and add both shapefiles and raster overlays. Folium is a wrapper around **Leaflet.js**
# - `geopandas` is used to load shapefile geometries.
# - `rasterio` and `numpy` are used to process GeoTIFFs (mask background values and apply colors).
# - A **custom HTML legend** is embedded using `branca`.
# - The exported map is saved as an HTML file that can be opened in any web browser.
# 
# ---
# 
# **Important:**
# - Download the exported shapefiles and GeoTIFFs from Google Drive before running the mapping script.
# - Ensure all file paths are correct relative to the notebook's working directory.

# --- Code Cell [39] ---
import folium
import rasterio
import numpy as np
import geopandas as gpd
import branca
from folium import raster_layers

# Load shapefiles
gdf_boundary = gpd.read_file(f"{gma_name}.shp")
# gdf_clusters = gpd.read_file(f"{gma_name}_Edible_Vegetation_Clusters.shp")

# Load binary GeoTIFFs
tiff_layers = {
    "Water Bodies": f"{gma_name}_water_bodies.tif",
    "Human Activity": f"{gma_name}_human_activity.tif",
    "Wetlands": f"{gma_name}_wetlands.tif",
    "Edible Vegetation": f"{gma_name}_edible_vegetation.tif",
    "Non Edible Vegetation": f"{gma_name}_non_edible_vegetation.tif",
    "Evergreen Vegetation": f"{gma_name}_evergreen_vegetation.tif",  # optional
}

# Define multi-class layers
uai_layers = {
    "UAI Segmentation": f"{gma_name}_uai_segmentation.tif",
    "MCDA UAI Segmentation": f"{gma_name}_mcda_uai_segmentation.tif"
}

# Color settings for binary layers
color_map = {
    "Water Bodies": "#003df5",
    "Human Activity": "#800080",
    "Wetlands": "#87CEFA",
    "Edible Vegetation": "#7CFC00",
    "Non Edible Vegetation": "#F4A460",
    "Evergreen Vegetation": "#006400",
}

# Color settings for UAIs (multi-class)
uai_colormap = {
    1: "#ffffcc",  # Zone 1
    2: "#a1dab4",  # Zone 2
    3: "#41b6c4",  # Zone 3
    4: "#2c7fb8",  # Zone 4
    5: "#253494",  # Zone 5
}

# Center map
centroid = gdf_boundary.geometry.centroid.iloc[0]
latitude, longitude = centroid.y, centroid.x

m = folium.Map(location=[latitude, longitude], zoom_start=10, tiles=None)

# Base layers
folium.TileLayer('OpenStreetMap').add_to(m)
folium.TileLayer(
    tiles='https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}',
    attr='Esri',
    name='Esri Satellite',
    overlay=False,
    control=True
).add_to(m)

# Add binary TIFF layer
def add_tiff_layer(filepath, layer_name, color_hex, map_object):
    with rasterio.open(filepath) as src:
        img = src.read(1)
        bounds = src.bounds

    mask = np.where(img == 1, 255, 0).astype(np.uint8)
    color_img = np.zeros((img.shape[0], img.shape[1], 4), dtype=np.uint8)
    r, g, b = tuple(int(color_hex.lstrip("#")[i:i+2], 16) for i in (0, 2, 4))

    color_img[:, :, 0] = r
    color_img[:, :, 1] = g
    color_img[:, :, 2] = b
    color_img[:, :, 3] = mask

    raster = raster_layers.ImageOverlay(
        image=color_img,
        bounds=[[bounds.bottom, bounds.left], [bounds.top, bounds.right]],
        name=layer_name,
        opacity=1,
        interactive=True,
        cross_origin=False,
        zindex=1,
    )
    raster.add_to(map_object)

# Add multi-class TIFF layer
def add_multiclass_tiff(filepath, layer_name, class_colormap, map_object):
    with rasterio.open(filepath) as src:
        img = src.read(1)
        bounds = src.bounds

    height, width = img.shape
    rgba_img = np.zeros((height, width, 4), dtype=np.uint8)

    for class_val, color_hex in class_colormap.items():
        mask = img == class_val
        r, g, b = tuple(int(color_hex.lstrip("#")[i:i+2], 16) for i in (0, 2, 4))
        rgba_img[mask, 0] = r
        rgba_img[mask, 1] = g
        rgba_img[mask, 2] = b
        rgba_img[mask, 3] = 255  # Fully visible

    raster = raster_layers.ImageOverlay(
        image=rgba_img,
        bounds=[[bounds.bottom, bounds.left], [bounds.top, bounds.right]],
        name=layer_name,
        opacity=1,
        interactive=True,
        cross_origin=False,
        zindex=1,
    )
    raster.add_to(map_object)

# Add all binary raster layers
for name, filepath in tiff_layers.items():
    add_tiff_layer(filepath, name, color_map[name], m)

# Add all UAI multi-class raster layers
for name, filepath in uai_layers.items():
    add_multiclass_tiff(filepath, name, uai_colormap, m)

# Add boundaries and clusters
folium.GeoJson(gdf_boundary, name=f"{gma_name} Boundary", style_function=lambda x: {
    "color": "black", "weight": 3, "fillOpacity": 0
}).add_to(m)

# folium.GeoJson(gdf_clusters, name="Edible Vegetation Clusters", style_function=lambda x: {
#     "color": "red", "weight": 1, "fillOpacity": 0
# }).add_to(m)

# Legend HTML
legend_html = '''
<div style="position: fixed; bottom: 50px; left: 50px; width: 250px; height: auto;
            background-color: white; z-index:9999; font-size:14px;
            border:1px solid black; padding: 10px;">
<b>Map Legend</b><br>
<span style="color:#003df5;">■</span> Water Bodies <br>
<span style="color:#800080;">■</span> Human Activity <br>
<span style="color:#006400;">■</span> Evergreen Vegetation <br>
<span style="color:#87CEFA;">■</span> Wetlands <br>
<span style="color:#7CFC00;">■</span> Edible Vegetation <br>
<span style="color:#F4A460;">■</span> Non Edible Vegetation <br>
<div style="display:inline-block; width:7px; height:7px; border:1px solid red; background:transparent; margin-right:2px;"></div> Edible Vegetation Clusters <br>
<b>UAI Zones</b><br>
<span style="color:#ffffcc;">■</span> Zone 1<br>
<span style="color:#a1dab4;">■</span> Zone 2<br>
<span style="color:#41b6c4;">■</span> Zone 3<br>
<span style="color:#2c7fb8;">■</span> Zone 4<br>
<span style="color:#253494;">■</span> Zone 5<br>
</div>
'''
m.get_root().html.add_child(branca.element.Element(legend_html))

# Final map controls
folium.LayerControl(collapsed=False).add_to(m)

# Save map
m.save(f"{gma_name}_Interactive_Map.html")
print(f"Interactive map for {gma_name} using GeoTIFF layers generated successfully.")


# --- Code Cell [40] ---

