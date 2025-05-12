import streamlit as st
import geopandas as gpd
import rasterio
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from streamlit_folium import st_folium
import folium
from folium import raster_layers
from folium.plugins import MousePosition
import branca
import os
from matplotlib.colors import hex2color
import io

# ===========================
# SETUP
# ===========================
st.set_page_config(layout="wide", page_title="Vegetation Analysis Dashboard")
st.sidebar.title("Dashboard")
tab = st.sidebar.radio("Select View", ["Map", "Charts", "Documentation"])

gma_name = st.sidebar.text_input("Enter GMA name", value="Bangweulu")

# Optional: Upload CSVs for dynamic charting
uploaded_csvs = st.sidebar.file_uploader("Upload CSV charts", type="csv", accept_multiple_files=True)

# ===========================
# FILE PATHS
# ===========================
tiff_layers = {
    "Water Bodies": f"assets/{gma_name}_water_bodies.tif",
    "Human Activity": f"assets/{gma_name}_human_activity.tif",
    "Wetlands": f"assets/{gma_name}_wetlands.tif",
    "Edible Vegetation": f"assets/{gma_name}_edible_vegetation.tif",
    "Non Edible Vegetation": f"assets/{gma_name}_non_edible_vegetation.tif",
    "Evergreen Vegetation": f"assets/{gma_name}_evergreen_vegetation.tif",
}

uai_layers = {
    "UAI Segmentation": f"assets/{gma_name}_uai_segmentation.tif",
    "MCDA UAI Segmentation": f"assets/{gma_name}_mcda_uai_segmentation.tif"
}

color_map = {
    "Water Bodies": "#003df5",
    "Human Activity": "#800080",
    "Wetlands": "#87CEFA",
    "Edible Vegetation": "#7CFC00",
    "Non Edible Vegetation": "#F4A460",
    "Evergreen Vegetation": "#006400",
}

uai_colormap = {
    1: "#ffffcc", 2: "#a1dab4", 3: "#41b6c4", 4: "#2c7fb8", 5: "#253494"
}

# Helper function to safely convert hex color to RGB tuple
def hex_to_rgb(hex_color):
    # Remove # if present
    hex_color = hex_color.lstrip('#')
    # Convert to RGB
    return tuple(int(hex_color[i:i+2], 16) for i in (0, 2, 4))

# ===========================
# 1. MAP TAB
# ===========================
if tab == "Map":
    st.header("Interactive Map with Edible Vegetation and UAI Layers")

    shape_path = f"assets/{gma_name}.shp"
    if os.path.exists(shape_path):
        gdf_boundary = gpd.read_file(shape_path)
        centroid = gdf_boundary.geometry.centroid.iloc[0]
        m = folium.Map(location=[centroid.y, centroid.x], zoom_start=10)

        folium.TileLayer('OpenStreetMap').add_to(m)
        folium.TileLayer('Stamen Terrain').add_to(m)
        MousePosition().add_to(m)

        folium.GeoJson(gdf_boundary, name="Boundary", style_function=lambda x: {
            "color": "black", "weight": 2, "fillOpacity": 0
        }).add_to(m)

        def add_tiff_layer(filepath, name, color, map_obj):
            if not os.path.exists(filepath):
                st.warning(f"Layer file not found: {filepath}")
                return
                
            try:
                with rasterio.open(filepath) as src:
                    img = src.read(1)
                    bounds = src.bounds
                    
                # Create mask
                mask = np.where(img == 1, 255, 0).astype(np.uint8)
                
                # Get RGB values from hex color
                r, g, b = hex_to_rgb(color)
                
                # Create RGBA array
                h, w = img.shape
                rgba = np.zeros((h, w, 4), dtype=np.uint8)
                rgba[..., 0] = np.where(mask == 255, r, 0)
                rgba[..., 1] = np.where(mask == 255, g, 0)
                rgba[..., 2] = np.where(mask == 255, b, 0)
                rgba[..., 3] = mask
                
                # Create raster layer
                raster = raster_layers.ImageOverlay(
                    image=rgba,
                    bounds=[[bounds.bottom, bounds.left], [bounds.top, bounds.right]],
                    name=name,
                    opacity=0.7
                )
                raster.add_to(map_obj)
            except Exception as e:
                st.error(f"Error adding layer {name}: {str(e)}")

        def add_multiclass_layer(filepath, name, cmap, map_obj):
            if not os.path.exists(filepath):
                st.warning(f"Layer file not found: {filepath}")
                return
                
            try:
                with rasterio.open(filepath) as src:
                    img = src.read(1)
                    bounds = src.bounds
                    
                # Create RGBA array
                h, w = img.shape
                rgba = np.zeros((h, w, 4), dtype=np.uint8)
                
                # Assign colors based on class values
                for val, hex_color in cmap.items():
                    r, g, b = hex_to_rgb(hex_color)
                    mask = (img == val)
                    rgba[mask, 0] = r
                    rgba[mask, 1] = g
                    rgba[mask, 2] = b
                    rgba[mask, 3] = 255
                
                # Create raster layer
                raster = raster_layers.ImageOverlay(
                    image=rgba,
                    bounds=[[bounds.bottom, bounds.left], [bounds.top, bounds.right]],
                    name=name,
                    opacity=0.7
                )
                raster.add_to(map_obj)
            except Exception as e:
                st.error(f"Error adding multiclass layer {name}: {str(e)}")

        # Add boundary first then layers
        with st.spinner("Loading map layers..."):
            # Add tiff layers
            for name, path in tiff_layers.items():
                if os.path.exists(path):
                    add_tiff_layer(path, name, color_map[name], m)
                else:
                    st.info(f"Layer not found: {name}")

            # Add UAI layers
            for name, path in uai_layers.items():
                if os.path.exists(path):
                    add_multiclass_layer(path, name, uai_colormap, m)
                else:
                    st.info(f"Layer not found: {name}")

        # Legend
        legend_html = '''
        <div style="position: fixed; 
                    bottom: 50px; 
                    left: 50px; 
                    width: 250px; 
                    background-color: white; 
                    border: 2px solid grey; 
                    border-radius: 5px;
                    padding: 10px; 
                    z-index: 9999; 
                    font-size: 14px;">
        <h4 style="margin-top: 0;">Map Legend</h4>
        '''
        
        # Add layer colors
        for k, v in color_map.items():
            legend_html += f'<div><span style="color:{v}; font-size: 20px;">■</span> {k}</div>'
        
        # Add UAI zone colors
        legend_html += '<h4>UAI Zones</h4>'
        for i, color in uai_colormap.items():
            legend_html += f'<div><span style="color:{color}; font-size: 20px;">■</span> Zone {i}</div>'
        
        legend_html += '</div>'
        
        m.get_root().html.add_child(branca.element.Element(legend_html))
        folium.LayerControl().add_to(m)
        
        # Display map with specified width and height
        map_data = st_folium(m, width=1200, height=700)
    else:
        st.error(f"Shapefile not found: {shape_path}")
        st.info("Please make sure the shapefile exists in the assets folder or enter a different GMA name.")

# ===========================
# 2. CHARTS TAB
# ===========================
elif tab == "Charts":
    st.header("NDVI Time Series Charts")

    predefined_charts = [
        ("NDVI vs MCDA (MODIS)", f"assets/{gma_name}_NDVI_Comparison_NDVI_vs_MCDA.csv"),
        ("MODIS UAIs", f"assets/{gma_name}_MODIS_NDVI_TimeSeries_UAIs.csv"),
        ("MODIS MCDA UAIs", f"assets/{gma_name}_MODIS_NDVI_TimeSeries_MCDA_UAIs.csv"),
        ("VIIRS UAIs", f"assets/{gma_name}_VIIRS_NDVI_TimeSeries_UAIs.csv"),
        ("VIIRS MCDA UAIs", f"assets/{gma_name}_VIIRS_NDVI_TimeSeries_MCDA_UAIs.csv")
    ]

    for title, path in predefined_charts:
        if os.path.exists(path):
            st.subheader(title)
            try:
                df = pd.read_csv(path)
                if df.shape[1] > 1:
                    options = df.columns[1:]
                    selected = st.multiselect(f"Columns for {title}", options, default=list(options)[:3]) # Default to first 3 columns
                    if selected:
                        st.line_chart(df.set_index(df.columns[0])[selected])
                    else:
                        st.info("Please select columns to display the chart")
                else:
                    st.warning(f"The file {path} contains only one column. At least two columns are needed for a chart.")
            except Exception as e:
                st.error(f"Error loading {title}: {str(e)}")
        else:
            st.info(f"Chart file not found: {path}")

    if uploaded_csvs:
        st.markdown("---")
        st.subheader("Uploaded Charts")
        for file in uploaded_csvs:
            st.subheader(file.name)
            try:
                df = pd.read_csv(file)
                if df.shape[1] > 1:
                    options = df.columns[1:]
                    selected = st.multiselect(f"Columns for {file.name}", options, default=list(options)[:3])
                    if selected:
                        st.line_chart(df.set_index(df.columns[0])[selected])
                    else:
                        st.info("Please select columns to display the chart")
                else:
                    st.warning(f"The file {file.name} contains only one column. At least two columns are needed for a chart.")
            except Exception as e:
                st.error(f"Error loading {file.name}: {str(e)}")

# ===========================
# 3. DOCUMENTATION TAB
# ===========================
elif tab == "Documentation":
    st.header("Workflow Narrative & Code")

    st.markdown("""
    ## Short Narrative (1-page)

    This dashboard supports the derivation of **Edible Vegetation Masks** and **Unit Areas of Insurance (UAIs)** in Game Management Areas using MODIS NDVI and other supporting datasets. The goal is to segment land based on forage availability for wildlife.

    ### Workflow Overview:
    1. Load ESA WorldCover and MODIS NDVI (2020–2024).
    2. Mask non-edible areas (e.g., water, settlements, evergreen forests).
    3. Compute NDVI mean and amplitude to identify edible vegetation.
    4. Segment vegetation into homogeneous NDVI zones using a 0.2 unit interval.
    5. Apply Multi-Criteria Decision Analysis (MCDA) using NDVI, LAI, precipitation, and soil moisture.
    6. Generate UAI shapefiles and raster outputs.
    7. Visualize in an interactive dashboard.

    The workflow is fully automated, reproducible, and designed with minimal dependence on desktop GIS tools.
    """)

    code_file = "assets/forage_gma.py"
    with st.expander("Click to view dashboard source code"):
        if os.path.exists(code_file):
            try:
                with open(code_file) as f:
                    st.code(f.read(), language="python")
            except Exception as e:
                st.error(f"Error reading source code file: {str(e)}")
        else:
            st.info(f"Source code file not found: {code_file}")
            st.code("""
# Source code file not found. 
# This would typically contain the processing logic for vegetation analysis.
""", language="python")