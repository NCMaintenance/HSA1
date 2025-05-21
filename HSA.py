import streamlit as st
import pandas as pd
import plotly.express as px
from io import BytesIO
import time # For rate limiting geocoding requests

# Attempt to import geopy and streamlit_folium, provide instructions if missing
try:
    from geopy.geocoders import Nominatim
    from geopy.exc import GeocoderTimedOut, GeocoderUnavailable
except ImportError:
    st.error("The 'geopy' library is not installed. Please install it by running: pip install geopy")
    st.stop()

try:
    from streamlit_folium import st_folium
    import folium
except ImportError:
    st.error("The 'streamlit-folium' and 'folium' libraries are not installed. Please install them by running: pip install streamlit-folium folium")
    st.stop()

# --- Configuration ---
LOGO_URL = "https://www.esther.ie/wp-content/uploads/2022/05/HSE-Logo-Green-NEW-no-background.png"
ADDRESS_COLUMN_NAME = "Site Address " # IMPORTANT: Change if your address column has a different name
DATE_COLUMN_NAME = "Date of Inspection"
THEME_COLUMN_NAME = "Theme"
DIVISION_COLUMN_NAME = "Division "

# Page configuration
st.set_page_config(layout="wide", page_title="Spreadsheet Analysis Dashboard with Map")

# --- Session State Initialization ---
if 'df' not in st.session_state:
    st.session_state.df = None # Original uploaded dataframe
if 'geocoded_df' not in st.session_state:
    st.session_state.geocoded_df = None # DataFrame with latitude/longitude
if 'file_uploader_key' not in st.session_state:
    st.session_state.file_uploader_key = 0
if 'geocoding_done' not in st.session_state:
    st.session_state.geocoding_done = False


# --- Geocoding Function ---
@st.cache_data(show_spinner=False) # Cache results of geocoding for unique addresses
def geocode_address(address_string, attempt=1, max_attempts=3):
    """
    Converts an address string to latitude and longitude using Nominatim.
    Includes basic error handling, retries, and respects rate limits.
    """
    # It's good practice to specify a unique user_agent for Nominatim
    geolocator = Nominatim(user_agent=f"streamlit_dashboard_app_{time.time()}") # Unique user agent

    try:
        # IMPORTANT: Nominatim usage policy: max 1 request per second.
        time.sleep(1) # Wait 1 second before making the request
        location = geolocator.geocode(address_string, timeout=10) # 10 second timeout
        if location:
            return (location.latitude, location.longitude)
        else:
            return (None, None)
    except GeocoderTimedOut:
        if attempt <= max_attempts:
            st.warning(f"Geocoding timeout for '{address_string}', retrying ({attempt}/{max_attempts})...")
            time.sleep(1 + attempt) # Exponential backoff
            return geocode_address(address_string, attempt + 1, max_attempts)
        else:
            st.error(f"Failed to geocode '{address_string}' after {max_attempts} attempts due to timeout.")
            return (None, None)
    except GeocoderUnavailable as e:
        st.error(f"Geocoder service unavailable for '{address_string}': {e}")
        return (None, None)
    except Exception as e:
        st.error(f"An unexpected error occurred while geocoding '{address_string}': {e}")
        return (None, None)

def batch_geocode_dataframe(input_df, address_col):
    """
    Geocodes unique addresses from a DataFrame column and adds Latitude/Longitude.
    """
    if address_col not in input_df.columns:
        st.warning(f"Address column '{address_col}' not found. Skipping map generation.")
        return input_df

    unique_addresses = input_df[address_col].dropna().unique()
    if len(unique_addresses) == 0:
        st.info("No addresses found to geocode.")
        return input_df

    st.write(f"Found {len(unique_addresses)} unique addresses to geocode. This may take some time...")
    progress_bar = st.progress(0)
    
    # Use a dictionary to store geocoded results for unique addresses to avoid redundant API calls
    # This dictionary will be populated by the cached geocode_address function
    address_to_coords = {}
    
    for i, addr in enumerate(unique_addresses):
        if pd.isna(addr) or str(addr).strip() == "": # Skip if address is NaN or empty
            continue
        lat, lon = geocode_address(str(addr)) # Ensure address is a string
        address_to_coords[addr] = (lat, lon)
        progress_bar.progress((i + 1) / len(unique_addresses))

    st.success(f"Geocoding complete! Successfully geocoded {sum(1 for lat, lon in address_to_coords.values() if lat is not None)} addresses.")

    # Map coordinates back to the original DataFrame
    df_with_coords = input_df.copy()
    # Create temporary Series for mapping
    lat_series = df_with_coords[address_col].map(lambda x: address_to_coords.get(x, (None, None))[0])
    lon_series = df_with_coords[address_col].map(lambda x: address_to_coords.get(x, (None, None))[1])
    
    df_with_coords['Latitude'] = lat_series
    df_with_coords['Longitude'] = lon_series
    
    return df_with_coords

# --- Helper function to load data ---
def load_data(uploaded_file):
    """Loads data from uploaded file (Excel or CSV) into a pandas DataFrame."""
    try:
        if uploaded_file.name.endswith('.csv'):
            df = pd.read_csv(uploaded_file)
        elif uploaded_file.name.endswith(('.xls', '.xlsx')):
            df = pd.read_excel(uploaded_file)
        else:
            st.error("Unsupported file format. Please upload a CSV or Excel file.")
            return None
        return df
    except Exception as e:
        st.error(f"Error loading file: {e}")
        return None

# --- File Uploader ---
uploaded_file = st.file_uploader(
    "Upload your Spreadsheet (CSV or Excel)",
    type=["csv", "xls", "xlsx"],
    key=f"file_uploader_{st.session_state.file_uploader_key}"
)

if uploaded_file:
    if st.session_state.df is None or not hasattr(uploaded_file, '_uploaded_file_id') or \
       (hasattr(uploaded_file, '_uploaded_file_id') and st.session_state.get('_last_uploaded_file_id') != uploaded_file._uploaded_file_id):
        
        with st.spinner("Loading data..."):
            st.session_state.df = load_data(uploaded_file)
        
        if st.session_state.df is not None:
            st.success("Spreadsheet loaded successfully!")
            st.session_state.geocoding_done = False # Reset geocoding status for new file
            st.session_state.geocoded_df = None # Clear previous geocoded data

            if hasattr(uploaded_file, 'file_id'):
                 st.session_state._last_uploaded_file_id = uploaded_file.file_id
            else:
                 st.session_state._last_uploaded_file_id = uploaded_file.name + str(uploaded_file.size)
            
            # --- Automatic Geocoding after upload if Address column exists ---
            if ADDRESS_COLUMN_NAME in st.session_state.df.columns:
                with st.spinner(f"Geocoding addresses from '{ADDRESS_COLUMN_NAME}' column. This might take a while..."):
                    st.session_state.geocoded_df = batch_geocode_dataframe(st.session_state.df.copy(), ADDRESS_COLUMN_NAME)
                st.session_state.geocoding_done = True
            else:
                st.warning(f"Column '{ADDRESS_COLUMN_NAME}' not found in the uploaded file. Map functionality will be unavailable.")
                st.session_state.geocoded_df = st.session_state.df.copy() # Use original df if no address col
                st.session_state.geocoding_done = True # Mark as done to proceed

        else:
            st.session_state.df = None
            st.session_state.geocoded_df = None
            st.session_state.geocoding_done = False
            st.session_state.file_uploader_key += 1
            st.rerun()

# --- Main Application Logic ---
# Operates on geocoded_df if available and geocoding is done, otherwise on df
current_df_to_use = None
if st.session_state.geocoding_done:
    if st.session_state.geocoded_df is not None:
        current_df_to_use = st.session_state.geocoded_df.copy()
    elif st.session_state.df is not None: # Fallback if geocoded_df is None but geocoding was "done" (e.g. no address col)
        current_df_to_use = st.session_state.df.copy()

if current_df_to_use is not None:
    df_display = current_df_to_use # This df will be filtered and used for display

    st.image(LOGO_URL, width=200)
    st.markdown("---")

    # --- Column Checks (Date, Theme, Division) ---
    # These columns are still expected for the charts and other filters
    required_chart_cols = {DATE_COLUMN_NAME, THEME_COLUMN_NAME, DIVISION_COLUMN_NAME}
    missing_chart_cols = required_chart_cols - set(df_display.columns)

    if missing_chart_cols:
        st.warning(f"The uploaded spreadsheet is missing some expected columns for charts/filters: {', '.join(missing_chart_cols)}. Some features might not work as expected.")

    date_column_valid = False
    if DATE_COLUMN_NAME in df_display.columns:
        try:
            df_display[DATE_COLUMN_NAME] = pd.to_datetime(df_display[DATE_COLUMN_NAME], errors='coerce')
            if not df_display[DATE_COLUMN_NAME].isnull().all():
                df_display.dropna(subset=[DATE_COLUMN_NAME], inplace=True) # Keep only valid dates
                date_column_valid = True
            else:
                st.warning(f"Could not parse any dates in the '{DATE_COLUMN_NAME}' column.")
        except Exception as e:
            st.warning(f"Error converting '{DATE_COLUMN_NAME}' column: {e}.")
    
    # --- Sidebar Filters ---
    st.sidebar.header("Filters")

    selected_date_range = None
    if date_column_valid and not df_display[DATE_COLUMN_NAME].empty:
        min_date = df_display[DATE_COLUMN_NAME].min().date()
        max_date = df_display[DATE_COLUMN_NAME].max().date()
        try:
            selected_date_range = st.sidebar.date_input(
                "Select Date Range",
                value=(min_date, max_date),
                min_value=min_date,
                max_value=max_date,
                key="date_filter"
            )
            if len(selected_date_range) != 2: selected_date_range = (min_date, max_date)
        except Exception: selected_date_range = (min_date, max_date)
    else:
        st.sidebar.info(f"'{DATE_COLUMN_NAME}' column not available/valid for date filtering.")

    selected_themes = []
    if THEME_COLUMN_NAME in df_display.columns and not df_display[THEME_COLUMN_NAME].dropna().empty:
        unique_themes = sorted(df_display[THEME_COLUMN_NAME].dropna().astype(str).unique())
        selected_themes = st.sidebar.multiselect("Select Theme(s)", options=unique_themes, default=unique_themes, key="theme_filter")
    else:
        st.sidebar.info(f"'{THEME_COLUMN_NAME}' column not available for filtering.")

    selected_divisions = []
    if DIVISION_COLUMN_NAME in df_display.columns and not df_display[DIVISION_COLUMN_NAME].dropna().empty:
        unique_divisions = sorted(df_display[DIVISION_COLUMN_NAME].dropna().astype(str).unique())
        selected_divisions = st.sidebar.multiselect("Select Division(s)", options=unique_divisions, default=unique_divisions, key="division_filter")
    else:
        st.sidebar.info(f"'{DIVISION_COLUMN_NAME}' column not available for filtering.")

    # --- Filtering Logic ---
    filtered_df_display = df_display.copy()

    if selected_date_range and date_column_valid and len(selected_date_range) == 2:
        start_date = pd.to_datetime(selected_date_range[0])
        end_date = pd.to_datetime(selected_date_range[1])
        filtered_df_display = filtered_df_display[(filtered_df_display[DATE_COLUMN_NAME] >= start_date) & (filtered_df_display[DATE_COLUMN_NAME] <= end_date)]

    if selected_themes and THEME_COLUMN_NAME in filtered_df_display.columns:
        filtered_df_display = filtered_df_display[filtered_df_display[THEME_COLUMN_NAME].isin(selected_themes)]

    if selected_divisions and DIVISION_COLUMN_NAME in filtered_df_display.columns:
        filtered_df_display = filtered_df_display[filtered_df_display[DIVISION_COLUMN_NAME].isin(selected_divisions)]

    # --- Main Area Display ---
    st.header("Dashboard Overview")
    num_observations = len(filtered_df_display)
    st.metric(label="Number of Observations (after filters)", value=num_observations)
    st.markdown("---")

    # Charts
    col1, col2 = st.columns(2)
    with col1:
        st.subheader(f"Observations by {THEME_COLUMN_NAME}")
        if not filtered_df_display.empty and THEME_COLUMN_NAME in filtered_df_display.columns and not filtered_df_display[THEME_COLUMN_NAME].dropna().empty:
            theme_counts = filtered_df_display[THEME_COLUMN_NAME].value_counts().reset_index()
            theme_counts.columns = [THEME_COLUMN_NAME, 'Count']
            fig_theme = px.bar(theme_counts, x=THEME_COLUMN_NAME, y='Count', title=f"{THEME_COLUMN_NAME} Distribution", color=THEME_COLUMN_NAME)
            st.plotly_chart(fig_theme, use_container_width=True)
        else:
            st.info(f"No data to display for {THEME_COLUMN_NAME} based on current filters or column not found.")

    with col2:
        st.subheader(f"Observations by {DIVISION_COLUMN_NAME}")
        if not filtered_df_display.empty and DIVISION_COLUMN_NAME in filtered_df_display.columns and not filtered_df_display[DIVISION_COLUMN_NAME].dropna().empty:
            division_counts = filtered_df_display[DIVISION_COLUMN_NAME].value_counts().reset_index()
            division_counts.columns = [DIVISION_COLUMN_NAME, 'Count']
            fig_division = px.bar(division_counts, x=DIVISION_COLUMN_NAME, y='Count', title=f"{DIVISION_COLUMN_NAME} Distribution", color=DIVISION_COLUMN_NAME)
            st.plotly_chart(fig_division, use_container_width=True)
        else:
            st.info(f"No data to display for {DIVISION_COLUMN_NAME} based on current filters or column not found.")
    
    st.markdown("---")

    # --- Folium Map Display ---
    st.header("Location Map")
    if 'Latitude' in filtered_df_display.columns and 'Longitude' in filtered_df_display.columns:
        # Filter out rows where geocoding might have failed
        map_df = filtered_df_display.dropna(subset=['Latitude', 'Longitude'])
        
        if not map_df.empty:
            # Calculate a central point for the map
            # Using the mean of coordinates of the filtered data points
            map_center_lat = map_df['Latitude'].mean()
            map_center_lon = map_df['Longitude'].mean()
            
            # Create Folium map
            m = folium.Map(location=[map_center_lat, map_center_lon], zoom_start=6) # Adjust zoom as needed

            # Add markers for each geocoded address in the filtered data
            # Using a subset for performance if there are too many points, e.g., first 1000
            # For a large number of points, consider MarkerCluster: from folium.plugins import MarkerCluster
            # marker_cluster = MarkerCluster().add_to(m)

            points_to_plot = map_df.head(1000) # Limit points for performance, adjust as needed
            if len(map_df) > 1000:
                st.info(f"Displaying the first 1000 out of {len(map_df)} geocoded points on the map for performance.")

            for idx, row in points_to_plot.iterrows():
                popup_text = f"Address: {row[ADDRESS_COLUMN_NAME]}"
                if THEME_COLUMN_NAME in row:
                    popup_text += f"<br>Theme: {row[THEME_COLUMN_NAME]}"
                if DIVISION_COLUMN_NAME in row:
                    popup_text += f"<br>Division: {row[DIVISION_COLUMN_NAME]}"
                
                folium.Marker(
                    location=[row['Latitude'], row['Longitude']],
                    popup=folium.Popup(popup_text, max_width=300),
                    tooltip=str(row[ADDRESS_COLUMN_NAME])
                ).add_to(m) # or add_to(marker_cluster) if using MarkerCluster

            # Display map
            st_folium(m, width='100%', height=500)
        elif ADDRESS_COLUMN_NAME not in df_display.columns:
             st.info(f"The '{ADDRESS_COLUMN_NAME}' column was not found in your data, so the map cannot be generated.")
        elif not filtered_df_display.empty and map_df.empty:
            st.warning("No valid geocoded locations to display on the map for the current filter selection, or all addresses failed to geocode.")
        else:
            st.info("Upload data and apply filters to see locations on the map.")
    else:
        if ADDRESS_COLUMN_NAME in df_display.columns:
            st.info("Addresses are being geocoded or no valid coordinates found after geocoding. Map will appear once processing is complete and data is available.")
        else:
            st.info(f"To see a map, please ensure your spreadsheet has an '{ADDRESS_COLUMN_NAME}' column and contains valid addresses.")


    # Optional: Display filtered data table
    if st.checkbox("Show Filtered Data Table", key="show_data_table"):
        st.subheader("Filtered Data (with Latitude/Longitude if available)")
        st.dataframe(filtered_df_display, use_container_width=True)

else:
    st.info("Please upload a spreadsheet to begin analysis.")
    st.markdown(f"""
        ### Welcome to the Spreadsheet Analysis Dashboard!
        - Use the uploader above to load your data (CSV or Excel).
        - Ensure your file contains '{DATE_COLUMN_NAME}', '{THEME_COLUMN_NAME}', and '{DIVISION_COLUMN_NAME}' columns for charts and filters.
        - For map functionality, include an '{ADDRESS_COLUMN_NAME}' column with addresses.
        - Geocoding will happen automatically for unique addresses (1 request/second, so it can be slow for many unique addresses).
    """)

# --- Add a button to clear uploaded file and reset ---
if st.session_state.df is not None or st.session_state.geocoded_df is not None:
    if st.sidebar.button("Clear Data and Reset", key="clear_data"):
        st.session_state.df = None
        st.session_state.geocoded_df = None
        st.session_state.geocoding_done = False
        st.session_state.file_uploader_key += 1
        # Clear other session state variables related to filters if necessary
        keys_to_clear = ["date_filter", "theme_filter", "division_filter", "show_data_table", "_last_uploaded_file_id"]
        for key in keys_to_clear:
            if key in st.session_state:
                del st.session_state[key]
        # Clear geopy cache if you want to force re-geocoding on next identical upload (optional)
        # geocode_address.clear() # Uncomment if you want to clear the geocoding cache on reset
        st.rerun()

