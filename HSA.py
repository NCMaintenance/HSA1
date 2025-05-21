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
# Column names are expected AFTER stripping whitespace
ADDRESS_COLUMN_NAME = "Site Address"
DATE_COLUMN_NAME = "Date of Inspection"
THEME_COLUMN_NAME = "Theme"
DIVISION_COLUMN_NAME = "Division"

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
if 'lookup_df' not in st.session_state:
    st.session_state.lookup_df = None # For storing uploaded lookup table
if 'geocoded_pairs_for_download' not in st.session_state:
    st.session_state.geocoded_pairs_for_download = {} # Stores {address: (lat,lon)} for download

# --- Geocoding Function ---
@st.cache_data(show_spinner=False) # Cache results of geocoding for unique addresses (API calls)
def geocode_address_via_api(address_string, attempt=1, max_attempts=3):
    """
    Converts an address string to latitude and longitude using Nominatim API.
    Includes basic error handling, retries, and respects rate limits.
    """
    geolocator = Nominatim(user_agent=f"streamlit_dashboard_app_{int(time.time())}") # More unique user agent

    try:
        time.sleep(1) # IMPORTANT: Nominatim usage policy: max 1 request per second.
        location = geolocator.geocode(address_string, timeout=10)
        if location:
            return (location.latitude, location.longitude)
        else:
            return (None, None)
    except GeocoderTimedOut:
        if attempt <= max_attempts:
            st.warning(f"API Geocoding timeout for '{address_string}', retrying ({attempt}/{max_attempts})...")
            time.sleep(1 + attempt)
            return geocode_address_via_api(address_string, attempt + 1, max_attempts)
        else:
            st.error(f"API Geocoding failed for '{address_string}' after {max_attempts} attempts due to timeout.")
            return (None, None)
    except GeocoderUnavailable as e:
        st.error(f"API Geocoder service unavailable for '{address_string}': {e}")
        return (None, None)
    except Exception as e:
        st.error(f"An unexpected error occurred during API geocoding for '{address_string}': {e}")
        return (None, None)

def batch_geocode_dataframe(input_df, address_col, lookup_data=None):
    """
    Geocodes unique addresses from a DataFrame column.
    Uses lookup_data first, then falls back to API geocoding.
    Returns the DataFrame with 'Latitude' and 'Longitude' columns, and a dict of geocoded pairs.
    """
    if address_col not in input_df.columns:
        st.warning(f"Address column '{address_col}' not found. Skipping map generation.")
        return input_df, {}

    unique_addresses_in_df = input_df[address_col].dropna().astype(str).unique()
    if len(unique_addresses_in_df) == 0:
        st.info("No addresses found to geocode in the main file.")
        return input_df, {}

    st.write(f"Found {len(unique_addresses_in_df)} unique addresses in the main file to process...")
    progress_bar = st.progress(0)
    
    address_to_coords = {}
    api_calls_made = 0
    found_in_lookup = 0

    # Prepare lookup dictionary if lookup_data is provided
    lookup_dict = {}
    if lookup_data is not None and not lookup_data.empty:
        if address_col in lookup_data.columns and 'Latitude' in lookup_data.columns and 'Longitude' in lookup_data.columns:
            # Ensure address column in lookup_dict is string for matching
            lookup_data_copy = lookup_data.copy()
            lookup_data_copy[address_col] = lookup_data_copy[address_col].astype(str)
            lookup_dict = pd.Series(
                list(zip(lookup_data_copy['Latitude'], lookup_data_copy['Longitude'])),
                index=lookup_data_copy[address_col]
            ).to_dict()
            st.info(f"Using provided lookup table with {len(lookup_dict)} entries.")
        else:
            st.warning("Lookup table is missing required columns (Address, Latitude, Longitude). It will not be used.")

    for i, addr_str in enumerate(unique_addresses_in_df):
        if not addr_str or pd.isna(addr_str): # Skip if address is empty or NaN
            address_to_coords[addr_str] = (None, None)
            continue

        # 1. Try to find in lookup_dict
        if addr_str in lookup_dict:
            lat, lon = lookup_dict[addr_str]
            # Check if lat/lon from lookup are valid numbers
            if pd.notna(lat) and pd.notna(lon):
                 address_to_coords[addr_str] = (float(lat), float(lon))
                 found_in_lookup +=1
            else: # If lookup has invalid coords for this address, try API
                 address_to_coords[addr_str] = geocode_address_via_api(addr_str)
                 if address_to_coords[addr_str] != (None, None): api_calls_made += 1
        else:
            # 2. If not in lookup, geocode via API
            address_to_coords[addr_str] = geocode_address_via_api(addr_str)
            if address_to_coords[addr_str] != (None, None): api_calls_made += 1
        
        progress_bar.progress((i + 1) / len(unique_addresses_in_df))

    success_count = sum(1 for lat, lon in address_to_coords.values() if lat is not None and lon is not None)
    st.success(f"Geocoding process complete! Successfully processed {success_count} addresses.")
    if lookup_dict: st.info(f"{found_in_lookup} addresses found in the lookup table.")
    st.info(f"{api_calls_made} calls made to Nominatim API.")

    df_with_coords = input_df.copy()
    df_with_coords['Latitude'] = df_with_coords[address_col].astype(str).map(lambda x: address_to_coords.get(x, (None, None))[0])
    df_with_coords['Longitude'] = df_with_coords[address_col].astype(str).map(lambda x: address_to_coords.get(x, (None, None))[1])
    
    return df_with_coords, address_to_coords

# --- Helper function to load data ---
def load_data(uploaded_file_obj):
    """Loads data from uploaded file (Excel or CSV) into a pandas DataFrame.
       Also strips whitespace from column names.
    """
    df = None
    try:
        if uploaded_file_obj.name.endswith('.csv'):
            df = pd.read_csv(uploaded_file_obj)
        elif uploaded_file_obj.name.endswith(('.xls', '.xlsx')):
            df = pd.read_excel(uploaded_file_obj)
        else:
            st.error("Unsupported file format. Please upload a CSV or Excel file.")
            return None
        
        if df is not None:
            df.columns = df.columns.str.strip() # Strip whitespace from column names
        return df
    except Exception as e:
        st.error(f"Error loading file: {e}")
        return None

# --- File Uploader for Main Data ---
uploaded_file = st.file_uploader(
    "Upload your Main Spreadsheet (CSV or Excel)",
    type=["csv", "xls", "xlsx"],
    key=f"file_uploader_{st.session_state.file_uploader_key}"
)

# --- Sidebar: Lookup File Uploader ---
st.sidebar.subheader("Geocoding Lookup Table (Optional)")
uploaded_lookup_file = st.sidebar.file_uploader(
    "Upload Geocoded Lookup File (CSV/Excel)",
    type=["csv", "xls", "xlsx"],
    key="lookup_uploader",
    help=f"Must contain columns: '{ADDRESS_COLUMN_NAME}', 'Latitude', 'Longitude'"
)

if uploaded_lookup_file:
    # Check if it's a new lookup file or the same one to avoid reprocessing
    # This simple check might not be robust for all cases but works for basic re-upload scenarios
    if st.session_state.lookup_df is None or not hasattr(uploaded_lookup_file, '_uploaded_file_id_lookup') or \
       (hasattr(uploaded_lookup_file, '_uploaded_file_id_lookup') and st.session_state.get('_last_lookup_file_id') != uploaded_lookup_file._uploaded_file_id_lookup):
        with st.spinner("Loading lookup table..."):
            temp_lookup_df = load_data(uploaded_lookup_file) # Use the same load_data to strip columns
        if temp_lookup_df is not None:
            # Validate lookup_df
            required_lookup_cols = {ADDRESS_COLUMN_NAME, 'Latitude', 'Longitude'}
            if required_lookup_cols.issubset(temp_lookup_df.columns):
                st.session_state.lookup_df = temp_lookup_df
                st.sidebar.success("Lookup table loaded successfully.")
                st.session_state.geocoding_done = False # Trigger re-geocoding if main file is already loaded
                if hasattr(uploaded_lookup_file, 'file_id'):
                    st.session_state._last_lookup_file_id = uploaded_lookup_file.file_id
                else: # Fallback for older versions or different upload objects
                    st.session_state._last_lookup_file_id = uploaded_lookup_file.name + str(uploaded_lookup_file.size)
                # Add an attribute to the file object itself to help track if it's the "same" object instance
                setattr(uploaded_lookup_file, '_uploaded_file_id_lookup', st.session_state._last_lookup_file_id)

            else:
                st.sidebar.error(f"Lookup file must contain columns: {', '.join(required_lookup_cols)}.")
                st.session_state.lookup_df = None # Invalidate if incorrect
        else:
            st.session_state.lookup_df = None # Invalidate if loading failed
elif 'lookup_uploader' in st.session_state and st.session_state.lookup_uploader is None: # If user clears the lookup uploader
    if st.session_state.lookup_df is not None:
        st.session_state.lookup_df = None
        st.sidebar.info("Lookup table cleared.")
        st.session_state.geocoding_done = False # Trigger re-geocoding

# --- Process Main Uploaded File ---
if uploaded_file:
    # Check if this is a new file or the same one
    if st.session_state.df is None or not hasattr(uploaded_file, '_uploaded_file_id_main') or \
       (hasattr(uploaded_file, '_uploaded_file_id_main') and st.session_state.get('_last_uploaded_main_file_id') != uploaded_file._uploaded_file_id_main):
        
        with st.spinner("Loading main data..."):
            st.session_state.df = load_data(uploaded_file)
        
        if st.session_state.df is not None:
            st.success("Main spreadsheet loaded successfully!")
            st.session_state.geocoding_done = False # Reset geocoding status for new file
            st.session_state.geocoded_df = None
            st.session_state.geocoded_pairs_for_download = {}

            if hasattr(uploaded_file, 'file_id'):
                 st.session_state._last_uploaded_main_file_id = uploaded_file.file_id
            else:
                 st.session_state._last_uploaded_main_file_id = uploaded_file.name + str(uploaded_file.size)
            setattr(uploaded_file, '_uploaded_file_id_main', st.session_state._last_uploaded_main_file_id)

        else: # Main file loading failed
            st.session_state.df = None
            st.session_state.geocoded_df = None
            st.session_state.geocoding_done = False
            st.session_state.file_uploader_key += 1 # Attempt to reset uploader
            st.rerun()

# --- Perform Geocoding if main data is loaded and geocoding not yet done ---
if st.session_state.df is not None and not st.session_state.geocoding_done:
    if ADDRESS_COLUMN_NAME in st.session_state.df.columns:
        with st.spinner(f"Processing addresses from '{ADDRESS_COLUMN_NAME}' column..."):
            df_with_coords, geocoded_pairs = batch_geocode_dataframe(
                st.session_state.df.copy(),
                ADDRESS_COLUMN_NAME,
                st.session_state.lookup_df # Pass the loaded lookup_df
            )
            st.session_state.geocoded_df = df_with_coords
            st.session_state.geocoded_pairs_for_download = geocoded_pairs
        st.session_state.geocoding_done = True
    else:
        st.warning(f"Column '{ADDRESS_COLUMN_NAME}' not found in the uploaded file. Map functionality will be unavailable.")
        st.session_state.geocoded_df = st.session_state.df.copy() # Use original df
        st.session_state.geocoding_done = True # Mark as done to proceed

# --- Main Application Logic (operates on geocoded_df) ---
current_df_to_use = None
if st.session_state.geocoding_done and st.session_state.geocoded_df is not None:
    current_df_to_use = st.session_state.geocoded_df.copy()

if current_df_to_use is not None:
    df_display = current_df_to_use

    st.image(LOGO_URL, width=200)
    st.markdown("---")

    required_cols_for_app = {DATE_COLUMN_NAME, THEME_COLUMN_NAME, DIVISION_COLUMN_NAME}
    if ADDRESS_COLUMN_NAME in df_display.columns: # Address is only needed if map is desired
        required_cols_for_app.add(ADDRESS_COLUMN_NAME)
    
    missing_cols = required_cols_for_app - set(df_display.columns)
    if missing_cols:
        st.warning(f"The spreadsheet is missing some expected columns: {', '.join(missing_cols)}. Some features might not work.")

    date_column_valid = False
    if DATE_COLUMN_NAME in df_display.columns:
        try:
            # Ensure it's not all NaNs before attempting conversion
            if not df_display[DATE_COLUMN_NAME].isnull().all():
                df_display[DATE_COLUMN_NAME] = pd.to_datetime(df_display[DATE_COLUMN_NAME], errors='coerce')
                if not df_display[DATE_COLUMN_NAME].isnull().all(): # Check again after coerce
                    df_display.dropna(subset=[DATE_COLUMN_NAME], inplace=True)
                    if not df_display.empty: # Check if df is empty after dropna
                         date_column_valid = True
                    else:
                        st.warning(f"All rows were removed after attempting to parse dates in '{DATE_COLUMN_NAME}'. No valid dates found.")
                else:
                    st.warning(f"Could not parse any dates in the '{DATE_COLUMN_NAME}' column (all became NaT).")
            else:
                st.info(f"'{DATE_COLUMN_NAME}' column is present but contains all empty values.")
        except Exception as e:
            st.warning(f"Error processing '{DATE_COLUMN_NAME}' column: {e}. Date filtering may be unavailable.")
    
    # --- Sidebar Filters ---
    st.sidebar.header("Filters")

    # Date Filter
    selected_date_range = None
    if date_column_valid and not df_display[DATE_COLUMN_NAME].empty:
        min_dt_val = df_display[DATE_COLUMN_NAME].min()
        max_dt_val = df_display[DATE_COLUMN_NAME].max()

        if pd.isna(min_dt_val) or pd.isna(max_dt_val):
            st.sidebar.warning(f"Could not determine a valid date range from '{DATE_COLUMN_NAME}'.")
        else:
            min_date = min_dt_val.date()
            max_date = max_dt_val.date()
            
            default_start_date, default_end_date = min_date, max_date
            if min_date > max_date: # Should not happen with proper data
                st.sidebar.warning("Min date is after max date in the data. Using swapped values for filter.")
                default_start_date, default_end_date = max_date, min_date
            
            try:
                selected_date_range = st.sidebar.date_input(
                    "Select Date Range",
                    value=(default_start_date, default_end_date),
                    min_value=min_date, # Actual min from data
                    max_value=max_date, # Actual max from data
                    key="date_filter"
                )
                if len(selected_date_range) != 2:
                    selected_date_range = (default_start_date, default_end_date)
            except Exception as e:
                st.sidebar.error(f"Error initializing date filter: {e}")
                selected_date_range = (default_start_date, default_end_date)
    else:
        st.sidebar.info(f"'{DATE_COLUMN_NAME}' column not available/valid for date filtering.")

    # Division Filter (now before Theme)
    selected_divisions = []
    if DIVISION_COLUMN_NAME in df_display.columns and not df_display[DIVISION_COLUMN_NAME].dropna().empty:
        unique_divisions = sorted(df_display[DIVISION_COLUMN_NAME].dropna().astype(str).unique())
        selected_divisions = st.sidebar.multiselect(f"Select {DIVISION_COLUMN_NAME}(s)", options=unique_divisions, default=unique_divisions, key="division_filter")
    else:
        st.sidebar.info(f"'{DIVISION_COLUMN_NAME}' column not available for filtering.")

    # Theme Filter
    selected_themes = []
    if THEME_COLUMN_NAME in df_display.columns and not df_display[THEME_COLUMN_NAME].dropna().empty:
        unique_themes = sorted(df_display[THEME_COLUMN_NAME].dropna().astype(str).unique())
        selected_themes = st.sidebar.multiselect(f"Select {THEME_COLUMN_NAME}(s)", options=unique_themes, default=unique_themes, key="theme_filter")
    else:
        st.sidebar.info(f"'{THEME_COLUMN_NAME}' column not available for filtering.")

    # --- Filtering Logic ---
    filtered_df_display = df_display.copy()

    if selected_date_range and date_column_valid and len(selected_date_range) == 2:
        start_date = pd.to_datetime(selected_date_range[0])
        end_date = pd.to_datetime(selected_date_range[1])
        # Ensure comparison is between datetime objects
        filtered_df_display = filtered_df_display[
            (pd.to_datetime(filtered_df_display[DATE_COLUMN_NAME]) >= start_date) &
            (pd.to_datetime(filtered_df_display[DATE_COLUMN_NAME]) <= end_date)
        ]
    
    if selected_divisions and DIVISION_COLUMN_NAME in filtered_df_display.columns:
        filtered_df_display = filtered_df_display[filtered_df_display[DIVISION_COLUMN_NAME].isin(selected_divisions)]
    
    if selected_themes and THEME_COLUMN_NAME in filtered_df_display.columns:
        filtered_df_display = filtered_df_display[filtered_df_display[THEME_COLUMN_NAME].isin(selected_themes)]

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
        map_df = filtered_df_display.dropna(subset=['Latitude', 'Longitude'])
        
        if not map_df.empty:
            map_center_lat = map_df['Latitude'].mean()
            map_center_lon = map_df['Longitude'].mean()
            m = folium.Map(location=[map_center_lat, map_center_lon], zoom_start=6)

            points_to_plot = map_df.head(1000)
            if len(map_df) > 1000:
                st.info(f"Displaying the first 1000 out of {len(map_df)} geocoded points on the map for performance.")

            for idx, row in points_to_plot.iterrows():
                popup_text = f"<b>{ADDRESS_COLUMN_NAME}:</b> {row[ADDRESS_COLUMN_NAME]}"
                if THEME_COLUMN_NAME in row and pd.notna(row[THEME_COLUMN_NAME]):
                    popup_text += f"<br><b>{THEME_COLUMN_NAME}:</b> {row[THEME_COLUMN_NAME]}"
                if DIVISION_COLUMN_NAME in row and pd.notna(row[DIVISION_COLUMN_NAME]):
                    popup_text += f"<br><b>{DIVISION_COLUMN_NAME}:</b> {row[DIVISION_COLUMN_NAME]}"
                
                folium.Marker(
                    location=[row['Latitude'], row['Longitude']],
                    popup=folium.Popup(popup_text, max_width=300),
                    tooltip=str(row.get(ADDRESS_COLUMN_NAME, "N/A"))
                ).add_to(m)
            st_folium(m, width='100%', height=500, key="folium_map")
        elif ADDRESS_COLUMN_NAME not in df_display.columns:
             st.info(f"The '{ADDRESS_COLUMN_NAME}' column was not found, so the map cannot be generated.")
        elif not filtered_df_display.empty and map_df.empty: # Data exists, but no valid coords after filtering/geocoding
            st.warning("No valid geocoded locations to display on the map for the current filter selection, or all addresses failed/yielded no coordinates.")
        else: # No data after filtering, or initial state
            st.info("Apply filters or upload data with addresses to see locations on the map.")
    else: # Latitude/Longitude columns don't exist (e.g., geocoding not run or failed, or no address col)
        if ADDRESS_COLUMN_NAME in df_display.columns:
             st.info("Map will appear once addresses are processed and geocoded successfully.")
        else:
            st.info(f"To see a map, please ensure your spreadsheet has an '{ADDRESS_COLUMN_NAME}' column.")

    # --- Download Button for Geocoded Pairs ---
    if st.session_state.get('geocoded_pairs_for_download'):
        # Prepare data for download
        dl_df_data = []
        for addr, (lat, lon) in st.session_state.geocoded_pairs_for_download.items():
            if lat is not None and lon is not None: # Only include successfully geocoded pairs
                dl_df_data.append({ADDRESS_COLUMN_NAME: addr, 'Latitude': lat, 'Longitude': lon})
        
        if dl_df_data:
            downloadable_df = pd.DataFrame(dl_df_data)
            csv_export = downloadable_df.to_csv(index=False).encode('utf-8')
            st.sidebar.download_button(
                label="Download Geocoded Addresses (CSV)",
                data=csv_export,
                file_name="geocoded_addresses_export.csv",
                mime="text/csv",
                key="download_geocoded_csv"
            )
        else:
            st.sidebar.info("No successfully geocoded addresses to download from the current main file.")


    # Optional: Display filtered data table
    if st.checkbox("Show Filtered Data Table", key="show_data_table"):
        st.subheader("Filtered Data (with Latitude/Longitude if available)")
        st.dataframe(filtered_df_display, use_container_width=True)

elif not uploaded_file : # Only show welcome if no main file is uploaded yet
    st.info("Please upload a main spreadsheet to begin analysis.")
    st.markdown(f"""
        ### Welcome to the Spreadsheet Analysis Dashboard!
        - Use the uploader above to load your main data (CSV or Excel).
        - Optionally, upload a **Geocoding Lookup Table** in the sidebar to speed up geocoding.
        - Ensure your main file contains '{DATE_COLUMN_NAME}', '{THEME_COLUMN_NAME}', and '{DIVISION_COLUMN_NAME}' columns for charts and filters.
        - For map functionality, include an '{ADDRESS_COLUMN_NAME}' column.
        - Geocoding will use the lookup table first, then the Nominatim API (1 request/second).
    """)

# --- Reset Button ---
if st.session_state.df is not None or st.session_state.geocoded_df is not None:
    if st.sidebar.button("Clear All Data and Reset App", key="clear_data"):
        # Clear main data related states
        st.session_state.df = None
        st.session_state.geocoded_df = None
        st.session_state.geocoding_done = False
        st.session_state.geocoded_pairs_for_download = {}
        st.session_state.file_uploader_key += 1 # Resets the main file uploader
        
        # Clear lookup data related states
        st.session_state.lookup_df = None
        if 'lookup_uploader' in st.session_state: # Reset lookup file uploader by clearing its state
            st.session_state.lookup_uploader = None 
        if '_last_lookup_file_id' in st.session_state:
            del st.session_state['_last_lookup_file_id']


        # Clear filter states and other UI component states by their keys
        keys_to_clear_from_session = [
            "date_filter", "theme_filter", "division_filter", 
            "show_data_table", "_last_uploaded_main_file_id",
            "folium_map" # Clear map state if needed
        ]
        for key in keys_to_clear_from_session:
            if key in st.session_state:
                del st.session_state[key]
        
        # Clear geopy API cache (optional, if you want to force re-geocoding from API on next identical addresses)
        # geocode_address_via_api.clear()
        st.success("Application reset. Please upload new files.")
        st.rerun()
