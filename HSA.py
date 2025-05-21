import streamlit as st
import pandas as pd
import plotly.express as px
from io import BytesIO
import time
import datetime

# --- Imports for geocoding and maps ---
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
ADDRESS_COLUMN_NAME = "Site Address"
DATE_COLUMN_NAME = "Date of Inspection"
THEME_COLUMN_NAME = "Theme"
DIVISION_COLUMN_NAME = "Division"

st.set_page_config(layout="wide", page_title="Spreadsheet Analysis Dashboard with Map")

# --- Session State Initialization ---
if 'df' not in st.session_state:
    st.session_state.df = None
if 'geocoded_df' not in st.session_state:
    st.session_state.geocoded_df = None
if 'file_uploader_key' not in st.session_state:
    st.session_state.file_uploader_key = 0
if 'geocoding_done' not in st.session_state:
    st.session_state.geocoding_done = False
if 'lookup_df' not in st.session_state:
    st.session_state.lookup_df = None
if 'geocoded_pairs_for_download' not in st.session_state:
    st.session_state.geocoded_pairs_for_download = {}

# --- Geocoding Function ---
@st.cache_data(show_spinner=False)
def geocode_address_via_api(address_string, attempt=1, max_attempts=3):
    geolocator = Nominatim(user_agent=f"streamlit_dashboard_app_{int(time.time())}")
    try:
        time.sleep(1)
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
    if address_col not in input_df.columns:
        st.warning(f"Address column '{address_col}' not found. Skipping map generation.")
        return input_df, {}

    unique_addresses_in_df = input_df[address_col].dropna().astype(str).unique()
    if len(unique_addresses_in_df) == 0:
        st.info("No valid (non-empty) addresses found to geocode in the main file.")
        return input_df, {}

    st.write(f"Found {len(unique_addresses_in_df)} unique addresses in the main file to process...")
    progress_bar = st.progress(0)
    address_to_coords = {}
    api_calls_made = 0
    found_in_lookup = 0

    lookup_dict = {}
    if lookup_data is not None and not lookup_data.empty:
        if address_col in lookup_data.columns and 'Latitude' in lookup_data.columns and 'Longitude' in lookup_data.columns:
            lookup_data_copy = lookup_data.copy()
            lookup_data_copy[address_col] = lookup_data_copy[address_col].astype(str)
            lookup_data_copy.dropna(subset=[address_col, 'Latitude', 'Longitude'], inplace=True)
            lookup_dict = pd.Series(
                list(zip(lookup_data_copy['Latitude'], lookup_data_copy['Longitude'])),
                index=lookup_data_copy[address_col]
            ).to_dict()
            st.info(f"Using provided lookup table with {len(lookup_dict)} valid entries.")
        else:
            st.warning(f"Lookup table is missing required columns ('{address_col}', 'Latitude', 'Longitude'). It will not be used.")

    for i, addr_str in enumerate(unique_addresses_in_df):
        if addr_str in lookup_dict:
            lat, lon = lookup_dict[addr_str]
            if pd.notna(lat) and pd.notna(lon):
                address_to_coords[addr_str] = (float(lat), float(lon))
                found_in_lookup += 1
            else:
                address_to_coords[addr_str] = geocode_address_via_api(addr_str)
                if address_to_coords[addr_str] != (None, None): api_calls_made += 1
        else:
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

def load_data(uploaded_file_obj):
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
            df.columns = df.columns.str.strip()
        return df
    except Exception as e:
        st.error(f"Error loading file '{uploaded_file_obj.name}': {e}")
        return None

uploaded_file = st.file_uploader(
    "Upload your Main Spreadsheet (CSV or Excel)",
    type=["csv", "xls", "xlsx"],
    key=f"file_uploader_{st.session_state.file_uploader_key}"
)

st.sidebar.subheader("Geocoding Lookup Table (Optional)")
uploaded_lookup_file = st.sidebar.file_uploader(
    "Upload Geocoded Lookup File (CSV/Excel)",
    type=["csv", "xls", "xlsx"],
    key="lookup_uploader",
    help=f"Must contain columns: '{ADDRESS_COLUMN_NAME}', 'Latitude', 'Longitude'"
)

if uploaded_lookup_file:
    current_lookup_file_id = getattr(uploaded_lookup_file, 'file_id', uploaded_lookup_file.name + str(uploaded_lookup_file.size))
    if st.session_state.lookup_df is None or not hasattr(uploaded_lookup_file, '_uploaded_file_id_lookup') or \
       (hasattr(uploaded_lookup_file, '_uploaded_file_id_lookup') and st.session_state.get('_last_lookup_file_id') != current_lookup_file_id):
        with st.spinner("Loading lookup table..."):
            temp_lookup_df = load_data(uploaded_lookup_file)
        if temp_lookup_df is not None:
            required_lookup_cols = {ADDRESS_COLUMN_NAME, 'Latitude', 'Longitude'}
            if required_lookup_cols.issubset(temp_lookup_df.columns):
                st.session_state.lookup_df = temp_lookup_df
                st.sidebar.success("Lookup table loaded successfully.")
                st.session_state.geocoding_done = False 
                st.session_state._last_lookup_file_id = current_lookup_file_id
                setattr(uploaded_lookup_file, '_uploaded_file_id_lookup', current_lookup_file_id)
            else:
                st.sidebar.error(f"Lookup file must contain columns: {', '.join(required_lookup_cols)}.")
                st.session_state.lookup_df = None
        else:
            st.session_state.lookup_df = None
elif 'lookup_uploader' in st.session_state and st.session_state.lookup_uploader is None: 
    if st.session_state.lookup_df is not None:
        st.session_state.lookup_df = None
        st.sidebar.info("Lookup table cleared.")
        st.session_state.geocoding_done = False

if uploaded_file:
    current_main_file_id = getattr(uploaded_file, 'file_id', uploaded_file.name + str(uploaded_file.size))
    if st.session_state.df is None or not hasattr(uploaded_file, '_uploaded_file_id_main') or \
       (hasattr(uploaded_file, '_uploaded_file_id_main') and st.session_state.get('_last_uploaded_main_file_id') != current_main_file_id):
        with st.spinner("Loading main data..."):
            st.session_state.df = load_data(uploaded_file)
        if st.session_state.df is not None:
            st.success("Main spreadsheet loaded successfully!")
            st.session_state.geocoding_done = False
            st.session_state.geocoded_df = None
            st.session_state.geocoded_pairs_for_download = {}
            st.session_state._last_uploaded_main_file_id = current_main_file_id
            setattr(uploaded_file, '_uploaded_file_id_main', current_main_file_id)
        else: 
            st.session_state.df = None
            st.session_state.geocoded_df = None
            st.session_state.geocoding_done = False
            st.session_state.file_uploader_key += 1
            st.rerun()

if st.session_state.df is not None and not st.session_state.geocoding_done:
    if ADDRESS_COLUMN_NAME in st.session_state.df.columns:
        with st.spinner(f"Processing addresses from '{ADDRESS_COLUMN_NAME}' column..."):
            df_with_coords, geocoded_pairs = batch_geocode_dataframe(
                st.session_state.df.copy(),
                ADDRESS_COLUMN_NAME,
                st.session_state.lookup_df
            )
            st.session_state.geocoded_df = df_with_coords
            st.session_state.geocoded_pairs_for_download = geocoded_pairs
        st.session_state.geocoding_done = True
    else:
        st.warning(f"Column '{ADDRESS_COLUMN_NAME}' not found in the main file. Map functionality will be unavailable.")
        st.session_state.geocoded_df = st.session_state.df.copy()
        st.session_state.geocoding_done = True

current_df_to_use = None
if st.session_state.geocoding_done and st.session_state.geocoded_df is not None:
    current_df_to_use = st.session_state.geocoded_df.copy()

if current_df_to_use is not None:
    df_display = current_df_to_use.copy()

    st.image(LOGO_URL, width=200)
    st.markdown("---")

    date_column_valid = False
    min_date_for_filter = None
    max_date_for_filter = None

    if DATE_COLUMN_NAME in df_display.columns:
        date_series_original = df_display[DATE_COLUMN_NAME]
        # Replace possible invalids
        date_series_original = date_series_original.replace([0, '0', '', ' '], pd.NA)

        def parse_mixed_dates(val):
            if pd.isnull(val) or val == '':
                return pd.NaT
            try:
                # Try string date first (dayfirst for Europe)
                return pd.to_datetime(val, dayfirst=True, errors='raise')
            except Exception:
                try:
                    v = float(val)
                    # Excel serial date (typically > 40000 for recent years)
                    if v > 40000:
                        return pd.to_datetime('1899-12-30') + pd.to_timedelta(v, 'D')
                    # Unix timestamp
                    if v > 1000000000:
                        return pd.to_datetime(int(v), unit='s')
                except Exception:
                    pass
            return pd.NaT

        parsed_dates = date_series_original.apply(parse_mixed_dates)
        df_display[DATE_COLUMN_NAME] = parsed_dates
        df_display.dropna(subset=[DATE_COLUMN_NAME], inplace=True)

        if not df_display.empty:
            min_dt_val_series = df_display[DATE_COLUMN_NAME].min()
            max_dt_val_series = df_display[DATE_COLUMN_NAME].max()
            if pd.notna(min_dt_val_series) and pd.notna(max_dt_val_series):
                min_date_for_filter = min_dt_val_series.date()
                max_date_for_filter = max_dt_val_series.date()
                date_column_valid = True
            else:
                st.warning(f"Could not determine a valid min/max date from '{DATE_COLUMN_NAME}' after parsing. Date filtering disabled.")
        else:
            st.warning(f"No valid dates remained in '{DATE_COLUMN_NAME}' after parsing and cleaning. Date filtering disabled.")
    else:
        st.info(f"'{DATE_COLUMN_NAME}' column not found. Date filtering disabled.")
    
    st.sidebar.header("Filters")
    selected_date_range = None

    if date_column_valid:
        default_start_date = min_date_for_filter
        default_end_date = max_date_for_filter

        if min_date_for_filter > max_date_for_filter:
            st.sidebar.warning(f"Min date ({min_date_for_filter}) is after max date ({max_date_for_filter}). Swapping for filter.")
            default_start_date, default_end_date = max_date_for_filter, min_date_for_filter

        current_value_for_widget = (default_start_date, default_end_date)
        if 'date_filter' in st.session_state and \
           isinstance(st.session_state.date_filter, tuple) and \
           len(st.session_state.date_filter) == 2:
            prev_start, prev_end = st.session_state.date_filter
            if isinstance(prev_start, datetime.date) and isinstance(prev_end, datetime.date):
                clamped_prev_start = max(min_date_for_filter, min(prev_start, max_date_for_filter))
                clamped_prev_end = max(min_date_for_filter, min(prev_end, max_date_for_filter))
                if clamped_prev_start <= clamped_prev_end:
                    current_value_for_widget = (clamped_prev_start, clamped_prev_end)
        try:
            selected_date_range = st.sidebar.date_input(
                "Select Date Range",
                value=current_value_for_widget,
                min_value=min_date_for_filter, 
                max_value=max_date_for_filter,
                key="date_filter"
            )
            if not (isinstance(selected_date_range, tuple) and len(selected_date_range) == 2 and
                    isinstance(selected_date_range[0], datetime.date) and
                    isinstance(selected_date_range[1], datetime.date)):
                st.sidebar.warning("Date range selection became invalid. Resetting to default for the current data.")
                selected_date_range = (default_start_date, default_end_date)
                st.session_state.date_filter = selected_date_range
        except Exception as e:
            st.sidebar.error(f"Error rendering date filter: {e}. Using default range.")
            selected_date_range = (default_start_date, default_end_date)
            if 'date_filter' in st.session_state: del st.session_state.date_filter 
    else:
        st.sidebar.info(f"Date range filter is disabled because a valid date range could not be determined from the '{DATE_COLUMN_NAME}' column. Please check the data and format (e.g., dd/mm/yyyy).")

    selected_divisions = []
    if DIVISION_COLUMN_NAME in df_display.columns and not df_display[DIVISION_COLUMN_NAME].dropna().empty:
        unique_divisions = sorted(df_display[DIVISION_COLUMN_NAME].dropna().astype(str).unique())
        selected_divisions = st.sidebar.multiselect(f"Select {DIVISION_COLUMN_NAME}(s)", options=unique_divisions, default=unique_divisions, key="division_filter")
    else:
        st.sidebar.info(f"'{DIVISION_COLUMN_NAME}' column not available/valid for filtering.")

    selected_themes = []
    if THEME_COLUMN_NAME in df_display.columns and not df_display[THEME_COLUMN_NAME].dropna().empty:
        unique_themes = sorted(df_display[THEME_COLUMN_NAME].dropna().astype(str).unique())
        selected_themes = st.sidebar.multiselect(f"Select {THEME_COLUMN_NAME}(s)", options=unique_themes, default=unique_themes, key="theme_filter")
    else:
        st.sidebar.info(f"'{THEME_COLUMN_NAME}' column not available/valid for filtering.")

    filtered_df_display = df_display.copy()

    if selected_date_range and date_column_valid and len(selected_date_range) == 2:
        try:
            start_date = pd.to_datetime(selected_date_range[0])
            end_date = pd.to_datetime(selected_date_range[1])
            if DATE_COLUMN_NAME in filtered_df_display and pd.api.types.is_datetime64_any_dtype(filtered_df_display[DATE_COLUMN_NAME]):
                filtered_df_display = filtered_df_display[
                    (filtered_df_display[DATE_COLUMN_NAME] >= start_date) &
                    (filtered_df_display[DATE_COLUMN_NAME] <= end_date)
                ]
            else:
                st.warning(f"Cannot apply date filter as '{DATE_COLUMN_NAME}' is not in the expected datetime format in the filtered data.")
        except Exception as e:
            st.error(f"Error applying date filter: {e}")

    if selected_divisions and DIVISION_COLUMN_NAME in filtered_df_display.columns:
        filtered_df_display = filtered_df_display[filtered_df_display[DIVISION_COLUMN_NAME].isin(selected_divisions)]

    if selected_themes and THEME_COLUMN_NAME in filtered_df_display.columns:
        filtered_df_display = filtered_df_display[filtered_df_display[THEME_COLUMN_NAME].isin(selected_themes)]

    st.header("Dashboard Overview")
    num_observations = len(filtered_df_display)
    st.metric(label="Number of Observations (after filters)", value=num_observations)
    st.markdown("---")

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
                popup_text = f"<b>{ADDRESS_COLUMN_NAME}:</b> {row.get(ADDRESS_COLUMN_NAME, 'N/A')}"
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
        elif not filtered_df_display.empty and map_df.empty:
            st.warning("No valid geocoded locations to display on the map for the current filter selection, or all addresses failed/yielded no coordinates.")
        else: 
            st.info("Apply filters or upload data with addresses to see locations on the map.")
    else: 
        if ADDRESS_COLUMN_NAME in df_display.columns:
            st.info("Map will appear once addresses are processed and geocoded successfully.")

    if st.session_state.get('geocoded_pairs_for_download'):
        dl_df_data = []
        for addr, (lat, lon) in st.session_state.geocoded_pairs_for_download.items():
            if lat is not None and lon is not None:
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

    if st.checkbox("Show Filtered Data Table", key="show_data_table"):
        st.subheader("Filtered Data (with Latitude/Longitude if available)")
        st.dataframe(filtered_df_display, use_container_width=True)

elif not uploaded_file : 
    st.info("Please upload a main spreadsheet to begin analysis.")
    st.markdown(f"""
        ### Welcome to the Spreadsheet Analysis Dashboard!
        - Use the uploader above to load your main data (CSV or Excel).
        - Optionally, upload a **Geocoding Lookup Table** in the sidebar to speed up geocoding.
        - Ensure your main file contains '{DATE_COLUMN_NAME}', '{THEME_COLUMN_NAME}', and '{DIVISION_COLUMN_NAME}' columns for charts and filters.
        - For map functionality, include an '{ADDRESS_COLUMN_NAME}' column.
        - Geocoding will use the lookup table first, then the Nominatim API (1 request/second).
    """)

if st.session_state.df is not None or st.session_state.geocoded_df is not None:
    if st.sidebar.button("Clear All Data and Reset App", key="clear_data"):
        st.session_state.df = None
        st.session_state.geocoded_df = None
        st.session_state.geocoding_done = False
        st.session_state.geocoded_pairs_for_download = {}
        st.session_state.file_uploader_key += 1
        st.session_state.lookup_df = None
        if 'lookup_uploader' in st.session_state:
            st.session_state.lookup_uploader = None 
        if '_last_lookup_file_id' in st.session_state:
            del st.session_state['_last_lookup_file_id']
        if '_last_uploaded_main_file_id' in st.session_state:
            del st.session_state['_last_uploaded_main_file_id']
        keys_to_clear_from_session = [
            "date_filter", "theme_filter", "division_filter", 
            "show_data_table", "folium_map"
        ]
        for key in keys_to_clear_from_session:
            if key in st.session_state:
                del st.session_state[key]
        st.success("Application reset. Please upload new files.")
        st.rerun()
