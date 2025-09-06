import streamlit as st
import requests
import pandas as pd
import matplotlib.pyplot as plt

# Set page configuration
st.set_page_config(
    page_title="Rooftop Rainwater Harvesting Assessment",
    page_icon="💧",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for styling
st.markdown("""
<style>
    .main-header {font-size: 2.5rem; color: #1f77b4; font-weight: 700;}
    .sub-header {font-size: 1.8rem; color: #1f77b4; border-bottom: 2px solid #1f77b4; padding-bottom: 0.3rem;}
    .result-box {background-color: #f0f8ff; padding: 20px; border-radius: 8px; border-left: 5px solid #1f77b4;}
    .recommendation {background-color: #e6f3ff; padding: 15px; border-radius: 5px; margin: 10px 0;}
    footer {text-align: center; margin-top: 2rem; padding: 1rem; color: #666; font-size: 0.8rem;}
</style>
""", unsafe_allow_html=True)

# API endpoints (to be configured based on your backend)
API_BASE_URL = "http://localhost:8000"  # Change this to your backend URL
GEOCODING_API_URL = f"{API_BASE_URL}/api/geocode"
RAINFALL_API_URL = f"{API_BASE_URL}/api/rainfall"
GROUNDWATER_API_URL = f"{API_BASE_URL}/api/groundwater"
AQUIFER_API_URL = f"{API_BASE_URL}/api/aquifer"
CALCULATE_API_URL = f"{API_BASE_URL}/api/calculate"
RECOMMEND_API_URL = f"{API_BASE_URL}/api/recommend"
SOIL_TYPE_API_URL = f"{API_BASE_URL}/api/soil-type"
WATER_LEVEL_API_URL = f"{API_BASE_URL}/api/water-level-trends"

# Function to get data from backend APIs
def get_geocoding_data(location):
    """
    Fetches geocoding data from backend API
    Returns: dict with latitude and longitude or None if API call fails
    """
    try:
        response = requests.get(f"{GEOCODING_API_URL}?address={location}")
        if response.status_code == 200:
            return response.json()
        else:
            st.error(f"Geocoding API error: {response.status_code}")
            return None
    except Exception as e:
        st.error(f"Error fetching geocoding data: {e}")
        return None

def get_rainfall_data(lat, lon):
    """
    Fetches rainfall data from backend API
    Returns: dict with rainfall data or None if API call fails
    """
    try:
        response = requests.get(f"{RAINFALL_API_URL}?lat={lat}&lon={lon}")
        if response.status_code == 200:
            return response.json()
        else:
            st.error(f"Rainfall API error: {response.status_code}")
            return None
    except Exception as e:
        st.error(f"Error fetching rainfall data: {e}")
        return None

def get_groundwater_data(lat, lon):
    """
    Fetches groundwater data from backend API
    Returns: dict with groundwater data or None if API call fails
    """
    try:
        response = requests.get(f"{GROUNDWATER_API_URL}?lat={lat}&lon={lon}")
        if response.status_code == 200:
            return response.json()
        else:
            st.error(f"Groundwater API error: {response.status_code}")
            return None
    except Exception as e:
        st.error(f"Error fetching groundwater data: {e}")
        return None

def get_soil_type(lat, lon):
    """
    Fetches soil type data from backend API
    Returns: soil type string or None if API call fails
    """
    try:
        response = requests.get(f"{SOIL_TYPE_API_URL}?lat={lat}&lon={lon}")
        if response.status_code == 200:
            data = response.json()
            return data.get("soil_type")
        else:
            st.error(f"Soil type API error: {response.status_code}")
            return None
    except Exception as e:
        st.error(f"Error fetching soil type data: {e}")
        return None

def get_aquifer_info(aquifer_type):
    """
    Fetches aquifer information from backend API
    Returns: dict with aquifer info or None if API call fails
    """
    try:
        response = requests.get(f"{AQUIFER_API_URL}?type={aquifer_type}")
        if response.status_code == 200:
            return response.json()
        else:
            st.error(f"Aquifer API error: {response.status_code}")
            return None
    except Exception as e:
        st.error(f"Error fetching aquifer data: {e}")
        return None

def get_water_level_trends(lat, lon):
    """
    Fetches water level trends from backend API
    Returns: dict with water level trends or None if API call fails
    """
    try:
        response = requests.get(f"{WATER_LEVEL_API_URL}?lat={lat}&lon={lon}")
        if response.status_code == 200:
            return response.json()
        else:
            st.error(f"Water level API error: {response.status_code}")
            return None
    except Exception as e:
        st.error(f"Error fetching water level data: {e}")
        return None

def calculate_potential(roof_area, roof_type, rainfall, dwellers):
    """
    Calls backend API to calculate water harvesting potential
    Returns: dict with calculation results or None if API call fails
    """
    try:
        payload = {
            "roof_area": roof_area,
            "roof_type": roof_type,
            "rainfall": rainfall,
            "dwellers": dwellers
        }
        response = requests.post(CALCULATE_API_URL, json=payload)
        if response.status_code == 200:
            return response.json()
        else:
            st.error(f"Calculation API error: {response.status_code}")
            return None
    except Exception as e:
        st.error(f"Error calculating potential: {e}")
        return None

def get_recommendations(roof_area, open_space, soil_type, aquifer_type, water_depth, rainfall):
    """
    Calls ML model via backend API to get structure recommendations
    Returns: dict with recommendations or None if API call fails
    """
    try:
        payload = {
            "roof_area": roof_area,
            "open_space": open_space,
            "soil_type": soil_type,
            "aquifer_type": aquifer_type,
            "water_depth": water_depth,
            "rainfall": rainfall
        }
        response = requests.post(RECOMMEND_API_URL, json=payload)
        if response.status_code == 200:
            return response.json()
        else:
            st.error(f"Recommendation API error: {response.status_code}")
            return None
    except Exception as e:
        st.error(f"Error getting recommendations: {e}")
        return None

# App title and description
st.markdown('<p class="main-header">💧 Roof Top Rain Water Harvesting Assessment Tool</p>', unsafe_allow_html=True)
st.markdown("""
This tool helps you assess the potential for rooftop rainwater harvesting and artificial recharge at your location. 
Enter your details below to get personalized recommendations based on scientific models and local data.
""")

# Initialize session state for user data
if 'user_data' not in st.session_state:
    st.session_state.user_data = {
        'name': '',
        'location': '',
        'dwellers': 1,
        'roof_area': 100,
        'open_space': 100,
        'roof_type': 'Concrete',
        'roof_age': 5,
        'soil_type': '',
        'lat': None,
        'lon': None
    }

# Sidebar for user input
with st.sidebar:
    st.header("User Input")
    st.session_state.user_data['name'] = st.text_input("Name", value=st.session_state.user_data['name'])
    st.session_state.user_data['location'] = st.text_input("Location/Address", value=st.session_state.user_data['location'])
    
    st.session_state.user_data['dwellers'] = st.number_input("Number of Dwellers", min_value=1, max_value=50, value=st.session_state.user_data['dwellers'])
    st.session_state.user_data['roof_area'] = st.number_input("Roof Area (sq. meters)", min_value=10, max_value=1000, value=st.session_state.user_data['roof_area'])
    st.session_state.user_data['open_space'] = st.number_input("Available Open Space (sq. meters)", min_value=0, max_value=1000, value=st.session_state.user_data['open_space'])
    st.session_state.user_data['roof_type'] = st.selectbox("Roof Type", 
                                                          ['Concrete', 'Tiled', 'Metal', 'Asbestos', 'Thatched'],
                                                          index=0)
    
    # Calculate button
    calculate_button = st.button("Calculate Potential", type="primary")

# Main content area
tab1, tab2, tab3, tab4 = st.tabs(["Assessment", "Recommendations", "Groundwater Info", "About"])

with tab1:
    st.markdown('<p class="sub-header">Rainwater Harvesting Potential Assessment</p>', unsafe_allow_html=True)
    
    if calculate_button:
        # Get geocoding data
        with st.spinner('Getting location coordinates...'):
            geocoding_data = get_geocoding_data(st.session_state.user_data['location'])
            
            if geocoding_data and geocoding_data.get('success'):
                st.session_state.user_data['lat'] = geocoding_data.get('lat')
                st.session_state.user_data['lon'] = geocoding_data.get('lon')
            else:
                st.error("Could not geocode the provided location. Please check the address and try again.")
                st.stop()
        
        # Get rainfall data
        with st.spinner('Fetching rainfall data for your location...'):
            rainfall_data = get_rainfall_data(
                st.session_state.user_data['lat'], 
                st.session_state.user_data['lon']
            )
            
            if not rainfall_data or not rainfall_data.get('success'):
                st.error("Could not fetch rainfall data for your location. Please try again later.")
                st.stop()
        
        # Get groundwater data
        with st.spinner('Fetching groundwater information...'):
            groundwater_data = get_groundwater_data(
                st.session_state.user_data['lat'], 
                st.session_state.user_data['lon']
            )
            
            if not groundwater_data or not groundwater_data.get('success'):
                st.error("Could not fetch groundwater data for your location. Please try again later.")
                st.stop()
        
        # Get soil type
        with st.spinner('Determining soil type...'):
            soil_type = get_soil_type(
                st.session_state.user_data['lat'], 
                st.session_state.user_data['lon']
            )
            
            if soil_type:
                st.session_state.user_data['soil_type'] = soil_type
            else:
                st.error("Could not determine soil type for your location. Please try again later.")
                st.stop()
        
        # Calculate potential
        with st.spinner('Calculating water harvesting potential...'):
            calculation_result = calculate_potential(
                st.session_state.user_data['roof_area'],
                st.session_state.user_data['roof_type'],
                rainfall_data['annual_rainfall'],
                st.session_state.user_data['dwellers']
            )
            
            if not calculation_result or not calculation_result.get('success'):
                st.error("Could not calculate water harvesting potential. Please try again later.")
                st.stop()
        
        # Display results
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric("Annual Harvestable Water", f"{calculation_result['harvestable_water']:.2f} m³")
            st.metric("Equivalent to", f"{(calculation_result['harvestable_water'] * 1000):.0f} liters")
            
        with col2:
            st.metric("Annual Water Demand", f"{calculation_result['annual_demand']:.2f} m³")
            st.metric("Potential Savings", f"{calculation_result['savings_percentage']:.1f}%")
            
        with col3:
            st.metric("Depth to Water Level", f"{groundwater_data['depth_to_water']} m")
            st.metric("Annual Rainfall", f"{rainfall_data['annual_rainfall']} mm")
        
        # Display rainfall pattern chart if data is available
        if 'monthly_breakdown' in rainfall_data:
            st.subheader("Monthly Rainfall Pattern")
            months = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
            fig, ax = plt.subplots(figsize=(10, 4))
            ax.bar(months, rainfall_data['monthly_breakdown'], color='skyblue')
            ax.set_ylabel('Rainfall (mm)')
            ax.set_title('Monthly Rainfall Distribution')
            st.pyplot(fig)

with tab2:
    st.markdown('<p class="sub-header">Recommended RWH Structures</p>', unsafe_allow_html=True)
    
    if calculate_button:
        # Get recommendations from ML model
        with st.spinner('Analyzing your inputs and generating recommendations...'):
            recommendations_data = get_recommendations(
                st.session_state.user_data['roof_area'],
                st.session_state.user_data['open_space'],
                st.session_state.user_data['soil_type'],
                groundwater_data['aquifer_type'],
                groundwater_data['depth_to_water'],
                rainfall_data['annual_rainfall']
            )
            
            if not recommendations_data or not recommendations_data.get('success'):
                st.error("Could not generate recommendations. Please try again later.")
                st.stop()
        
        recommendations = recommendations_data['recommendations']
        
        # Display recommendations
        for rec in recommendations:
            with st.expander(f"{rec['name']} - {rec['cost']}"):
                st.write(rec['description'])
                if 'dimensions' in rec:
                    st.write(f"**Dimensions:** {rec['dimensions']}")
                if 'capacity' in rec:
                    st.write(f"**Capacity:** {rec['capacity']}")
                st.write(f"**Estimated Cost:** {rec['cost']}")
                if 'confidence' in rec:
                    st.write(f"**Confidence Score:** {rec['confidence']*100:.0f}%")
        
        # Display cost-benefit analysis if available
        if 'cost_benefit_analysis' in recommendations_data:
            st.subheader("Cost-Benefit Analysis")
            
            cba = recommendations_data['cost_benefit_analysis']
            
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Estimated Annual Water Savings", f"{cba['annual_water_savings']:.0f} liters")
            with col2:
                st.metric("Value of Saved Water", f"₹{cba['annual_savings_value']:.0f}")
            with col3:
                st.metric("Payback Period", f"{cba['payback_period']:.1f} years")

with tab3:
    st.markdown('<p class="sub-header">Groundwater Information</p>', unsafe_allow_html=True)
    
    if calculate_button:
        # Display groundwater information
        st.subheader("Aquifer Characteristics")
        
        # Get aquifer info
        aquifer_info = get_aquifer_info(groundwater_data['aquifer_type'])
        
        if aquifer_info and aquifer_info.get('success'):
            st.write(f"**Aquifer Type:** {groundwater_data['aquifer_type']}")
            st.write(f"**Description:** {aquifer_info['description']}")
            st.write(f"**Recharge Potential:** {aquifer_info['recharge_potential']}")
            st.write(f"**Suitable Structures:** {', '.join(aquifer_info['suitable_structures'])}")
        else:
            st.write(f"**Aquifer Type:** {groundwater_data['aquifer_type']}")
            st.warning("Detailed aquifer information is not available for this location.")
        
        # Display water level trends
        st.subheader("Water Level Trends")
        
        # Get water level trends
        water_level_trends = get_water_level_trends(
            st.session_state.user_data['lat'], 
            st.session_state.user_data['lon']
        )
        
        if water_level_trends and water_level_trends.get('success'):
            # Create a DataFrame for plotting
            trends_df = pd.DataFrame(water_level_trends['trends'])
            
            fig, ax = plt.subplots(figsize=(10, 4))
            ax.plot(trends_df['year'], trends_df['water_level'], marker='o', linewidth=2)
            ax.set_xlabel('Year')
            ax.set_ylabel('Depth to Water Level (m)')
            ax.set_title('Historical Water Level Trends')
            ax.grid(True, linestyle='--', alpha=0.7)
            st.pyplot(fig)
            
            # Show trend analysis if available
            if 'trend_analysis' in water_level_trends:
                st.write(f"**Trend:** {water_level_trends['trend_analysis']}")
        else:
            st.warning("Water level trend data is not available for this location.")

with tab4:
    st.markdown('<p class="sub-header">About This Tool</p>', unsafe_allow_html=True)
    
    st.write("""
    This Rooftop Rainwater Harvesting Assessment Tool is designed to promote public participation 
    in groundwater conservation by enabling users to estimate the feasibility of rooftop rainwater 
    harvesting (RTRWH) and artificial recharge at their locations.
    
    ### How It Works
    The tool uses scientific models based on guidelines from the Central Ground Water Board (CGWB) 
    to calculate:
    - Harvestable rainwater based on roof area and local rainfall patterns
    - Appropriate recharge structures based on soil conditions and available space
    - Cost estimates and cost-benefit analysis for implementation
    
    ### API Integration
    This application integrates with several backend services:
    - Geocoding API: Converts addresses to coordinates
    - Rainfall Data API: Provides localized rainfall patterns
    - Groundwater API: Offers current water level and aquifer information
    - Soil Type API: Identifies soil characteristics for recharge suitability
    - Calculation API: Performs water harvesting potential calculations
    - Recommendation API: Uses ML models to suggest optimal structures
    - Water Level API: Provides historical water level trends
    
    ### Benefits of Rainwater Harvesting
    - Replenishes groundwater resources
    - Reduces water bills and dependence on municipal supply
    - Mitigates urban flooding by reducing runoff
    - Improves groundwater quality by dilution of contaminants
    
    ### Disclaimer
    This tool provides preliminary estimates based on standard parameters. For detailed design 
    and implementation, consult with certified rainwater harvesting professionals.
    """)

# Footer
st.markdown("---")
st.markdown("<footer>Developed for sustainable water management | © 2023 CGWB</footer>", unsafe_allow_html=True)