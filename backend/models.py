from sqlalchemy import Column, Integer, String, Float, DateTime, Text, JSON
from sqlalchemy.sql import func
from database import Base

class UserAssessment(Base):
    __tablename__ = "user_assessments"
    
    id = Column(Integer, primary_key=True, index=True)
    name = Column(String(100), index=True)
    location = Column(String(200))
    latitude = Column(Float)
    longitude = Column(Float)
    dwellers = Column(Integer)
    roof_area = Column(Float)
    open_space = Column(Float)
    roof_type = Column(String(50))
    soil_type = Column(String(50))
    
    # Results
    annual_rainfall = Column(Float)
    depth_to_water = Column(Float)
    aquifer_type = Column(String(100))
    harvestable_water = Column(Float)
    annual_demand = Column(Float)
    potential_savings = Column(Float)
    savings_percentage = Column(Float)
    
    # Recommendations (stored as JSON)
    recommendations = Column(JSON)
    cost_benefit_analysis = Column(JSON)
    
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    updated_at = Column(DateTime(timezone=True), onupdate=func.now())

# Removed the static data tables since we'll get this from external APIs