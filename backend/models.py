from sqlalchemy import Column, Integer, String, Float, DateTime, Text
from sqlalchemy.sql import func
from .database import Base

class UserAssessment(Base):
    __tablename__ = "user_assessments"
    
    id = Column(Integer, primary_key=True, index=True)
    name = Column(String(100), index=True)
    location = Column(String(200))
    latitude = Column(Float, nullable=True)
    longitude = Column(Float, nullable=True)
    dwellers = Column(Integer)
    roof_area = Column(Float)
    open_space = Column(Float)
    roof_type = Column(String(50))
    roof_age = Column(Integer)
    
    # API Data
    soil_type = Column(String(50), nullable=True)
    aquifer_type = Column(String(100), nullable=True)
    water_depth = Column(Float, nullable=True)
    annual_rainfall = Column(Float, nullable=True)
    
    # ML Model Outputs
    runoff_coefficient = Column(Float, nullable=True)
    recommended_structure = Column(String(100), nullable=True)
    annual_harvestable_water = Column(Float, nullable=True)
    installation_cost = Column(Float, nullable=True)
    payback_period = Column(Float, nullable=True)
    
    # Feedback
    feedback_rating = Column(Float, nullable=True)
    feedback_notes = Column(Text, nullable=True)
    
    # Timestamps
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    updated_at = Column(DateTime(timezone=True), onupdate=func.now())
