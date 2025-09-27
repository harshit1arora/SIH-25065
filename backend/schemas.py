from pydantic import BaseModel
from typing import Optional, List, Dict, Any
from datetime import datetime

class AssessmentBase(BaseModel):
    name: str
    location: str
    dwellers: int
    roof_area: float
    open_space: float
    roof_type: str
    roof_age: int

class AssessmentCreate(AssessmentBase):
    pass

class AssessmentUpdate(BaseModel):
    latitude: Optional[float] = None
    longitude: Optional[float] = None
    soil_type: Optional[str] = None
    aquifer_type: Optional[str] = None
    water_depth: Optional[float] = None
    annual_rainfall: Optional[float] = None
    runoff_coefficient: Optional[float] = None
    recommended_structure: Optional[str] = None
    annual_harvestable_water: Optional[float] = None
    installation_cost: Optional[float] = None
    payback_period: Optional[float] = None
    feedback_rating: Optional[float] = None
    feedback_notes: Optional[str] = None

class Assessment(AssessmentBase):
    id: int
    latitude: Optional[float] = None
    longitude: Optional[float] = None
    soil_type: Optional[str] = None
    aquifer_type: Optional[str] = None
    water_depth: Optional[float] = None
    annual_rainfall: Optional[float] = None
    runoff_coefficient: Optional[float] = None
    recommended_structure: Optional[str] = None
    annual_harvestable_water: Optional[float] = None
    installation_cost: Optional[float] = None
    payback_period: Optional[float] = None
    feedback_rating: Optional[float] = None
    feedback_notes: Optional[str] = None
    created_at: datetime
    updated_at: Optional[datetime] = None
    
    class Config:
        from_attributes = True

# API Request/Response Schemas
class GeocodingRequest(BaseModel):
    address: str

class GeocodingResponse(BaseModel):
    latitude: Optional[float] = None
    longitude: Optional[float] = None
    success: bool
    error: Optional[str] = None

class RainfallRequest(BaseModel):
    latitude: float
    longitude: float

class RainfallResponse(BaseModel):
    annual_rainfall: Optional[float] = None
    monthly_breakdown: Optional[List[float]] = None
    success: bool
    error: Optional[str] = None

class GroundwaterRequest(BaseModel):
    latitude: float
    longitude: float

class GroundwaterResponse(BaseModel):
    depth_to_water: Optional[float] = None
    aquifer_type: Optional[str] = None
    water_quality: Optional[str] = None
    success: bool
    error: Optional[str] = None

class SoilTypeRequest(BaseModel):
    latitude: float
    longitude: float

class SoilTypeResponse(BaseModel):
    soil_type: Optional[str] = None
    success: bool
    error: Optional[str] = None

class CalculateRequest(BaseModel):
    roof_area: float
    rainfall: float
    roof_type: str
    dwellers: int

class CalculateResponse(BaseModel):
    harvestable_water: float
    annual_demand: float
    potential_savings: float
    savings_percentage: float

class RecommendationRequest(BaseModel):
    roof_area: float
    open_space: float
    soil_type: str
    aquifer_type: str
    water_depth: float
    rainfall: float

class RecommendationItem(BaseModel):
    name: str
    description: str
    cost: str

class RecommendationResponse(BaseModel):
    recommendations: List[RecommendationItem]
    cost_benefit_analysis: Dict[str, Any]

class AquiferInfoResponse(BaseModel):
    description: str
    recharge_potential: str
    suitable_structures: List[str]

class MLPredictionRequest(BaseModel):
    roof_type: str
    roof_age: int
    region: str
    roof_area: float
    open_space: float
    soil_type: str
    aquifer_type: str
    water_depth: float
    annual_rainfall: float

class MLPredictionResponse(BaseModel):
    runoff_coefficient: float
    recommended_structure: str
    annual_harvestable_water: float
    installation_cost: float
    payback_period: float
    success: bool = True

class FeedbackSchema(BaseModel):
    rating: float
    notes: Optional[str] = None

class FeedbackResponse(BaseModel):
    message: str
