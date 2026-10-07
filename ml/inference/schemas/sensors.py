from pydantic import BaseModel, Field
from typing import Literal, Optional


class SensorReading(BaseModel):
    soil_moisture: float
    temperature: float
    humidity: float
    ph: float
    tds: float = 0.0
    turbidity: float = 0.0
    N: float = 0.0
    P: float = 0.0
    K: float = 0.0
    rainfall: float = 0.0
    crop_type: Optional[str] = None


class IrrigationResponse(BaseModel):
    irrigate: bool
    confidence: float


class FertilizerResponse(BaseModel):
    recommendation: str
    confidence: float


class WaterQualityResponse(BaseModel):
    quality: Literal["good", "acceptable", "poor"]


class AnomalyResponse(BaseModel):
    is_anomaly: bool
    score: float


class VisionResponse(BaseModel):
    label: str
    confidence: float
    boxes: list = Field(default_factory=list)
