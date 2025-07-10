from pydantic import BaseModel

class InputData(BaseModel):
    engagement_score: float
    duration_minutes: float
    completion_rate: float
    quiz_score: float
    material_rating: float
    interaction_duration: float
    material_engagement_score: float
