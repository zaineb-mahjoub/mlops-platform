from pydantic import BaseModel
from typing import List, Optional

class ExtractedProjectSpec(BaseModel):
    project_name: str
    domain: str
    application_type: List[str]
    functional_requirements: List[str]
    technologies: List[str]
    integrations: List[str]
    database: List[str]
    security_requirements: List[str]
    ai_ml_requirements: List[str]
    cloud_requirements: List[str]
    data_requirements: List[str]
    real_time_requirements: List[str]
    deadline_days: Optional[int]
    constraints: List[str]
    