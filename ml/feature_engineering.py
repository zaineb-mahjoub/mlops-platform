from schemas import ExtractedProjectSpec

DEFAULT_DEADLINE_DAYS = 60

def map_to_features(spec: ExtractedProjectSpec, raw_text: str) -> dict:
    app_types = [a.lower() for a in spec.application_type]
    app_text = " ".join(app_types)

    return {
        "num_functional_requirements": len(spec.functional_requirements),
        "num_integrations": len(spec.integrations),
        "requires_web_component": int("web" in app_text),
        "requires_mobile_component": int("mobile" in app_text),
        "requires_ai_ml_component": int(len(spec.ai_ml_requirements) > 0),
        "requires_cloud_infra": int(len(spec.cloud_requirements) > 0),
        "requires_data_pipeline": int(len(spec.data_requirements) > 0),
        "num_technologies_mentioned": len(spec.technologies),
        "has_deadline": int(spec.deadline_days is not None),
        "deadline_days": float(spec.deadline_days) if spec.deadline_days is not None else float(DEFAULT_DEADLINE_DAYS),
        "complexity_keyword_count": len(spec.security_requirements)
            + int(len(spec.real_time_requirements) > 0)
            + len(spec.constraints),
        "requirement_text_length": len(raw_text.split()),
    }
