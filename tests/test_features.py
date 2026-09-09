import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "ml"))
from schemas import ExtractedProjectSpec
from feature_engineering import map_to_features

def test_map_to_features_returns_12_fields():
    spec = ExtractedProjectSpec(
        project_name="Test", domain="Test",
        application_type=["web"], functional_requirements=["a", "b"],
        technologies=[], integrations=[], database=[],
        security_requirements=[], ai_ml_requirements=[],
        cloud_requirements=[], data_requirements=[],
        real_time_requirements=[], deadline_days=30, constraints=[],
    )
    features = map_to_features(spec, "some raw text here")
    assert len(features) == 12
    assert features["requires_web_component"] == 1
    assert features["num_functional_requirements"] == 2

def test_web_platform_substring_match():
    spec = ExtractedProjectSpec(
        project_name="T", domain="T", application_type=["Web Platform"],
        functional_requirements=[], technologies=[], integrations=[], database=[],
        security_requirements=[], ai_ml_requirements=[], cloud_requirements=[],
        data_requirements=[], real_time_requirements=[], deadline_days=None, constraints=[],
    )
    features = map_to_features(spec, "text")
    assert features["requires_web_component"] == 1
    