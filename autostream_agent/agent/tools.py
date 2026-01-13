def mock_lead_capture(name: str, email: str, platform: str):
    """
    Mock backend API call for lead capture.
    In production, this would be an HTTP request or CRM integration.
    """
    print(f"Lead captured successfully: {name}, {email}, {platform}")
