AREA_PROFILES = {
    "Traffic Signal": {
        "description": "Intersection/crossroad enforcement",
        "focus": ["Red-Light Violation", "Stop-Line / Wrong Way", "Helmet Non-Compliance", "Seatbelt Non-Compliance"],
        "priors": {
            "Red-Light Violation": 0.85,
            "Stop-Line / Wrong Way": 0.80,
            "Helmet Non-Compliance": 0.70,
            "Suspected Helmet Non-Compliance": 0.70,
            "Seatbelt Non-Compliance": 0.65,
        },
        "speed_limits": {"motorcycle": 35, "car": 40, "bus": 35, "truck": 30, "default": 40},
    },
    "Market / No-Parking Zone": {
        "description": "Dense roadside market or commercial street",
        "focus": ["Suspected Illegal Parking", "Wrong-Side Driving", "Overcrowding / Triple Riding"],
        "priors": {
            "Suspected Illegal Parking": 0.85,
            "Illegal Parking": 0.90,
            "Suspected Wrong-Side Driving": 0.70,
            "Overcrowding / Triple Riding": 0.65,
        },
        "speed_limits": {"motorcycle": 25, "car": 30, "bus": 25, "truck": 20, "default": 30},
    },
    "Highway": {
        "description": "High-speed corridor",
        "focus": ["Speeding", "Seatbelt Non-Compliance", "Hazardous/Long Load", "Wrong-Side Driving"],
        "priors": {
            "Speeding": 0.85,
            "Seatbelt Non-Compliance": 0.70,
            "Hazardous/Long Load": 0.65,
            "Wrong-Side Driving": 0.60,
        },
        "speed_limits": {"motorcycle": 70, "car": 80, "bus": 65, "truck": 60, "default": 70},
    },
    "School Zone": {
        "description": "Low-speed pedestrian-sensitive zone",
        "focus": ["Speeding", "Illegal Parking", "Stop-Line / Wrong Way"],
        "priors": {
            "Speeding": 0.80,
            "Suspected Illegal Parking": 0.70,
            "Stop-Line / Wrong Way": 0.65,
        },
        "speed_limits": {"motorcycle": 25, "car": 25, "bus": 20, "truck": 20, "default": 25},
    },
    "Residential Road": {
        "description": "Local street with mixed pedestrians and two-wheelers",
        "focus": ["Helmet Non-Compliance", "Wrong-Side Driving", "Illegal Parking"],
        "priors": {
            "Helmet Non-Compliance": 0.70,
            "Suspected Helmet Non-Compliance": 0.70,
            "Suspected Wrong-Side Driving": 0.65,
            "Suspected Illegal Parking": 0.60,
        },
        "speed_limits": {"motorcycle": 35, "car": 35, "bus": 30, "truck": 25, "default": 35},
    },
    "Industrial Zone": {
        "description": "Truck-heavy road or warehouse district",
        "focus": ["Hazardous/Long Load", "Speeding", "Wrong-Side Driving"],
        "priors": {
            "Hazardous/Long Load": 0.80,
            "Speeding": 0.65,
            "Suspected Wrong-Side Driving": 0.60,
        },
        "speed_limits": {"motorcycle": 40, "car": 45, "bus": 35, "truck": 30, "default": 40},
    },
}


def horizontal_zone(width, height, start_ratio, end_ratio=None):
    start_y = int(round(height * float(start_ratio)))
    end_y = int(round(height * float(end_ratio if end_ratio is not None else min(start_ratio + 0.12, 1.0))))
    start_y = max(0, min(height, start_y))
    end_y = max(start_y + 1, min(height, end_y))
    return [(0, start_y), (width, start_y), (width, end_y), (0, end_y)]


def vertical_zone(width, height, side="left", ratio=0.5):
    split = int(round(width * float(ratio)))
    split = max(1, min(width - 1, split))
    if side == "right":
        return [(split, 0), (width, 0), (width, height), (split, height)]
    return [(0, 0), (split, 0), (split, height), (0, height)]


def build_rule_context(
    area_type,
    image_shape,
    traffic_light_status="RED",
    stop_line_ratio=0.60,
    enable_stop_zone=True,
    enable_no_parking_zone=False,
    enable_wrong_side_zone=False,
    wrong_side="left",
):
    height, width = int(image_shape[0]), int(image_shape[1])
    profile = AREA_PROFILES.get(area_type, AREA_PROFILES["Traffic Signal"])
    context = {
        "area_type": area_type,
        "area_profile": profile,
        "traffic_light_status": traffic_light_status,
        "class_speed_limits": profile.get("speed_limits", {}),
        "area_violation_priors": profile.get("priors", {}),
    }

    if enable_stop_zone:
        stop_zone = horizontal_zone(width, height, stop_line_ratio)
        context["stop_zone_polygon"] = stop_zone
        context["red_light_zone_polygon"] = stop_zone

    if enable_no_parking_zone:
        context["no_parking_zone_polygon"] = horizontal_zone(width, height, 0.55, 1.0)

    if enable_wrong_side_zone:
        context["wrong_side_zone_polygon"] = vertical_zone(width, height, side=wrong_side)

    return context
