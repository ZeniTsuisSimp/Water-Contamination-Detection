def detect_anomaly(ph, tds):
    """
    Checks for CRITICAL safety violations that override the model.
    Subtle violations (e.g. pH 6.0) are left for the ML model to decide.
    Returns: (is_anomaly, message)
    """
    if ph < 4.0 or ph > 10.0:
        return True, f"⚠️ CRITICAL: pH {ph} is DANGEROUSLY outside safe range (4.0-10.0)!"
    if tds > 3000:
        return True, f"⚠️ CRITICAL: TDS {tds} ppm is dangerously high (>3000)!"
    return False, None
