

def romanian_to_german(grade):
    """
    Converts Romanian Bac grade (1-10) to German Abitur grade (1-6 scale).
    Formula: German = 1 + 3 * (10 - Romanian) / 5
    """
    return round(1 + 3 * (10 - grade) / 5, 2)

def german_to_tum_points(german_grade):
    """
    Converts German Abitur grade (1-6) to TUM points (0-100 scale).
    Formula: Points = 120 - 20 * German_grade
    """
    return max(0, round(120 - 20 * german_grade))

def calculate_subject_points(math, science, german, english):
    """
    Computes weighted subject-specific score for TUM (0-100 scale).
    Uses weights: Math x3, Science x2, German x1, English x1.
    """
    total_weight = 3 + 2 + 1 + 1  # 7
    weighted_sum = (3 * math) + (2 * science) + german + english
    avg_german = weighted_sum / total_weight
    return german_to_tum_points(avg_german)

def calculate_final_score(hzb_points, subject_points, extras=0):
    """
    Computes the final admission score.
    Formula: 0.65 * HZB Points + 0.35 * Subject Points + Extras (max 6 points)
    """
    return min(100, round(0.65 * hzb_points + 0.35 * subject_points + extras))

if __name__ == "__main__":
    print("TUM Informatics Admission Calculator")
    
    # Simulated user input (replace with actual inputs if running interactively)
    bac_average = (8.9+ 9.3+ 9.5+ 8.7)/4  # Example: Romanian Bac average (1-10)
    
    math_grades = [6, 8]  # Example: Two grades (11th & 12th grade) in Math
    science_grades = [8, 8]  # Example: Two grades (11th & 12th grade) in Science/Informatics
    german_grades = [7, 8]  # Example: Two grades (11th & 12th grade) in German (or 0 if not applicable)
    english_grades = [8, 9]  # Example: Two grades (11th & 12th grade) in English
    extras = 2  # Example: Extra qualification points (0-6)
    
    # Compute semester averages
    math_avg = sum(math_grades) / len(math_grades)
    science_avg = sum(science_grades) / len(science_grades)
    german_avg = sum(german_grades) / len(german_grades) if sum(german_grades) > 0 else 4.0
    english_avg = sum(english_grades) / len(english_grades)
    
    # Convert Romanian grades to German scale and then to TUM points
    german_bac = romanian_to_german(bac_average)
    hzb_points = german_to_tum_points(german_bac)
    
    german_math = romanian_to_german(math_avg)
    german_science = romanian_to_german(science_avg)
    german_german = romanian_to_german(german_avg)
    german_english = romanian_to_german(english_avg)
    
    subject_points = calculate_subject_points(german_math, german_science, german_german, german_english)
    
    final_score = calculate_final_score(hzb_points, subject_points, extras)
    
    print(f"\nFinal TUM Admission Score: {final_score}/100")
    if final_score >= 84:
        print("✅ Direct admission! 🎉")
    elif final_score >= 72:
        print("⚠️ You need to take the second-stage test.")
    else:
        print("❌ Admission unlikely. Consider exceptions or alternative paths.")
